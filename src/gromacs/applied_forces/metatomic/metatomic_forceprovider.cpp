/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2024- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * https://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at https://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */
/*! \internal \file
 * \brief
 * Implements the Metatomic Force Provider class with per-rank model evaluation.
 *
 * MTA atoms are the atoms of metatomic-input-group, the atoms the model sees.
 * In domain decomposition, every rank gathers all MTA atoms and the pairs of
 * every rank's GROMACS pairlist, so each home atom sees its whole environment,
 * including the message-passing receptive field. The model returns home-atom
 * outputs only.
 *
 * Common design points:
 *  - Forces: home forces applied directly; non-home forces exchanged via
 *    sparse indexed communication.  ForceWithVirial is not communicated
 *    by dd_move_f, so we handle it ourselves.  Dense allreduce fallback
 *    for small systems (N_total < 1000).
 *  - Ghost deduplication: periodic ghost images share the same model index
 *    but all GROMACS local indices are mapped via gmxLocalToMtaIdx_.
 *
 * \author Eric Boittier <eric.boittier@epfl.ch>
 * \author Rohit Goswami <r95g10@gmail.com>
 * \author Philip Loche <ploche@physik.fu-berlin.de>
 * \author Guillaume Fraux <guillaume.fraux@epfl.ch>
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "metatomic_forceprovider.h"

#include "config.h"

#include <cmath>
#include <cstdint>
#include <cstdio>

#include <algorithm>
#include <array>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "gromacs/domdec/localatomset.h"
#include "gromacs/mdlib/broadcaststructs.h"
#include "gromacs/mdrunutility/mdmodulesnotifiers.h"
#include "gromacs/mdrunutility/plainpairlistranges.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/mpicomm.h"
#include "gromacs/utility/strconvert.h"
#include "gromacs/utility/stringutil.h"

#include "metatomic_cellshift.h"

#include <metatensor.hpp>
#include <metatensor/dlpack/dlpack.h>
#include <metatomic.hpp>


namespace gmx
{

/*! \brief The variant of one output: its own option, else metatomic-variant.
 *
 * `off` in the option of the output selects the output without variant.
 */
static std::optional<std::string> output_variant(const std::string& variant, const std::string& outputVariant)
{
    if (!outputVariant.empty())
    {
        return outputVariant == "off" ? std::nullopt : std::optional<std::string>(outputVariant);
    }
    return variant.empty() ? std::nullopt : std::optional<std::string>(variant);
}

static std::string quantity_name(const std::string& base, const std::optional<std::string>& variant)
{
    if (variant.has_value())
    {
        return base + "/" + variant.value();
    }
    return base;
}

static const metatomic::Quantity* find_output(const std::vector<metatomic::Quantity>& outputs,
                                              const std::string&                     name)
{
    for (const auto& output : outputs)
    {
        if (output.name() == name)
        {
            return &output;
        }
    }
    return nullptr;
}

struct DlpackShape
{
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
};

static void dlpack_deleter(DLManagedTensorVersioned* self)
{
    delete static_cast<DlpackShape*>(self->manager_ctx);
    delete self;
}

//! Wrap a caller-owned contiguous CPU buffer. The deleter frees the header only.
static metatomic::DLPackTensor wrap_dlpack(void* data, std::vector<int64_t> shape, DLDataType dtype)
{
    auto* ctx   = new DlpackShape;
    ctx->shape  = std::move(shape);
    ctx->strides.resize(ctx->shape.size());
    int64_t stride = 1;
    for (int axis = static_cast<int>(ctx->shape.size()) - 1; axis >= 0; --axis)
    {
        ctx->strides[axis] = stride;
        stride *= ctx->shape[axis];
    }

    auto* tensor                 = new DLManagedTensorVersioned{};
    tensor->version.major        = DLPACK_MAJOR_VERSION;
    tensor->version.minor        = DLPACK_MINOR_VERSION;
    tensor->manager_ctx          = ctx;
    tensor->deleter              = dlpack_deleter;
    tensor->flags                = DLPACK_FLAG_BITMASK_READ_ONLY;
    tensor->dl_tensor.data       = data;
    tensor->dl_tensor.byte_offset = 0;
    tensor->dl_tensor.device     = { kDLCPU, 0 };
    tensor->dl_tensor.dtype      = dtype;
    tensor->dl_tensor.ndim       = static_cast<int32_t>(ctx->shape.size());
    tensor->dl_tensor.shape      = ctx->shape.data();
    tensor->dl_tensor.strides    = ctx->strides.data();
    return metatomic::DLPackTensor(tensor);
}

template<typename T>
static int64_t count_above(metatensor::TensorBlock& block, double threshold)
{
    auto   values = block.values<T>();
    size_t n      = 1;
    for (size_t extent : values.shape())
    {
        n *= extent;
    }
    const T* data   = values.data();
    int64_t  nAbove = 0;
    for (size_t i = 0; i < n; ++i)
    {
        if (static_cast<double>(data[i]) > threshold)
        {
            ++nAbove;
        }
    }
    return nAbove;
}

template<typename T>
static double sum_block(metatensor::TensorBlock& block)
{
    auto          values = block.values<T>();
    size_t        n      = 1;
    for (size_t extent : values.shape())
    {
        n *= extent;
    }
    const T* data = values.data();
    double   sum  = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        sum += static_cast<double>(data[i]);
    }
    return sum;
}

/*! \brief Add per-atom vectors, shape (samples, 3, 1), into \p vectors.
 *
 * Atoms are read from the \p atomColumn column of the samples.
 */
template<typename T>
static void add_atom_vectors(metatensor::TensorBlock& block, int atomColumn, int32_t nAtoms, std::vector<double>* vectors)
{
    auto       sampleLabels = block.samples();
    const auto samples      = sampleLabels.values();
    const auto values       = block.values<T>();
    for (size_t row = 0; row < sampleLabels.count(); ++row)
    {
        const int32_t atom = samples(row, static_cast<size_t>(atomColumn));
        if (atom < 0 || atom >= nAtoms)
        {
            GMX_THROW(APIError("Metatomic output names an atom outside the local set"));
        }
        for (size_t d = 0; d < 3; ++d)
        {
            (*vectors)[3 * atom + d] += static_cast<double>(values(row, d, 0));
        }
    }
}

/*! \brief Add half of the 3x3 matrices of \p block, shape (samples, 3, 3, 1), to \p virial.
 *
 * For a strain gradient dE/de this is the GROMACS virial, -0.5 sum_i x_i (x) f_i.
 * A stress s gives the same after multiplication by the volume, as dE/de = V s.
 */
template<typename T>
static void add_virial(metatensor::TensorBlock& block, matrix virial)
{
    const auto values = block.values<T>();
    for (size_t row = 0; row < block.samples().count(); ++row)
    {
        for (size_t a = 0; a < 3; ++a)
        {
            for (size_t b = 0; b < 3; ++b)
            {
                virial[a][b] += static_cast<real>(0.5 * static_cast<double>(values(row, a, b, 0)));
            }
        }
    }
}

/*! \brief Internal data structure for Metatomic runtime states. */
struct MetatomicData
{
    std::unique_ptr<metatomic::ExternalModel> model;
    metatomic::ModelCapabilities              capabilities = metatomic::ModelCapabilities::builder()
                                                      .atomic_types({})
                                                      .interaction_range(0.0)
                                                      .length_unit("nm")
                                                      .supported_devices({ metatomic::ModelCapabilities::Device::CPU })
                                                      .dtype(metatomic::ModelCapabilities::DType::Float64)
                                                      .outputs({})
                                                      .build();
    std::vector<metatomic::PairListOptions> pairLists;
    //! Each pair-list cutoff converted to nm at init.
    std::vector<double> cutoffNm;
    //! Convert model-declared cutoffs to GROMACS units.
    double              lengthToNm      = 1.0;
    bool                useFloat64      = true;
    bool                checkConsistency = false;

    std::vector<std::string> nlSampleNames = {
        "first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"
    };

    std::array<uint8_t, 3> pbc = { 1, 1, 1 };

    //! Outputs passed to execute_model, in request order, without the virial.
    std::vector<metatomic::Quantity> requested;
    /*! \brief The same outputs with what the virial needs: the strain gradient of the
     * energy, or the non-conservative stress, which comes last.
     */
    std::vector<metatomic::Quantity> requestedWithVirial;
    std::size_t                      energyIndex = 0;
    std::optional<std::size_t>       uqIndex;
    std::optional<std::size_t>       ncForceIndex;
    std::optional<std::size_t>       ncStressIndex;
    std::string                      energyName;
    std::string                      uqName;
    //! Uncertainty threshold in kJ/mol. Atoms above this trigger a warning.
    double uncertaintyThreshold = 0.0;

    //! Non-conservative mode: forces/stress are outputs, not energy gradients.
    bool nonConservative = false;

    //! Home-atom selection. Rebuilt when numHomeMta_ changes.
    std::optional<metatensor::Labels> selectedAtoms;

    //! Buffers used when GROMACS `real` is not the model dtype.
    std::vector<double> positions64;
    std::vector<float>  positions32;
    std::vector<double> cell64;
    std::vector<float>  cell32;
};

//! Plugins installed with the libmetatomic GROMACS was built with, all named *.so
static std::vector<std::filesystem::path> installedPlugins()
{
    std::vector<std::filesystem::path> plugins;
    std::error_code                    error;
    for (const auto& entry : std::filesystem::directory_iterator(GMX_METATOMIC_PLUGINS_DIR, error))
    {
        if (entry.path().extension() == ".so")
        {
            plugins.push_back(entry.path());
        }
    }
    std::sort(plugins.begin(), plugins.end());
    return plugins;
}

MetatomicForceProvider::MetatomicForceProvider(const MetatomicOptions& options,
                                               const MDLogger&         logger,
                                               const MpiComm&          mpiComm) :
    options_(options),
    logger_(logger),
    mpiComm_(mpiComm),
    box_{ { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } },
    data_(std::make_unique<MetatomicData>())
{
    try
    {
        if (options_.params_.pluginPath != "auto")
        {
            metatomic::load_plugin(options_.params_.pluginPath);
        }
        else
        {
            for (const auto& plugin : installedPlugins())
            {
                try
                {
                    metatomic::load_plugin(plugin.string());
                    GMX_LOG(logger_.info)
                            .asParagraph()
                            .appendTextFormatted("Loaded the metatomic plugin %s",
                                                 plugin.string().c_str());
                }
                catch (const std::exception& e)
                {
                    GMX_LOG(logger_.warning)
                            .asParagraph()
                            .appendTextFormatted("Could not load the metatomic plugin %s: %s",
                                                 plugin.string().c_str(),
                                                 e.what());
                }
            }
        }
        std::optional<std::string> modelOptions;
        if (!options_.params_.modelOptions.empty())
        {
            modelOptions = nlohmann::json(parseMetatomicModelOptions(options_.params_.modelOptions)).dump();
        }
        data_->model = std::make_unique<metatomic::ExternalModel>(
                metatomic::load_model(options_.params_.modelPath_, modelOptions));
        data_->capabilities = data_->model->capabilities();
        data_->lengthToNm = metatomic::unit_conversion_factor(data_->capabilities.length_unit(), "nm");
    }
    catch (const std::exception& e)
    {
        GMX_THROW(APIError("Failed to load metatomic model: " + std::string(e.what())));
    }

    data_->useFloat64 = data_->capabilities.dtype() == metatomic::ModelCapabilities::DType::Float64;
    if (data_->capabilities.dtype() != metatomic::ModelCapabilities::DType::Float64
        && data_->capabilities.dtype() != metatomic::ModelCapabilities::DType::Float32)
    {
        GMX_THROW(APIError("Unsupported dtype from model capabilities"));
    }
    data_->checkConsistency = options_.params_.checkConsistency;
    const PbcType* pbcType  = options_.params_.pbcType_.get();
    if (pbcType != nullptr && *pbcType == PbcType::Screw)
    {
        GMX_THROW(NotImplementedError("Metatomic does not support screw periodic boundary conditions."));
    }
    if (pbcType != nullptr && *pbcType == PbcType::XY)
    {
        data_->pbc = { 1, 1, 0 };
    }
    else if (pbcType != nullptr && *pbcType == PbcType::No)
    {
        data_->pbc = { 0, 0, 0 };
    }
    else
    {
        data_->pbc = { 1, 1, 1 };
    }

    GMX_LOG(logger_.info)
            .asParagraph()
            .appendTextFormatted("Metatomic model: length unit %s, dtype %s",
                                 data_->capabilities.length_unit().c_str(),
                                 data_->useFloat64 ? "float64" : "float32");

    const auto& params       = options_.params_;
    const auto& modelOutputs = data_->capabilities.outputs();
    data_->energyName        = quantity_name("energy", output_variant(params.variant, ""));
    const auto* energyCap    = find_output(modelOutputs, data_->energyName);
    if (energyCap == nullptr)
    {
        GMX_THROW(APIError(formatString(
                "The model at '%s' does not provide an '%s' output. "
                "Metatomic interface cannot proceed.",
                params.modelPath_.c_str(),
                data_->energyName.c_str())));
    }

    data_->nonConservative = params.nonConservative;
    if (!data_->nonConservative)
    {
        // The forces need the positions gradient, the virial the strain gradient.
        const auto& declared = energyCap->gradients();
        const auto  declares = [&declared](metatomic::Gradients gradient)
        { return std::find(declared.begin(), declared.end(), gradient) != declared.end(); };
        if (!declares(metatomic::Gradients::Positions) || !declares(metatomic::Gradients::Strain))
        {
            GMX_THROW(InconsistentInputError(formatString(
                    "The model's '%s' output does not declare gradients with respect to both "
                    "positions and strain, which GROMACS needs for the forces and the virial.",
                    data_->energyName.c_str())));
        }
    }
    {
        // Models may compute the strain gradient only on request, so it is only
        // requested for the steps that need the virial.
        using metatomic::Gradients;
        const auto energy = [&](std::vector<Gradients> gradients)
        {
            return metatomic::Quantity::builder()
                    .name(data_->energyName)
                    .unit("kJ/mol")
                    .sample_kind(energyCap->sample_kind())
                    .gradients(data_->nonConservative ? std::vector<Gradients>{} : gradients)
                    .build();
        };
        data_->energyIndex = data_->requested.size();
        data_->requested.push_back(energy({ Gradients::Positions }));
        data_->requestedWithVirial.push_back(energy({ Gradients::Positions, Gradients::Strain }));
    }

    if (params.uncertaintyThreshold != "off")
    {
        const auto  uqName = quantity_name("energy_uncertainty",
                                          output_variant(params.variant, params.variantEnergyUq));
        const auto* uqCap  = find_output(modelOutputs, uqName);
        if (uqCap != nullptr && uqCap->sample_kind() == metatomic::SampleKind::Atom)
        {
            data_->uqName  = uqName;
            data_->uqIndex = data_->requested.size();
            data_->requested.push_back(metatomic::Quantity::builder()
                                               .name(uqName)
                                               .unit("kJ/mol")
                                               .sample_kind(metatomic::SampleKind::Atom)
                                               .build());
            data_->requestedWithVirial.push_back(data_->requested.back());
            data_->uncertaintyThreshold = doubleFromString(params.uncertaintyThreshold.c_str());
            GMX_LOG(logger_.info)
                    .asParagraph()
                    .appendTextFormatted(
                            "Metatomic: found '%s' output, will check for atoms with "
                            "high uncertainty (threshold: %.4f kJ/mol)",
                            uqName.c_str(),
                            data_->uncertaintyThreshold);
        }
    }

    if (data_->nonConservative)
    {
        // non_conservative_force is always per atom
        const auto forceName = quantity_name(
                "non_conservative_force", output_variant(params.variant, params.variantNcForces));
        if (find_output(modelOutputs, forceName) == nullptr)
        {
            GMX_THROW(APIError(formatString(
                    "The model does not provide '%s' output, "
                    "we can not enable non-conservative simulations",
                    forceName.c_str())));
        }
        data_->ncForceIndex = data_->requested.size();
        data_->requested.push_back(metatomic::Quantity::builder()
                                           .name(forceName)
                                           .unit("kJ/mol/nm")
                                           .sample_kind(metatomic::SampleKind::Atom)
                                           .build());
        data_->requestedWithVirial.push_back(data_->requested.back());

        const auto stressName = quantity_name(
                "non_conservative_stress", output_variant(params.variant, params.variantNcStress));
        if (find_output(modelOutputs, stressName) != nullptr)
        {
            data_->ncStressIndex = data_->requestedWithVirial.size();
            data_->requestedWithVirial.push_back(metatomic::Quantity::builder()
                                                         .name(stressName)
                                                         .unit("kJ/mol/nm^3")
                                                         .sample_kind(metatomic::SampleKind::System)
                                                         .build());
        }
        GMX_LOG(logger_.info)
                .asParagraph()
                .appendTextFormatted(
                        "Metatomic: non-conservative mode enabled. Forces from '%s'%s",
                        forceName.c_str(),
                        data_->ncStressIndex.has_value() ? ", with stress" : "");
    }

    // The pair lists and extra inputs a model needs depend on the outputs it computes,
    // asked here for the outputs of the virial steps, which include the others.
    std::vector<metatomic::Quantity> inputs;
    try
    {
        data_->pairLists = data_->model->requested_pair_lists(data_->requestedWithVirial);
        inputs           = data_->model->requested_inputs(data_->requestedWithVirial);
    }
    catch (const std::exception& e)
    {
        GMX_THROW(APIError("Failed to query the metatomic model: " + std::string(e.what())));
    }
    if (!inputs.empty())
    {
        GMX_THROW(APIError(formatString(
                "The model requests the input '%s', which GROMACS can not provide.",
                inputs.front().name().c_str())));
    }
    data_->cutoffNm.reserve(data_->pairLists.size());
    for (const auto& pairs : data_->pairLists)
    {
        data_->cutoffNm.push_back(pairs.cutoff() * data_->lengthToNm);
    }

    GMX_LOG(logger_.info)
            .asParagraph()
            .appendText("MetatomicForceProvider initialization complete.");
}

MetatomicForceProvider::~MetatomicForceProvider() = default;

real MetatomicForceProvider::plainPairlistRange(const PlainPairlistRanges& ranges) const
{
    const double interactionRange = data_->capabilities.interaction_range() * data_->lengthToNm;
    if (interactionRange < 0.0)
    {
        GMX_THROW(InconsistentInputError("interaction_range is negative for this model."));
    }
    if (!std::isfinite(interactionRange) && mpiComm_.isParallel())
    {
        GMX_THROW(NotImplementedError(
                "interaction_range is infinite for this model; "
                "using multiple MPI domains is not supported."));
    }

    // The model only reads pairs within its pair-list cutoffs; a longer interaction range
    // comes from message passing over them. So the plain pairlist needs the largest
    // cutoff plus a buffer for the motion between pairlist updates, not the interaction
    // range (which would force rlist up to it).
    const double pairCutoff =
            data_->cutoffNm.empty() ? 0.0 : *std::max_element(data_->cutoffNm.begin(), data_->cutoffNm.end());
    if (pairCutoff == 0.0)
    {
        // The model reads no pairs.
        return 0.0;
    }

    // GROMACS builds the plain pairlist on search steps only, at the requested range, and
    // leaves the buffer for the motion until the next search to the requester. Twice the
    // RMS displacement over a pairlist lifetime, as PlainPairlistRanges::rmsdDistance()
    // recommends, and at least 0.1 nm for fast ML hydrogens. The normal pairlist holds no
    // pairs beyond rlist, so the buffer ends there; the cutoff itself must fit (mdrun
    // stops otherwise).
    const double rmsd  = ranges.rmsdDistance().value_or(0.0);
    const double range = std::max(
            pairCutoff, std::min(pairCutoff + std::max(2 * rmsd, 0.1), static_cast<double>(ranges.pairlistCutoff())));
    GMX_LOG(logger_.info)
            .asParagraph()
            .appendTextFormatted(
                    "Metatomic: plain pairlist range %.3f nm (pair-list cutoff %.3f nm + %.3f nm "
                    "buffer; interaction range %.3f nm)",
                    range,
                    pairCutoff,
                    range - pairCutoff,
                    interactionRange);
    return range;
}

/*! \brief Rebuild local MTA atom tables after domain decomposition.
 *
 * Called on every AtomsRedistributed signal. Scans the local + halo atoms
 * to find MTA atoms, deduplicates periodic ghost images, and builds:
 *  - mtaToGmxLocal_: model index -> GROMACS local buffer index
 *  - mtaToGlobalMta_: model index -> global MTA index (for force all-reduce)
 *  - gmxLocalToMtaIdx_: GROMACS local index -> model index (all images)
 *  - atomNumbers_: atomic numbers for the model input
 *
 * Home atoms get model indices [0, numHomeMta_), halo atoms get
 * [numHomeMta_, numLocalMta_).
 */
void MetatomicForceProvider::gatherAtomNumbersIndices(const MDModulesAtomsRedistributedSignal& signal)
{
    const auto&   mtaIndices  = options_.params_.mtaIndices_;
    const int32_t numTotalMta = static_cast<int32_t>(mtaIndices.size());

    mtaToGmxLocal_.clear();
    mtaToGlobalMta_.clear();
    atomNumbers_.clear();
    gmxLocalToMtaIdx_.clear();
    globalMtaToLocalHome_.clear();

    if (mpiComm_.isParallel())
    {
        GMX_RELEASE_ASSERT(signal.globalAtomIndices_.has_value(),
                           "Global atom indices required for domain decomposition.");
        auto          globalAtomIndices = signal.globalAtomIndices_.value();
        const int32_t numLocal          = signal.x_.size();
        const int32_t numLocalPlusHalo  = globalAtomIndices.size();

        // Build a map from global atom index to MTA index for fast lookup
        std::unordered_map<int32_t, int32_t> globalToMtaIdx;
        for (int32_t j = 0; j < numTotalMta; j++)
        {
            globalToMtaIdx[static_cast<int32_t>(mtaIndices[j])] = j;
        }

        // Separate home and halo MTA atoms, deduplicating periodic ghosts.
        // Each unique MTA atom gets one model index. Periodic ghost images
        // are NOT added as separate model atoms, but their GROMACS local
        // buffer indices ARE recorded in gmxLocalToMtaIdx_ so that
        // setPairlist can resolve pairlist entries referencing any image.
        std::vector<int32_t> homeGmxLocal;
        std::vector<int32_t> homeGlobalMta;
        std::vector<int32_t> haloGmxLocal;
        std::vector<int32_t> haloGlobalMta;

        // First pass: assign model indices to unique MTA atoms
        std::unordered_map<int32_t, int32_t> mtaIdxToModelIdx;

        for (int32_t i = 0; i < numLocalPlusHalo; i++)
        {
            int32_t globalIdx = globalAtomIndices[i];
            auto    it        = globalToMtaIdx.find(globalIdx);
            if (it != globalToMtaIdx.end())
            {
                int32_t mtaIdx = it->second;

                // A periodic ghost of an atom seen already gets no model atom of its own;
                // its local index is mapped in the second pass.
                if (mtaIdxToModelIdx.count(mtaIdx) == 0)
                {
                    if (i < numLocal)
                    {
                        // Will be assigned model index = homeGmxLocal.size() (filled later)
                        homeGmxLocal.push_back(i);
                        homeGlobalMta.push_back(mtaIdx);
                    }
                    else
                    {
                        haloGmxLocal.push_back(i);
                        haloGlobalMta.push_back(mtaIdx);
                    }
                    // Placeholder: model index will be set after we know numHomeMta_
                    mtaIdxToModelIdx[mtaIdx] = -1;
                }
            }
        }

        // Assign final model indices: home [0, numHome), halo [numHome, numLocal)
        int32_t modelIdx = 0;
        for (int32_t k = 0; k < static_cast<int32_t>(homeGmxLocal.size()); k++)
        {
            mtaIdxToModelIdx[homeGlobalMta[k]] = modelIdx++;
        }
        for (int32_t k = 0; k < static_cast<int32_t>(haloGmxLocal.size()); k++)
        {
            mtaIdxToModelIdx[haloGlobalMta[k]] = modelIdx++;
        }

        // Second pass: build complete gmxLocal → modelIdx mapping for ALL images.
        // We use a vector for O(1) direct lookup instead of a map.
        gmxLocalToMtaIdx_.assign(numLocalPlusHalo, -1);
        for (int32_t i = 0; i < numLocalPlusHalo; i++)
        {
            int32_t globalIdx = globalAtomIndices[i];
            auto    it        = globalToMtaIdx.find(globalIdx);
            if (it != globalToMtaIdx.end())
            {
                gmxLocalToMtaIdx_[i] = mtaIdxToModelIdx[it->second];
            }
        }

        numHomeMta_  = static_cast<int32_t>(homeGmxLocal.size());
        numLocalMta_ = numHomeMta_ + static_cast<int32_t>(haloGmxLocal.size());

        // Assign local model indices: home -> [0, numHomeMta_), halo -> [numHomeMta_, numLocalMta_)
        mtaToGmxLocal_.resize(numLocalMta_);
        mtaToGlobalMta_.resize(numLocalMta_);
        atomNumbers_.resize(numLocalMta_);

        for (int32_t k = 0; k < numHomeMta_; k++)
        {
            int32_t gmxLocal  = homeGmxLocal[k];
            int32_t globalIdx = globalAtomIndices[gmxLocal];

            mtaToGmxLocal_[k]  = gmxLocal;
            mtaToGlobalMta_[k] = homeGlobalMta[k];
            atomNumbers_[k]    = options_.params_.atoms_.atom[globalIdx].atomnumber;
        }

        for (int32_t k = 0; k < static_cast<int32_t>(haloGmxLocal.size()); k++)
        {
            int32_t haloModelIdx = numHomeMta_ + k;
            int32_t gmxLocal     = haloGmxLocal[k];
            int32_t globalIdx    = globalAtomIndices[gmxLocal];

            mtaToGmxLocal_[haloModelIdx]  = gmxLocal;
            mtaToGlobalMta_[haloModelIdx] = haloGlobalMta[k];
            atomNumbers_[haloModelIdx]    = options_.params_.atoms_.atom[globalIdx].atomnumber;
        }
    }
    else
    {
        // Serial / thread-MPI: all MTA atoms are home, no halos
        const auto* mtaAtoms = options_.params_.mtaAtoms_.get();
        numHomeMta_  = numTotalMta;
        numLocalMta_ = numTotalMta;

        mtaToGmxLocal_.resize(numTotalMta);
        mtaToGlobalMta_.resize(numTotalMta);
        atomNumbers_.resize(numTotalMta);
        gmxLocalToMtaIdx_.assign(signal.x_.size(), -1);

        for (int32_t i = 0; i < numTotalMta; i++)
        {
            int32_t localIndex = mtaAtoms->localIndex()[i];
            int32_t globalIdx  = mtaAtoms->globalIndex()[mtaAtoms->collectiveIndex()[i]];

            mtaToGmxLocal_[i]       = localIndex;
            mtaToGlobalMta_[i]      = i;
            atomNumbers_[i]         = options_.params_.atoms_.atom[globalIdx].atomnumber;
            gmxLocalToMtaIdx_[localIndex] = i;
        }
    }

    // Build reverse map: global MTA index -> local home model index.
    // Used by sparse force distribution to route incoming forces to home atoms.
    globalMtaToLocalHome_.reserve(numHomeMta_);
    for (int32_t i = 0; i < numHomeMta_; i++)
    {
        globalMtaToLocalHome_[mtaToGlobalMta_[i]] = i;
    }

    GMX_RELEASE_ASSERT(std::count(atomNumbers_.begin(), atomNumbers_.end(), 0) == 0,
                       "Some atom numbers not set.");
}

void MetatomicForceProvider::gatherAtomPositions(ArrayRef<const RVec> pos)
{
    positions_.resize(numLocalMta_);
    for (int32_t i = 0; i < numLocalMta_; i++)
    {
        positions_[i] = pos[mtaToGmxLocal_[i]];
    }
}

/*! \brief Convert GROMACS pairlists to MTA model indices.
 *
 * Called on every PairlistConstructed signal. Maps GROMACS local buffer
 * indices in the plain interacting pairlist and the excluded pairlist to
 * MTA model indices via gmxLocalToMtaIdx_, and negates cell shifts
 * (GROMACS shifts the first atom, metatensor shifts the second).
 *
 * The model needs every pair within its cutoff, excluded or not. grompp
 * excludes the classical non-bonded interactions among MTA atoms, so their
 * pairs come in the excluded list; both lists are read so that no pair is
 * lost, whatever the exclusions.
 */
void MetatomicForceProvider::setPairlist(const MDModulesPairlistConstructedSignal& signal)
{
    pairlistMta_.clear();
    cellShiftsMta_.clear();

    // Use gmxLocalToMtaIdx_ which maps ALL GROMACS local buffer indices
    // (including periodic ghost images) to their MTA model index.
    //
    // Sign convention: GROMACS shifts atom I (first): d = x[I]+shift - x[J].
    // Metatensor shifts atom J (second): r_ij = x[J]+cell·box - x[I].
    // So metatensor cell shift = -GROMACS cell shift.
    const auto appendPairlistEntries = [this](const auto& pairlistEntries) {
        for (const auto& entry : pairlistEntries)
        {
            const auto& [atomPair, shiftIndex] = entry;
            const int32_t idxA = gmxLocalToMtaIdx_[atomPair.first];
            const int32_t idxB = gmxLocalToMtaIdx_[atomPair.second];

            if (idxA != -1 && idxB != -1)
            {
                pairlistMta_.push_back(idxA);
                pairlistMta_.push_back(idxB);
                const IVec gmxShift = shiftIndexToXYZ(shiftIndex);
                cellShiftsMta_.push_back(IVec(-gmxShift[XX], -gmxShift[YY], -gmxShift[ZZ]));
            }
        }
    };
    appendPairlistEntries(signal.pairlist_);
    appendPairlistEntries(signal.excludedPairlist_);
}


int32_t MetatomicForceProvider::gatherAllMtaAtoms()
{
    const auto&   mtaIndices  = options_.params_.mtaIndices_;
    const int32_t numTotalMta = static_cast<int32_t>(mtaIndices.size());

    // Home atoms are owned by exactly one rank, so a sum of the home positions scattered
    // into a dense array gives all positions.
    std::vector<double> allPositions(3 * numTotalMta, 0.0);
    for (int32_t i = 0; i < numHomeMta_; i++)
    {
        const int32_t g = mtaToGlobalMta_[i];
        for (int d = 0; d < 3; d++)
        {
            allPositions[3 * g + d] = positions_[i][d];
        }
    }
    mpiComm_.sumReduce(static_cast<std::size_t>(3 * numTotalMta), allPositions.data());

    std::vector<bool> isLocal(numTotalMta, false);
    for (int32_t i = 0; i < numLocalMta_; i++)
    {
        isLocal[mtaToGlobalMta_[i]] = true;
    }

    int32_t numAdded = 0;
    for (int32_t g = 0; g < numTotalMta; g++)
    {
        if (isLocal[g])
        {
            continue;
        }
        positions_.push_back(RVec(static_cast<real>(allPositions[3 * g]),
                                  static_cast<real>(allPositions[3 * g + 1]),
                                  static_cast<real>(allPositions[3 * g + 2])));
        atomNumbers_.push_back(options_.params_.atoms_.atom[mtaIndices[g]].atomnumber);
        mtaToGlobalMta_.push_back(g);
        numAdded++;
    }
    numLocalMta_ += numAdded;
    return numAdded;
}


void MetatomicForceProvider::exchangeBackwardPairs(const matrix box, int maxRounds)
{
    backwardPairsMta_.clear();
    backwardShiftsMta_.clear();

    if (!mpiComm_.isParallel())
    {
        return;
    }

    const int nMyPairs = static_cast<int>(pairlistMta_.size() / 2);

    // Step 1: Pack local pairs as flat (globalI, globalJ) buffer.
    std::vector<int> myPairsBuf(2 * nMyPairs);
    for (int k = 0; k < nMyPairs; k++)
    {
        myPairsBuf[2 * k]     = mtaToGlobalMta_[pairlistMta_[2 * k]];
        myPairsBuf[2 * k + 1] = mtaToGlobalMta_[pairlistMta_[2 * k + 1]];
    }

    // Step 2: Existing canonical pairs — O(1) lookup via hashed set.
    struct PairHash
    {
        std::size_t operator()(const std::pair<int32_t, int32_t>& p) const
        {
            // Combine the two 32-bit ints into one 64-bit value for a perfect hash.
            return std::hash<int64_t>()(static_cast<int64_t>(p.first) << 32
                                        | static_cast<uint32_t>(p.second));
        }
    };
    std::unordered_set<std::pair<int32_t, int32_t>, PairHash> existingCanonical;
    existingCanonical.reserve(nMyPairs);
    for (int k = 0; k < nMyPairs; k++)
    {
        const int32_t gI = myPairsBuf[2 * k];
        const int32_t gJ = myPairsBuf[2 * k + 1];
        existingCanonical.insert({ std::min(gI, gJ), std::max(gI, gJ) });
    }

    // Step 3: Global MTA → local index mapping.
    std::unordered_map<int32_t, int32_t> globalToLocal;
    globalToLocal.reserve(numLocalMta_);
    for (int32_t i = 0; i < numLocalMta_; i++)
    {
        globalToLocal[mtaToGlobalMta_[i]] = i;
    }

    // Step 4: Ring exchange — P-1 rounds of MPI_Sendrecv.
    // Each round: send our pairs to rank+1, receive from rank-1.
#if GMX_MPI
    const int numRanks = mpiComm_.size();
    const int myRank   = mpiComm_.rank();
    const int sendTo   = (myRank + 1) % numRanks;
    const int recvFrom = (myRank - 1 + numRanks) % numRanks;

    std::vector<int> sendBuf = myPairsBuf;
    std::vector<int> recvBuf;

    for (int round = 0; round < maxRounds; round++)
    {
        // Exchange counts first so receiver knows buffer size.
        int sendCount = static_cast<int>(sendBuf.size());
        int recvCount = 0;
        MPI_Sendrecv(&sendCount, 1, MPI_INT, sendTo, 0,
                     &recvCount, 1, MPI_INT, recvFrom, 0,
                     mpiComm_.comm(), MPI_STATUS_IGNORE);

        recvBuf.resize(recvCount);
        MPI_Sendrecv(sendBuf.data(), sendCount, MPI_INT, sendTo, 1,
                     recvBuf.data(), recvCount, MPI_INT, recvFrom, 1,
                     mpiComm_.comm(), MPI_STATUS_IGNORE);

        // Keep the received pairs this rank does not have yet.
        const int nRecvPairs = recvCount / 2;
        for (int k = 0; k < nRecvPairs; k++)
        {
            const int32_t gI = recvBuf[2 * k];
            const int32_t gJ = recvBuf[2 * k + 1];

            // Keep pairs between two non-home atoms too: the environment of a non-home
            // neighbor enters the home atom energy through message passing.
            auto canonical = std::make_pair(std::min(gI, gJ), std::max(gI, gJ));
            if (existingCanonical.count(canonical) > 0)
            {
                continue;
            }

            // Both atoms must be in the local position array.
            auto itI = globalToLocal.find(gI);
            auto itJ = globalToLocal.find(gJ);
            if (itI == globalToLocal.end() || itJ == globalToLocal.end())
            {
                continue;
            }

            const int32_t localI = itI->second;
            const int32_t localJ = itJ->second;

            // Minimum-image shift along the periodic directions only: none along z
            // with pbc = xy, none at all with pbc = no.
            const IVec shift = minimumImageCellShift(
                    box, data_->pbc, positions_[localJ] - positions_[localI]);

            backwardPairsMta_.push_back(localI);
            backwardPairsMta_.push_back(localJ);
            backwardShiftsMta_.push_back(shift);
            existingCanonical.insert(canonical);
        }

        // Forward received buffer for the next round.
        sendBuf.swap(recvBuf);
    }
#else
    GMX_UNUSED_VALUE(box);
    GMX_UNUSED_VALUE(maxRounds);
    GMX_RELEASE_ASSERT(false, "Parallel metatomic evaluation needs an MPI build");
#endif
}


void MetatomicForceProvider::distributeNonHomeForces(const double*        forces,
                                                     ForceProviderOutput* outputs)
{
    const int32_t numTotalMta = static_cast<int32_t>(options_.params_.mtaIndices_.size());

    // A dense all-reduce is cheaper than the indexed exchange for small systems.
    if (numTotalMta < 1000)
    {
        // Dense fallback: allocate N_total buffer, scatter, allreduce, readback.
        std::vector<double> denseForces(3 * numTotalMta, 0.0);
        for (int32_t i = 0; i < numLocalMta_; i++)
        {
            int32_t g = mtaToGlobalMta_[i];
            denseForces[3 * g]     = forces[3 * i];
            denseForces[3 * g + 1] = forces[3 * i + 1];
            denseForces[3 * g + 2] = forces[3 * i + 2];
        }
        mpiComm_.sumReduce(static_cast<std::size_t>(3 * numTotalMta), denseForces.data());

        for (int32_t i = 0; i < numHomeMta_; i++)
        {
            int32_t gmxIdx = mtaToGmxLocal_[i];
            int32_t g      = mtaToGlobalMta_[i];
            outputs->forceWithVirial_.force_[gmxIdx][0] += static_cast<real>(denseForces[3 * g]);
            outputs->forceWithVirial_.force_[gmxIdx][1] += static_cast<real>(denseForces[3 * g + 1]);
            outputs->forceWithVirial_.force_[gmxIdx][2] += static_cast<real>(denseForces[3 * g + 2]);
        }
        return;
    }

    // Sparse path: apply home forces directly, exchange only non-home forces.

    // Step 1: Apply home atom forces directly (no communication needed).
    for (int32_t i = 0; i < numHomeMta_; i++)
    {
        int32_t gmxIdx = mtaToGmxLocal_[i];
        outputs->forceWithVirial_.force_[gmxIdx][0] += static_cast<real>(forces[3 * i]);
        outputs->forceWithVirial_.force_[gmxIdx][1] += static_cast<real>(forces[3 * i + 1]);
        outputs->forceWithVirial_.force_[gmxIdx][2] += static_cast<real>(forces[3 * i + 2]);
    }

    // Step 2: Pack non-home forces as sparse tuples (globalMtaIdx, fx, fy, fz).
    // Each tuple is 4 doubles: [globalMtaIdx_as_double, fx, fy, fz].
    const int32_t numNonHome = numLocalMta_ - numHomeMta_;
    std::vector<double> sendBuf(4 * numNonHome);
    for (int32_t i = numHomeMta_; i < numLocalMta_; i++)
    {
        int32_t k = i - numHomeMta_;
        sendBuf[4 * k]     = static_cast<double>(mtaToGlobalMta_[i]);
        sendBuf[4 * k + 1] = forces[3 * i];
        sendBuf[4 * k + 2] = forces[3 * i + 1];
        sendBuf[4 * k + 3] = forces[3 * i + 2];
    }

    // Step 3: Exchange counts via allreduce on a P-element array.
    const int numRanks = mpiComm_.size();
    std::vector<int> counts(numRanks, 0);
    counts[mpiComm_.rank()] = numNonHome;
    mpiComm_.sumReduce(ArrayRef<int>(counts));

    // Step 4: Allgatherv via Gatherv + Bcast (thread-MPI compatible).
    int totalNonHome = 0;
    std::vector<int> displs(numRanks);
    for (int r = 0; r < numRanks; r++)
    {
        displs[r] = totalNonHome;
        totalNonHome += counts[r];
    }

    // Scale counts/displs to doubles (4 per tuple)
    std::vector<int> dcounts(numRanks), ddispls(numRanks);
    for (int r = 0; r < numRanks; r++)
    {
        dcounts[r] = 4 * counts[r];
        ddispls[r] = 4 * displs[r];
    }

    std::vector<double> recvBuf(4 * totalNonHome);
#if GMX_MPI
    MPI_Gatherv(sendBuf.data(), 4 * numNonHome, MPI_DOUBLE,
                recvBuf.data(), dcounts.data(), ddispls.data(), MPI_DOUBLE,
                mpiComm_.mainRank(), mpiComm_.comm());
    MPI_Bcast(recvBuf.data(), 4 * totalNonHome, MPI_DOUBLE,
              mpiComm_.mainRank(), mpiComm_.comm());
#else
    GMX_RELEASE_ASSERT(false, "Parallel metatomic evaluation needs an MPI build");
#endif

    // Step 5: Scan received tuples for forces destined for our home atoms.
    for (int t = 0; t < totalNonHome; t++)
    {
        int32_t globalMtaIdx = static_cast<int32_t>(recvBuf[4 * t]);
        auto    it           = globalMtaToLocalHome_.find(globalMtaIdx);
        if (it != globalMtaToLocalHome_.end())
        {
            int32_t localHomeIdx = it->second;
            int32_t gmxIdx       = mtaToGmxLocal_[localHomeIdx];
            outputs->forceWithVirial_.force_[gmxIdx][0] += static_cast<real>(recvBuf[4 * t + 1]);
            outputs->forceWithVirial_.force_[gmxIdx][1] += static_cast<real>(recvBuf[4 * t + 2]);
            outputs->forceWithVirial_.force_[gmxIdx][2] += static_cast<real>(recvBuf[4 * t + 3]);
        }
    }
}


void MetatomicForceProvider::calculateForces(const ForceProviderInput& inputs, ForceProviderOutput* outputs)
{
    gatherAtomPositions(inputs.x_);
    copy_mat(inputs.box_, box_);
    // metatomic requires zero cell vectors along nonperiodic directions.
    // Keep inputs.box_ unchanged for GROMACS pair shifts and volume calculations.
    for (int axis = 0; axis < 3; ++axis)
    {
        if (!data_->pbc[axis])
        {
            clear_rvec(box_[axis]);
        }
    }

    // With domain decomposition every rank needs, for each home atom, all MTA atoms within
    // the model's interaction range and the pairs among them (message passing), not only
    // the pairs the eighth-shell decomposition assigned to it. The GROMACS pairlist is the
    // pair source; atoms and pairs are then exchanged across ranks.
    const bool parallel = mpiComm_.isParallel();

    // The model rows appended below are removed after the evaluation, so that
    // gatherAtomPositions on the next step only sees the rows mtaToGmxLocal_ maps.
    const int32_t origNumLocalMta = numLocalMta_;

    // Save original pairlist size; backward pairs are appended temporarily.
    const std::size_t origPairlistSize = pairlistMta_.size();
    const std::size_t origShiftsSize   = cellShiftsMta_.size();

    if (parallel)
    {
        // Step 1: Make all MTA atoms local. The DD halo covers only the forward direction,
        // while each home atom energy depends on all atoms within the model interaction
        // range. Extends positions_, atomNumbers_, mtaToGlobalMta_ and numLocalMta_.
        gatherAllMtaAtoms();

        // Step 2: Collect the pairs of all other ranks' pairlists, including pairs between
        // two non-home atoms. Every pair within the plain pairlist range is in exactly one
        // rank's list, so a full ring gives every rank the complete list.
        exchangeBackwardPairs(inputs.box_, mpiComm_.size() - 1);

        // Step 3: Temporarily extend pairlistMta_ with backward pairs
        // so the NL building loop processes all pairs in one pass.
        pairlistMta_.insert(pairlistMta_.end(),
                            backwardPairsMta_.begin(),
                            backwardPairsMta_.end());
        cellShiftsMta_.insert(cellShiftsMta_.end(),
                              backwardShiftsMta_.begin(),
                              backwardShiftsMta_.end());
    }

    double energy = 0.0;
    matrix virialMatrix;
    clear_mat(virialMatrix);

    {
        const int32_t       nAtoms       = numLocalMta_;
        const bool          modelIsDouble = data_->useFloat64;
        const DLDataType    floatType    = modelIsDouble ? DLDataType{ kDLFloat, 64, 1 }
                                                         : DLDataType{ kDLFloat, 32, 1 };
        const bool          gromacsIsDouble = std::is_same_v<real, double>;
        std::vector<int32_t> types(atomNumbers_.begin(), atomNumbers_.begin() + nAtoms);

        void* positionData = nullptr;
        void* cellData     = nullptr;
        if (modelIsDouble == gromacsIsDouble)
        {
            positionData = positions_.data();
            cellData     = &box_[0][0];
        }
        else
        {
            const auto fill = [&](auto& positions, auto& cell)
            {
                using T = typename std::decay_t<decltype(positions)>::value_type;
                positions.resize(static_cast<size_t>(nAtoms) * 3);
                cell.resize(9);
                for (int32_t i = 0; i < nAtoms; ++i)
                {
                    for (int d = 0; d < 3; ++d)
                    {
                        positions[3 * i + d] = static_cast<T>(positions_[i][d]);
                    }
                }
                for (int a = 0; a < 3; ++a)
                {
                    for (int b = 0; b < 3; ++b)
                    {
                        cell[3 * a + b] = static_cast<T>(box_[a][b]);
                    }
                }
                positionData = positions.data();
                cellData     = cell.data();
            };
            if (modelIsDouble)
            {
                fill(data_->positions64, data_->cell64);
            }
            else
            {
                fill(data_->positions32, data_->cell32);
            }
        }

        auto system = metatomic::System(
                "nm",
                wrap_dlpack(types.data(), { nAtoms }, { kDLInt, 32, 1 }),
                wrap_dlpack(positionData, { nAtoms, 3 }, floatType),
                wrap_dlpack(cellData, { 3, 3 }, floatType),
                wrap_dlpack(data_->pbc.data(), { 3 }, { kDLBool, 8, 1 }));

        for (size_t requestIndex = 0; requestIndex < data_->pairLists.size(); ++requestIndex)
        {
            const auto&   request  = data_->pairLists[requestIndex];
            const bool    full     = request.full_list();
            const bool    strict   = request.strict();
            const double  cutoff2  = data_->cutoffNm[requestIndex] * data_->cutoffNm[requestIndex];
            const int64_t nHalf    = static_cast<int64_t>(pairlistMta_.size() / 2);
            const int64_t maxPairs = full ? 2 * nHalf : nHalf;
            nlSamplesBuffer_.resize(static_cast<size_t>(maxPairs) * 5);

            // Pair vectors use the system length unit (nm), in the dtype of the model. They are
            // written straight into the array the pair block takes over.
            const auto pairValues = [&](auto zero) -> std::unique_ptr<metatensor::DataArrayBase>
            {
                using T = decltype(zero);
                std::vector<T> vectors(static_cast<size_t>(maxPairs) * 3);
                int64_t        nPairs = 0;
                const auto     addPair = [&](int32_t i, int32_t j, const IVec& n, double sign, const double* d)
                {
                    const int32_t sample[5] = { i, j, n[XX], n[YY], n[ZZ] };
                    std::copy(sample, sample + 5, nlSamplesBuffer_.begin() + 5 * nPairs);
                    for (int a = 0; a < 3; ++a)
                    {
                        vectors[3 * nPairs + a] = static_cast<T>(sign * d[a]);
                    }
                    ++nPairs;
                };
                for (int64_t k = 0; k < nHalf; ++k)
                {
                    const int32_t ai    = pairlistMta_[2 * k];
                    const int32_t aj    = pairlistMta_[2 * k + 1];
                    const IVec&   n     = cellShiftsMta_[k];
                    const RVec    shift = cellShiftVector(inputs.box_, n);
                    double        d[3];
                    for (int a = 0; a < 3; ++a)
                    {
                        d[a] = static_cast<double>(positions_[aj][a] - positions_[ai][a] + shift[a]);
                    }
                    if (strict && d[0] * d[0] + d[1] * d[1] + d[2] * d[2] > cutoff2)
                    {
                        continue;
                    }
                    addPair(ai, aj, n, 1.0, d);
                    if (full)
                    {
                        addPair(aj, ai, IVec(-n[XX], -n[YY], -n[ZZ]), -1.0, d);
                    }
                }
                nlSamplesBuffer_.resize(static_cast<size_t>(nPairs) * 5);
                vectors.resize(static_cast<size_t>(nPairs) * 3);
                return std::make_unique<metatensor::SimpleDataArray<T>>(
                        std::vector<uintptr_t>{ static_cast<uintptr_t>(nPairs), 3, 1 }, std::move(vectors));
            };
            auto values = modelIsDouble ? pairValues(0.0) : pairValues(0.0F);

            const size_t nPairs  = nlSamplesBuffer_.size() / 5;
            auto         samples = metatensor::Labels(
                    data_->nlSampleNames, nPairs == 0 ? nullptr : nlSamplesBuffer_.data(), nPairs);
            const int32_t                   xyzValues[3]  = { 0, 1, 2 };
            const int32_t                   distanceValue = 0;
            std::vector<metatensor::Labels> components;
            components.push_back(metatensor::Labels({ "xyz" }, xyzValues, 3));
            auto properties = metatensor::Labels({ "distance" }, &distanceValue, 1);
            system.add_pairs(request,
                             metatensor::TensorBlock(std::move(values), samples, components, properties));
        }

        std::optional<metatensor::Labels> selected;
        if (parallel)
        {
            if (!data_->selectedAtoms.has_value()
                || data_->selectedAtoms->count() != static_cast<size_t>(numHomeMta_))
            {
                std::vector<int32_t> rows(static_cast<size_t>(numHomeMta_) * 2);
                for (int32_t atom = 0; atom < numHomeMta_; ++atom)
                {
                    rows[2 * atom]     = 0;
                    rows[2 * atom + 1] = atom;
                }
                data_->selectedAtoms = metatensor::Labels(
                        { "system", "atom" },
                        numHomeMta_ == 0 ? nullptr : rows.data(),
                        static_cast<size_t>(numHomeMta_));
            }
            selected = data_->selectedAtoms;
        }

        const bool computeVirial = outputs->forceWithVirial_.computeVirial_;
        std::vector<metatomic::System> systems;
        systems.push_back(std::move(system));
        std::vector<metatensor::TensorMap> modelOutputs;
        try
        {
            modelOutputs = metatomic::execute_model(*data_->model,
                                                    systems,
                                                    selected,
                                                    computeVirial ? data_->requestedWithVirial
                                                                  : data_->requested,
                                                    data_->checkConsistency);
        }
        catch (const std::exception& e)
        {
            GMX_THROW(APIError("[Metatomic] Model evaluation failed: " + std::string(e.what())));
        }

        if (data_->uqIndex.has_value())
        {
            auto&      uqMap   = modelOutputs[data_->uqIndex.value()];
            const auto nBlocks = uqMap.keys().count();
            int64_t    nAbove  = 0;
            for (size_t blockId = 0; blockId < nBlocks; ++blockId)
            {
                auto block = uqMap.block_by_id(blockId);
                nAbove += modelIsDouble ? count_above<double>(block, data_->uncertaintyThreshold)
                                        : count_above<float>(block, data_->uncertaintyThreshold);
            }
            if (nAbove > 0)
            {
                GMX_LOG(logger_.warning)
                        .asParagraph()
                        .appendTextFormatted(
                                "Metatomic: uncertainty on atomic energies for %ld atoms "
                                "is larger than the threshold of %.4f kJ/mol. "
                                "Consider retraining the model.",
                                static_cast<long>(nAbove),
                                data_->uncertaintyThreshold);
            }
        }

        // Model gradients come in the system's nm and the requested kJ/mol.
        auto&               energyMap = modelOutputs[data_->energyIndex];
        std::vector<double> forces(static_cast<size_t>(nAtoms) * 3, 0.0);
        for (size_t blockId = 0; blockId < energyMap.keys().count(); ++blockId)
        {
            auto block = energyMap.block_by_id(blockId);
            energy += modelIsDouble ? sum_block<double>(block) : sum_block<float>(block);
            if (data_->nonConservative)
            {
                continue;
            }
            const auto gradients = block.gradients_list();
            const auto has       = [&gradients](const std::string& name)
            { return std::find(gradients.begin(), gradients.end(), name) != gradients.end(); };
            if (!has("positions") || (computeVirial && !has("strain")))
            {
                GMX_THROW(APIError(
                        "The energy block of the metatomic model lacks the requested 'positions' "
                        "or 'strain' gradient"));
            }
            auto positionsGradient = block.gradient("positions");
            if (modelIsDouble)
            {
                add_atom_vectors<double>(positionsGradient, 2, nAtoms, &forces);
            }
            else
            {
                add_atom_vectors<float>(positionsGradient, 2, nAtoms, &forces);
            }
            if (computeVirial)
            {
                auto strainGradient = block.gradient("strain");
                if (modelIsDouble)
                {
                    add_virial<double>(strainGradient, virialMatrix);
                }
                else
                {
                    add_virial<float>(strainGradient, virialMatrix);
                }
            }
        }
        if (!data_->nonConservative)
        {
            // the force is minus the positions gradient
            std::transform(forces.begin(), forces.end(), forces.begin(), std::negate<>());
        }
        else
        {
            auto& forceMap = modelOutputs[data_->ncForceIndex.value()];
            for (size_t blockId = 0; blockId < forceMap.keys().count(); ++blockId)
            {
                auto block = forceMap.block_by_id(blockId);
                if (modelIsDouble)
                {
                    add_atom_vectors<double>(block, 1, nAtoms, &forces);
                }
                else
                {
                    add_atom_vectors<float>(block, 1, nAtoms, &forces);
                }
            }
            if (computeVirial && data_->ncStressIndex.has_value())
            {
                auto&  stressMap = modelOutputs[data_->ncStressIndex.value()];
                matrix stressVirial;
                clear_mat(stressVirial);
                for (size_t blockId = 0; blockId < stressMap.keys().count(); ++blockId)
                {
                    auto block = stressMap.block_by_id(blockId);
                    if (modelIsDouble)
                    {
                        add_virial<double>(block, stressVirial);
                    }
                    else
                    {
                        add_virial<float>(block, stressVirial);
                    }
                }
                msmul(stressVirial, det(inputs.box_), virialMatrix);
            }
        }

        const double* forceData = forces.data();
        if (data_->nonConservative || !parallel)
        {
            const int32_t nApply = data_->nonConservative ? numHomeMta_ : nAtoms;
            for (int32_t i = 0; i < nApply; ++i)
            {
                const int32_t gmxIdx = mtaToGmxLocal_[i];
                outputs->forceWithVirial_.force_[gmxIdx][0] += static_cast<real>(forceData[3 * i]);
                outputs->forceWithVirial_.force_[gmxIdx][1] += static_cast<real>(forceData[3 * i + 1]);
                outputs->forceWithVirial_.force_[gmxIdx][2] += static_cast<real>(forceData[3 * i + 2]);
            }
        }
        else
        {
            distributeNonHomeForces(forceData, outputs);
        }
    }

    if (parallel)
    {
        if (numLocalMta_ != origNumLocalMta)
        {
            numLocalMta_ = origNumLocalMta;
            positions_.resize(origNumLocalMta);
            atomNumbers_.resize(origNumLocalMta);
            mtaToGlobalMta_.resize(origNumLocalMta);
        }
        pairlistMta_.resize(origPairlistSize);
        cellShiftsMta_.resize(origShiftsSize);
    }

    outputs->enerd_.term[InteractionFunction::MetatomicPotentialEnergy] = static_cast<real>(energy);
    outputs->forceWithVirial_.addVirialContribution(virialMatrix);
}

} // namespace gmx
