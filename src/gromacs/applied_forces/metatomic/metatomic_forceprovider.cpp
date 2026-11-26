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
 * In domain decomposition, every rank runs the model on its home MTA atoms
 * and the MTA atoms within the interaction range of them, with the pairs among
 * these atoms, and asks for the outputs of its home atoms only.
 *
 * Each rank sends its home atoms that may be near the home atoms of another rank
 * to that rank. Forces on home atoms are applied directly, and the forces on the
 * other atoms go back to their home ranks the same way, as ForceWithVirial is not
 * communicated by dd_move_f.
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
#include <metatensor.hpp>
#include <metatomic.hpp>
#include <numeric>
#include <optional>
#include <string>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <metatensor/dlpack/dlpack.h>

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
#include "gromacs/utility/functions.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/mpicomm.h"
#include "gromacs/utility/strconvert.h"
#include "gromacs/utility/stringutil.h"

#include "metatomic_cellshift.h"
#include "metatomic_environment.h"


namespace gmx
{

/*! \brief Longest bond capped by a link atom, in nm, longer than any covalent bond.
 *
 * With domain decomposition, the environment of a rank reaches twice this further than the
 * interaction range: a cap is within a bond of its two atoms, so the rank then holds both
 * atoms of every cap within the interaction range of its home atoms and of their caps.
 */
constexpr double c_maxLinkBondNm = 0.3;

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
                                              const std::string&                      name)
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
    auto* ctx  = new DlpackShape;
    ctx->shape = std::move(shape);
    ctx->strides.resize(ctx->shape.size());
    int64_t stride = 1;
    for (int axis = static_cast<int>(ctx->shape.size()) - 1; axis >= 0; --axis)
    {
        ctx->strides[axis] = stride;
        stride *= ctx->shape[axis];
    }

    auto* tensor                  = new DLManagedTensorVersioned{};
    tensor->version.major         = DLPACK_MAJOR_VERSION;
    tensor->version.minor         = DLPACK_MINOR_VERSION;
    tensor->manager_ctx           = ctx;
    tensor->deleter               = dlpack_deleter;
    tensor->flags                 = DLPACK_FLAG_BITMASK_READ_ONLY;
    tensor->dl_tensor.data        = data;
    tensor->dl_tensor.byte_offset = 0;
    tensor->dl_tensor.device      = { kDLCPU, 0 };
    tensor->dl_tensor.dtype       = dtype;
    tensor->dl_tensor.ndim        = static_cast<int32_t>(ctx->shape.size());
    tensor->dl_tensor.shape       = ctx->shape.data();
    tensor->dl_tensor.strides     = ctx->strides.data();
    return metatomic::DLPackTensor(tensor);
}

template<typename T>
static int count_above(metatensor::TensorBlock& block, double threshold)
{
    auto   values = block.values<T>();
    size_t n      = 1;
    for (size_t extent : values.shape())
    {
        n *= extent;
    }
    const T* data   = values.data();
    int      nAbove = 0;
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
    auto   values = block.values<T>();
    size_t n      = 1;
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
static void add_atom_vectors(metatensor::TensorBlock& block,
                             int                      atomColumn,
                             int32_t                  nAtoms,
                             std::vector<double>*     vectors)
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
    metatomic::ModelCapabilities              capabilities =
            metatomic::ModelCapabilities::builder()
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
    double lengthToNm       = 1.0;
    bool   useFloat64       = true;
    bool   checkConsistency = false;

    std::vector<std::string> nlSampleNames = { "first_atom",
                                               "second_atom",
                                               "cell_shift_a",
                                               "cell_shift_b",
                                               "cell_shift_c" };

    std::array<uint8_t, 3> pbc     = { 1, 1, 1 };
    PbcType                pbcType = PbcType::Xyz;
    //! Largest pair-list cutoff in nm, 0 without pair lists.
    double maxCutoffNm = 0.0;
    //! With domain decomposition, the atoms within this distance of the home atoms, in nm.
    double environmentRangeNm = 0.0;

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
    //! The model provides the virial: the strain gradient, or the non-conservative stress.
    bool modelVirial = false;

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
        if (!options_.params_.pluginPath.empty())
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
            modelOptions =
                    nlohmann::json(parseMetatomicModelOptions(options_.params_.modelOptions)).dump();
        }
        data_->model = std::make_unique<metatomic::ExternalModel>(
                metatomic::load_model(options_.params_.modelPath_, modelOptions));
        data_->capabilities = data_->model->capabilities();
        data_->lengthToNm =
                metatomic::unit_conversion_factor(data_->capabilities.length_unit(), "nm");
    }
    catch (const std::exception& e)
    {
        GMX_THROW(InvalidInputError(formatString("Failed to load the metatomic model '%s': %s",
                                                 options_.params_.modelPath_.c_str(),
                                                 e.what())));
    }

    data_->useFloat64 = data_->capabilities.dtype() == metatomic::ModelCapabilities::DType::Float64;
    if (data_->capabilities.dtype() != metatomic::ModelCapabilities::DType::Float64
        && data_->capabilities.dtype() != metatomic::ModelCapabilities::DType::Float32)
    {
        GMX_THROW(InvalidInputError(
                "The metatomic model uses a dtype other than float32 and float64, which GROMACS "
                "does not support."));
    }
    data_->checkConsistency = options_.params_.checkConsistency;
    const PbcType* pbcType  = options_.params_.pbcType_.get();
    if (pbcType != nullptr && *pbcType == PbcType::Screw)
    {
        GMX_THROW(NotImplementedError(
                "Metatomic does not support screw periodic boundary conditions."));
    }
    data_->pbcType = pbcType != nullptr ? *pbcType : PbcType::Xyz;
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
        GMX_THROW(InconsistentInputError(
                formatString("The metatomic model at '%s' does not provide an '%s' output. Check "
                             "metatomic-variant.",
                             params.modelPath_.c_str(),
                             data_->energyName.c_str())));
    }

    data_->nonConservative = params.nonConservative;
    // where the virial of the model comes from, for the messages when it does not
    std::string virialSource;
    if (!data_->nonConservative)
    {
        // The forces need the positions gradient, the virial the strain gradient.
        const auto& declared = energyCap->gradients();
        const auto  declares = [&declared](metatomic::Gradients gradient)
        { return std::find(declared.begin(), declared.end(), gradient) != declared.end(); };
        if (!declares(metatomic::Gradients::Positions))
        {
            GMX_THROW(InconsistentInputError(
                    formatString("The model's '%s' output does not declare gradients with respect "
                                 "to positions, which GROMACS needs for the forces.",
                                 data_->energyName.c_str())));
        }
        data_->modelVirial = declares(metatomic::Gradients::Strain);
        virialSource       = formatString("the strain gradient of '%s'", data_->energyName.c_str());
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
        std::vector<Gradients> gradients = { Gradients::Positions };
        data_->energyIndex               = data_->requested.size();
        data_->requested.push_back(energy(gradients));
        if (data_->modelVirial)
        {
            gradients.push_back(Gradients::Strain);
        }
        data_->requestedWithVirial.push_back(energy(gradients));
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
            GMX_THROW(InconsistentInputError(
                    formatString("The metatomic model does not provide a '%s' output, which "
                                 "metatomic-non-conservative = yes needs.",
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
        virialSource = formatString("the '%s' output", stressName.c_str());
        if (find_output(modelOutputs, stressName) != nullptr)
        {
            data_->modelVirial   = true;
            data_->ncStressIndex = data_->requestedWithVirial.size();
            data_->requestedWithVirial.push_back(metatomic::Quantity::builder()
                                                         .name(stressName)
                                                         .unit("kJ/mol/nm^3")
                                                         .sample_kind(metatomic::SampleKind::System)
                                                         .build());
        }
        GMX_LOG(logger_.info)
                .asParagraph()
                .appendTextFormatted("Metatomic: non-conservative mode enabled. Forces from '%s'%s",
                                     forceName.c_str(),
                                     data_->ncStressIndex.has_value() ? ", with stress" : "");
    }

    if (!data_->modelVirial)
    {
        if (params.pressureCoupling)
        {
            GMX_THROW(InconsistentInputError(
                    formatString("Pressure coupling needs the virial of the metatomic model, "
                                 "which comes from %s. The model does not provide it.",
                                 virialSource.c_str())));
        }
        GMX_LOG(logger_.warning)
                .asParagraph()
                .appendTextFormatted(
                        "The metatomic model does not provide %s, so it adds zero to the virial. "
                        "The virial and pressure that GROMACS reports leave out the model.",
                        virialSource.c_str());
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
        GMX_THROW(
                InvalidInputError("The metatomic model failed to give the pair lists and inputs "
                                  "it needs: "
                                  + std::string(e.what())));
    }
    if (!inputs.empty())
    {
        GMX_THROW(InconsistentInputError(formatString(
                "The metatomic model requests the input '%s', which GROMACS can not provide.",
                inputs.front().name().c_str())));
    }
    data_->cutoffNm.reserve(data_->pairLists.size());
    for (const auto& pairs : data_->pairLists)
    {
        data_->cutoffNm.push_back(pairs.cutoff() * data_->lengthToNm);
        data_->maxCutoffNm = std::max(data_->maxCutoffNm, data_->cutoffNm.back());
    }
    // A home atom depends on the atoms within the interaction range, and on its pairs.
    data_->environmentRangeNm =
            std::max(data_->capabilities.interaction_range() * data_->lengthToNm, data_->maxCutoffNm);

    // Link atoms: the bonds they cap, with the indices of their atoms in mtaIndices_
    if (!params.linkFrontier_.empty())
    {
        std::unordered_map<Index, int32_t> mtaOf;
        for (size_t k = 0; k < params.mtaIndices_.size(); ++k)
        {
            mtaOf.emplace(params.mtaIndices_[k], static_cast<int32_t>(k));
        }
        for (auto bond : params.linkFrontier_)
        {
            const auto embedded = mtaOf.find(bond.getEmbeddedIndex());
            const auto mm       = mtaOf.find(bond.getMMIndex());
            GMX_RELEASE_ASSERT(embedded != mtaOf.end() && mm != mtaOf.end(),
                               "The model sees both atoms of a bond capped by a link atom");
            bond.setInputIndices(embedded->second, mm->second);
            linkBonds_.push_back(bond);
        }
        data_->environmentRangeNm += 2 * c_maxLinkBondNm;
        GMX_LOG(logger_.info)
                .asParagraph()
                .appendTextFormatted(
                        "Metatomic: %zu link atoms cap the bonds between embedded and MM atoms",
                        linkBonds_.size());
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
        GMX_THROW(
                NotImplementedError("interaction_range is infinite for this model; "
                                    "using multiple MPI domains is not supported."));
    }

    if (mpiComm_.isParallel())
    {
        // With domain decomposition, the force provider finds the pairs itself.
        GMX_LOG(logger_.info)
                .asParagraph()
                .appendTextFormatted(
                        "Metatomic: with domain decomposition, each rank runs the model on its "
                        "home atoms and the atoms within %.3f nm of them",
                        data_->environmentRangeNm);
        return 0.0;
    }

    // The model only reads pairs within its pair-list cutoffs; a longer interaction range
    // comes from message passing over them. So the plain pairlist needs the largest
    // cutoff plus a buffer for the motion between pairlist updates, not the interaction
    // range (which would force rlist up to it).
    const double pairCutoff = data_->maxCutoffNm;
    if (pairCutoff == 0.0)
    {
        // The model reads no pairs.
        return 0.0;
    }

    // GROMACS builds the plain pairlist on search steps only, at the requested range, and
    // leaves the buffer for the motion until the next search to the requester. Twice the
    // RMS displacement over a pairlist lifetime, as PlainPairlistRanges::rmsdDistance()
    // recommends, and at least 0.1 nm for fast ML hydrogens. The normal pairlist holds no
    // pairs beyond rlist, so the buffer ends there.
    const std::optional<real> rmsd   = ranges.rmsdDistance();
    const double              buffer = std::max(2 * static_cast<double>(rmsd.value_or(0.0)), 0.1);
    // rounded to the precision of rlist, so that rlist set to the cutoff, or to the cutoff
    // plus the buffer, is enough
    const real cutoff = static_cast<real>(pairCutoff);
    const real wanted = static_cast<real>(pairCutoff + buffer);
    const real rlist  = ranges.pairlistCutoff();
    if (cutoff > rlist)
    {
        GMX_THROW(InconsistentInputError(formatString(
                "The metatomic model reads pairs up to %.3f nm, but the GROMACS pair list only "
                "reaches rlist = %.3f nm. Raise rlist to at least %.3f nm, by setting rvdw and "
                "rcoulomb to at least that, or rlist with verlet-buffer-tolerance = -1.",
                cutoff,
                rlist,
                wanted)));
    }
    const real range = std::min(wanted, rlist);
    // without motion between searches (energy minimization, or a search at every step),
    // the buffer is not needed
    const bool atomsMove = !rmsd.has_value() || rmsd.value() > 0;
    if (range < wanted && atomsMove)
    {
        GMX_LOG(logger_.warning)
                .asParagraph()
                .appendTextFormatted(
                        "Metatomic: the buffer of the pair list is %.3f nm, less than the %.3f nm "
                        "the atoms can move between pair searches, because rlist is %.3f nm. The "
                        "model misses the pairs that come within its cutoff between searches, "
                        "and the energy drifts. Raise rlist to at least %.3f nm, or lower "
                        "nstlist.",
                        range - cutoff,
                        buffer,
                        rlist,
                        wanted);
    }
    GMX_LOG(logger_.info)
            .asParagraph()
            .appendTextFormatted(
                    "Metatomic: plain pairlist range %.3f nm (pair-list cutoff %.3f nm + %.3f nm "
                    "buffer; interaction range %.3f nm)",
                    range,
                    cutoff,
                    range - cutoff,
                    interactionRange);
    return range;
}

/*! \brief Rebuild the home MTA atom tables after domain decomposition.
 *
 * Called on every AtomsRedistributed signal. Builds, for the home MTA atoms:
 *  - mtaToGmxLocal_: model index -> GROMACS local buffer index
 *  - mtaToGlobalMta_: model index -> global MTA index (for the force exchange)
 *  - atomNumbers_: atomic numbers for the model input
 * and, without domain decomposition, gmxLocalToMtaIdx_ to read the GROMACS pairlist.
 * The other atoms of a rank's environment are added at each step by gatherEnvironment().
 */
void MetatomicForceProvider::gatherAtomNumbersIndices(const MDModulesAtomsRedistributedSignal& signal)
{
    const auto&   mtaIndices  = options_.params_.mtaIndices_;
    const int32_t numTotalMta = static_cast<int32_t>(mtaIndices.size());

    mtaToGmxLocal_.clear();
    mtaToGlobalMta_.clear();
    atomNumbers_.clear();
    gmxLocalToMtaIdx_.clear();

    if (mpiComm_.isParallel())
    {
        GMX_RELEASE_ASSERT(signal.globalAtomIndices_.has_value(),
                           "Global atom indices required for domain decomposition.");
        const auto globalAtomIndices = signal.globalAtomIndices_.value();

        std::unordered_map<int32_t, int32_t> globalToMtaIdx;
        for (int32_t j = 0; j < numTotalMta; j++)
        {
            globalToMtaIdx[static_cast<int32_t>(mtaIndices[j])] = j;
        }
        // The home atoms come first in the local buffer, followed by the halo.
        for (int32_t i = 0; i < static_cast<int32_t>(signal.x_.size()); i++)
        {
            const auto it = globalToMtaIdx.find(globalAtomIndices[i]);
            if (it != globalToMtaIdx.end())
            {
                mtaToGmxLocal_.push_back(i);
                mtaToGlobalMta_.push_back(it->second);
                atomNumbers_.push_back(options_.params_.atoms_.atom[globalAtomIndices[i]].atomnumber);
            }
        }
        numHomeMta_  = static_cast<int32_t>(mtaToGmxLocal_.size());
        numLocalMta_ = numHomeMta_;
    }
    else
    {
        // Without DD all MTA atoms are home
        const auto* mtaAtoms = options_.params_.mtaAtoms_.get();
        numHomeMta_          = numTotalMta;
        numLocalMta_         = numTotalMta;

        mtaToGmxLocal_.resize(numTotalMta);
        mtaToGlobalMta_.resize(numTotalMta);
        atomNumbers_.resize(numTotalMta);
        gmxLocalToMtaIdx_.assign(signal.x_.size(), -1);

        for (int32_t i = 0; i < numTotalMta; i++)
        {
            int32_t localIndex = mtaAtoms->localIndex()[i];
            int32_t globalIdx  = mtaAtoms->globalIndex()[mtaAtoms->collectiveIndex()[i]];

            mtaToGmxLocal_[i]             = localIndex;
            mtaToGlobalMta_[i]            = i;
            atomNumbers_[i]               = options_.params_.atoms_.atom[globalIdx].atomnumber;
            gmxLocalToMtaIdx_[localIndex] = i;
        }
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
    const auto appendPairlistEntries = [this](const auto& pairlistEntries)
    {
        for (const auto& entry : pairlistEntries)
        {
            const auto& [atomPair, shiftIndex] = entry;
            const int32_t idxA                 = gmxLocalToMtaIdx_[atomPair.first];
            const int32_t idxB                 = gmxLocalToMtaIdx_[atomPair.second];

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


/*! \brief Sends \p counts[r] blocks of \p width numbers of \p send to rank r.
 *
 * \p receivedCounts[r] blocks come from rank r, in rank order in the result.
 */
static std::vector<double> exchangeBlocks(const MpiComm&             mpiComm,
                                          const std::vector<double>& send,
                                          const std::vector<int>&    counts,
                                          const std::vector<int>&    receivedCounts,
                                          int                        width)
{
#if GMX_MPI
    const int        numRanks = mpiComm.size();
    std::vector<int> sendCounts(numRanks), sendOffsets(numRanks), receiveCounts(numRanks),
            receiveOffsets(numRanks);
    int numSent     = 0;
    int numReceived = 0;
    for (int r = 0; r < numRanks; ++r)
    {
        sendCounts[r]     = width * counts[r];
        sendOffsets[r]    = numSent;
        receiveCounts[r]  = width * receivedCounts[r];
        receiveOffsets[r] = numReceived;
        numSent += sendCounts[r];
        numReceived += receiveCounts[r];
    }
    // thread-MPI rejects null buffers, which empty vectors give
    const double        nothing = 0;
    std::vector<double> received(numReceived + 1);
    MPI_Alltoallv(send.empty() ? &nothing : send.data(),
                  sendCounts.data(),
                  sendOffsets.data(),
                  MPI_DOUBLE,
                  received.data(),
                  receiveCounts.data(),
                  receiveOffsets.data(),
                  MPI_DOUBLE,
                  mpiComm.comm());
    received.pop_back();
    return received;
#else
    GMX_UNUSED_VALUE(mpiComm);
    GMX_UNUSED_VALUE(send);
    GMX_UNUSED_VALUE(counts);
    GMX_UNUSED_VALUE(receivedCounts);
    GMX_UNUSED_VALUE(width);
    GMX_RELEASE_ASSERT(false, "Parallel metatomic evaluation needs an MPI build");
    return {};
#endif
}

void MetatomicForceProvider::gatherEnvironment(const matrix box)
{
    const auto&  mtaIndices = options_.params_.mtaIndices_;
    const int    numRanks   = mpiComm_.size();
    const int    rank       = mpiComm_.rank();
    const double range      = data_->environmentRangeNm;

    t_pbc pbc;
    set_pbc(&pbc, data_->pbcType, box);
    if (data_->maxCutoffNm > 0 && data_->pbcType != PbcType::No
        && square(data_->maxCutoffNm) >= max_cutoff2(data_->pbcType, box))
    {
        GMX_THROW(InconsistentInputError(formatString(
                "The pair-list cutoff of the metatomic model, %g nm, is too long for the "
                "minimum image convention in this box.",
                data_->maxCutoffNm)));
    }

    // The region of the home atoms of every rank: a flag and the bounds, 7 numbers per rank.
    const LatticeCoordinates  lattice(box, data_->pbc);
    std::vector<LatticePoint> homePoints;
    for (int32_t i = 0; i < numHomeMta_; i++)
    {
        homePoints.push_back(lattice(positions_[i]));
    }
    const LatticeRegion own = latticeRegion(homePoints);
    std::vector<double> regions(7 * numRanks, 0.0);
    if (!own.empty())
    {
        regions[7 * rank] = 1;
        std::copy(own.lo.begin(), own.lo.end(), regions.begin() + 7 * rank + 1);
        std::copy(own.hi.begin(), own.hi.end(), regions.begin() + 7 * rank + 4);
    }
    mpiComm_.sumReduce(regions.size(), regions.data());

    // Each rank gets the home atoms that may be within range of its own home atoms, as
    // (global MTA index, x, y, z).
    sendAtoms_.clear();
    sendCounts_.assign(numRanks, 0);
    std::vector<double> sent;
    for (int r = 0; r < numRanks; ++r)
    {
        if (r == rank || regions[7 * r] == 0)
        {
            continue;
        }
        LatticeRegion other;
        std::copy(regions.begin() + 7 * r + 1, regions.begin() + 7 * r + 4, other.lo.begin());
        std::copy(regions.begin() + 7 * r + 4, regions.begin() + 7 * r + 7, other.hi.begin());
        if (!regionsAreNear(other, own, lattice, data_->pbc, range))
        {
            continue;
        }
        for (int32_t i = 0; i < numHomeMta_; i++)
        {
            if (isNearRegion(other, homePoints[i], lattice, data_->pbc, range))
            {
                sendAtoms_.push_back(i);
                sendCounts_[r]++;
                sent.insert(sent.end(),
                            { static_cast<double>(mtaToGlobalMta_[i]),
                              positions_[i][XX],
                              positions_[i][YY],
                              positions_[i][ZZ] });
            }
        }
    }
    receivedCounts_.assign(numRanks, 0);
#if GMX_MPI
    MPI_Alltoall(sendCounts_.data(), 1, MPI_INT, receivedCounts_.data(), 1, MPI_INT, mpiComm_.comm());
#endif
    const auto received = exchangeBlocks(mpiComm_, sent, sendCounts_, receivedCounts_, 4);

    // The environment: the received atoms within range of a home atom.
    std::vector<RVec> candidates;
    for (size_t q = 0; 4 * q < received.size(); q++)
    {
        candidates.emplace_back(static_cast<real>(received[4 * q + 1]),
                                static_cast<real>(received[4 * q + 2]),
                                static_cast<real>(received[4 * q + 3]));
    }
    const auto home = ArrayRef<const RVec>(positions_).subArray(0, numHomeMta_);
    environmentSources_.clear();
    for (const int32_t q : atomsWithinRange(&pbc, home, candidates, range))
    {
        const auto global = static_cast<int32_t>(received[4 * q]);
        positions_.push_back(candidates[q]);
        atomNumbers_.push_back(options_.params_.atoms_.atom[mtaIndices[global]].atomnumber);
        mtaToGlobalMta_.push_back(global);
        environmentSources_.push_back(q);
    }
    numLocalMta_ = static_cast<int32_t>(positions_.size());

    findPairs(&pbc, box, data_->pbc, positions_, data_->maxCutoffNm, &pairlistMta_, &cellShiftsMta_);
}


void MetatomicForceProvider::returnEnvironmentForces(ArrayRef<const double> forces,
                                                     ForceProviderOutput*   outputs)
{
    auto&& force = outputs->forceWithVirial_.force_;
    for (int32_t i = 0; i < numHomeMta_; i++)
    {
        for (int d = 0; d < 3; d++)
        {
            force[mtaToGmxLocal_[i]][d] += static_cast<real>(forces[3 * i + d]);
        }
    }

    // The forces on the environment go back in the order its atoms came in, zero for
    // the atoms that were not in range.
    const int numReceived = std::accumulate(receivedCounts_.begin(), receivedCounts_.end(), 0);
    std::vector<double> back(3 * numReceived, 0.0);
    for (int32_t e = numHomeMta_; e < numLocalMta_; e++)
    {
        const int32_t q = environmentSources_[e - numHomeMta_];
        for (int d = 0; d < 3; d++)
        {
            back[3 * q + d] = forces[3 * e + d];
        }
    }
    const auto returned = exchangeBlocks(mpiComm_, back, receivedCounts_, sendCounts_, 3);
    for (size_t k = 0; k < sendAtoms_.size(); k++)
    {
        for (int d = 0; d < 3; d++)
        {
            force[mtaToGmxLocal_[sendAtoms_[k]]][d] += static_cast<real>(returned[3 * k + d]);
        }
    }
}


void MetatomicForceProvider::placeLinkCaps(const matrix box)
{
    caps_.clear();
    rowOfMta_.assign(options_.params_.mtaIndices_.size(), -1);
    for (int32_t row = 0; row < numLocalMta_; ++row)
    {
        rowOfMta_[mtaToGlobalMta_[row]] = row;
    }
    modelTypes_.assign(atomNumbers_.begin(), atomNumbers_.begin() + numLocalMta_);
    modelPositions_.assign(positions_.begin(), positions_.begin() + numLocalMta_);
    isCapRow_.assign(numLocalMta_, 0);
    for (const auto& bond : linkBonds_)
    {
        const int32_t embedded = rowOfMta_[bond.getInputIndexEmb()];
        const int32_t mm       = rowOfMta_[bond.getInputIndexMM()];
        if (embedded < 0 || mm < 0)
        {
            // with DD, the cap is beyond the interaction range of the home atoms and their caps
            continue;
        }
        const RVec raw    = positions_[mm] - positions_[embedded];
        const IVec shift  = minimumImageCellShift(box, data_->pbc, raw);
        const RVec vector = raw + cellShiftVector(box, shift);
        const real length = norm(vector);
        if (length == 0)
        {
            GMX_THROW(InconsistentInputError("A bond capped by a link atom has zero length."));
        }
        // With DD, the environment holds the atoms near the caps when the bonds are at most
        // c_maxLinkBondNm long. Every rank checks the bonds of its home MM atoms.
        if (mm < numHomeMta_ && length > c_maxLinkBondNm)
        {
            GMX_THROW(InconsistentInputError(formatString(
                    "The bond between atoms %d and %d, capped by a link atom, is %g nm long, "
                    "longer than %g nm, the longest that metatomic-link-atoms supports.",
                    static_cast<int>(bond.getEmbeddedIndex()) + 1,
                    static_cast<int>(bond.getMMIndex()) + 1,
                    length,
                    c_maxLinkBondNm)));
        }
        LinkCap cap{ embedded, mm, mm, shift, bond.linkDistance() };
        if (isCapRow_[mm])
        {
            cap.row = static_cast<int32_t>(modelTypes_.size());
            modelTypes_.push_back(0);
            modelPositions_.emplace_back();
            isCapRow_.push_back(1);
        }
        isCapRow_[cap.row]       = 1;
        modelTypes_[cap.row]     = bond.linkAtomNumber();
        modelPositions_[cap.row] = positions_[embedded] + (cap.distance / length) * vector;
        caps_.push_back(cap);
    }

    // The pairs of the MM atoms are those of their caps, found from the cap positions.
    modelPairs_.clear();
    modelShifts_.clear();
    for (size_t k = 0; k < cellShiftsMta_.size(); ++k)
    {
        const int32_t i = pairlistMta_[2 * k];
        const int32_t j = pairlistMta_[2 * k + 1];
        if (!isCapRow_[i] && !isCapRow_[j])
        {
            modelPairs_.insert(modelPairs_.end(), { i, j });
            modelShifts_.push_back(cellShiftsMta_[k]);
        }
    }
    const double  cutoff2 = data_->maxCutoffNm * data_->maxCutoffNm;
    const int32_t nModel  = static_cast<int32_t>(modelTypes_.size());
    for (int32_t c = 0; c < nModel; ++c)
    {
        for (int32_t j = 0; isCapRow_[c] && j < nModel; ++j)
        {
            // each pair of caps once
            if (j == c || (isCapRow_[j] && j < c))
            {
                continue;
            }
            const RVec raw    = modelPositions_[j] - modelPositions_[c];
            const IVec shift  = minimumImageCellShift(box, data_->pbc, raw);
            const RVec vector = raw + cellShiftVector(box, shift);
            if (norm2(vector) <= cutoff2)
            {
                modelPairs_.insert(modelPairs_.end(), { c, j });
                modelShifts_.push_back(shift);
            }
        }
    }
}


void MetatomicForceProvider::spreadLinkCapForces(const matrix box, std::vector<double>* forces, matrix* virial) const
{
    // A cap row holds the force on the cap only: the first cap on an MM atom replaced it.
    std::vector<RVec> capForces;
    for (const auto& cap : caps_)
    {
        capForces.emplace_back(
                (*forces)[3 * cap.row], (*forces)[3 * cap.row + 1], (*forces)[3 * cap.row + 2]);
    }
    for (const auto& cap : caps_)
    {
        std::fill_n(forces->begin() + 3 * cap.row, 3, 0.0);
    }
    for (size_t k = 0; k < caps_.size(); ++k)
    {
        const auto& cap               = caps_[k];
        const RVec  mmShift           = cellShiftVector(box, cap.shift);
        const auto [onEmbedded, onMM] = spreadLinkAtomForce(
                capForces[k], positions_[cap.embedded], positions_[cap.mm], mmShift, cap.distance);
        for (int d = 0; d < DIM; ++d)
        {
            (*forces)[3 * cap.embedded + d] += onEmbedded[d];
            (*forces)[3 * cap.mm + d] += onMM[d];
        }
        if (virial != nullptr)
        {
            // The strain gradient of the model moves the cap with the box, at the scaled
            // distance (1 + e) d u from its embedded atom, while it stays at d. This changes
            // dE/de by d (u . F_cap) u (x) u, and the virial by half of it.
            const RVec   u = unitVector(positions_[cap.mm] + mmShift - positions_[cap.embedded]);
            const double scale = 0.5 * cap.distance * static_cast<double>(dot(u, capForces[k]));
            for (int a = 0; a < DIM; ++a)
            {
                for (int b = 0; b < DIM; ++b)
                {
                    (*virial)[a][b] += static_cast<real>(scale * u[a] * u[b]);
                }
            }
        }
    }
    forces->resize(3 * static_cast<size_t>(numLocalMta_));
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

    // With domain decomposition, the model sees the home atoms and their environment, and
    // the pairs among them, all found from the positions of every rank. The rows appended
    // for the environment are removed after the evaluation, so that gatherAtomPositions on
    // the next step only sees the rows mtaToGmxLocal_ maps.
    const bool parallel = mpiComm_.isParallel();
    if (parallel)
    {
        gatherEnvironment(inputs.box_);
    }
    // With link atoms, the model sees the caps in place of their MM atoms, and after the local atoms.
    const bool haveLinks = !linkBonds_.empty();
    if (haveLinks)
    {
        placeLinkCaps(inputs.box_);
    }
    const auto& modelTypes     = haveLinks ? modelTypes_ : atomNumbers_;
    const auto& modelPositions = haveLinks ? modelPositions_ : positions_;
    const auto& modelPairs     = haveLinks ? modelPairs_ : pairlistMta_;
    const auto& modelShifts    = haveLinks ? modelShifts_ : cellShiftsMta_;
    const auto  nModel = haveLinks ? static_cast<int32_t>(modelTypes_.size()) : numLocalMta_;

    double energy = 0.0;
    matrix virialMatrix;
    clear_mat(virialMatrix);
    std::vector<double> forces(static_cast<size_t>(nModel) * 3, 0.0);
    // home atoms whose energy uncertainty is above the threshold
    int numAboveThreshold = 0;

    // A rank without home atoms has nothing to evaluate, but takes part in the force exchange.
    if (numHomeMta_ > 0)
    {
        const int32_t    nAtoms        = nModel;
        const bool       modelIsDouble = data_->useFloat64;
        const DLDataType floatType =
                modelIsDouble ? DLDataType{ kDLFloat, 64, 1 } : DLDataType{ kDLFloat, 32, 1 };
        const bool           gromacsIsDouble = std::is_same_v<real, double>;
        std::vector<int32_t> types(modelTypes.begin(), modelTypes.begin() + nAtoms);

        void* positionData = nullptr;
        void* cellData     = nullptr;
        if (modelIsDouble == gromacsIsDouble)
        {
            positionData = const_cast<RVec*>(modelPositions.data());
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
                        positions[3 * i + d] = static_cast<T>(modelPositions[i][d]);
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

        auto system = metatomic::System("nm",
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
            const int64_t nHalf    = static_cast<int64_t>(modelPairs.size() / 2);
            const int64_t maxPairs = full ? 2 * nHalf : nHalf;
            nlSamplesBuffer_.resize(static_cast<size_t>(maxPairs) * 5);

            // Pair vectors use the system length unit (nm), in the dtype of the model. They are
            // written straight into the array the pair block takes over.
            const auto pairValues = [&](auto zero) -> std::unique_ptr<metatensor::DataArrayBase>
            {
                using T = decltype(zero);
                std::vector<T> vectors(static_cast<size_t>(maxPairs) * 3);
                int64_t        nPairs = 0;
                const auto addPair = [&](int32_t i, int32_t j, const IVec& n, double sign, const double* d)
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
                    const int32_t ai    = modelPairs[2 * k];
                    const int32_t aj    = modelPairs[2 * k + 1];
                    const IVec&   n     = modelShifts[k];
                    const RVec    shift = cellShiftVector(inputs.box_, n);
                    double        d[3];
                    for (int a = 0; a < 3; ++a)
                    {
                        d[a] = static_cast<double>(modelPositions[aj][a] - modelPositions[ai][a]
                                                   + shift[a]);
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

        // The caps beyond the local atoms belong to the home rank of their MM atom.
        std::vector<int32_t> selectedCaps;
        for (const auto& cap : caps_)
        {
            if (parallel && cap.row >= numLocalMta_ && cap.mm < numHomeMta_)
            {
                selectedCaps.insert(selectedCaps.end(), { 0, cap.row });
            }
        }
        std::optional<metatensor::Labels> selected;
        if (parallel && !selectedCaps.empty())
        {
            std::vector<int32_t> rows(static_cast<size_t>(numHomeMta_) * 2);
            for (int32_t atom = 0; atom < numHomeMta_; ++atom)
            {
                rows[2 * atom]     = 0;
                rows[2 * atom + 1] = atom;
            }
            rows.insert(rows.end(), selectedCaps.begin(), selectedCaps.end());
            selected = metatensor::Labels({ "system", "atom" }, rows.data(), rows.size() / 2);
        }
        else if (parallel)
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
                data_->selectedAtoms = metatensor::Labels({ "system", "atom" },
                                                          numHomeMta_ == 0 ? nullptr : rows.data(),
                                                          static_cast<size_t>(numHomeMta_));
            }
            selected = data_->selectedAtoms;
        }

        // A model without virial is never asked for it, as warned at setup.
        const bool computeVirial = outputs->forceWithVirial_.computeVirial_ && data_->modelVirial;
        std::vector<metatomic::System> systems;
        systems.push_back(std::move(system));
        std::vector<metatensor::TensorMap> modelOutputs;
        try
        {
            modelOutputs = metatomic::execute_model(
                    *data_->model,
                    systems,
                    selected,
                    computeVirial ? data_->requestedWithVirial : data_->requested,
                    data_->checkConsistency);
        }
        catch (const std::exception& e)
        {
            GMX_THROW(InvalidInputError("The metatomic model failed: " + std::string(e.what())));
        }

        if (data_->uqIndex.has_value())
        {
            auto& uqMap = modelOutputs[data_->uqIndex.value()];
            for (size_t blockId = 0; blockId < uqMap.keys().count(); ++blockId)
            {
                auto block = uqMap.block_by_id(blockId);
                numAboveThreshold += modelIsDouble
                                             ? count_above<double>(block, data_->uncertaintyThreshold)
                                             : count_above<float>(block, data_->uncertaintyThreshold);
            }
        }

        // Model gradients come in the system's nm and the requested kJ/mol.
        auto& energyMap = modelOutputs[data_->energyIndex];
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
            if (computeVirial)
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

        if (haveLinks)
        {
            // the virial term of the caps comes with the strain gradient of the energy
            spreadLinkCapForces(inputs.box_,
                                &forces,
                                computeVirial && !data_->nonConservative ? &virialMatrix : nullptr);
        }
    }

    // With domain decomposition, every rank counts its own home atoms, and only the main
    // rank logs.
    if (data_->uqIndex.has_value())
    {
        if (parallel)
        {
            mpiComm_.sumReduce(1, &numAboveThreshold);
        }
        if (numAboveThreshold > 0)
        {
            GMX_LOG(logger_.warning)
                    .asParagraph()
                    .appendTextFormatted(
                            "Metatomic: uncertainty on atomic energies for %d atoms "
                            "is larger than the threshold of %.4f kJ/mol. "
                            "Consider retraining the model.",
                            numAboveThreshold,
                            data_->uncertaintyThreshold);
        }
    }

    // Non-conservative forces are per-atom outputs, only for the selected home atoms, unless
    // caps spread them onto the other atom of their bond.
    const double* forceData = forces.data();
    if (!parallel || (data_->nonConservative && !haveLinks))
    {
        const int32_t nApply = parallel ? numHomeMta_ : numLocalMta_;
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
        returnEnvironmentForces(forces, outputs);
    }

    if (parallel)
    {
        numLocalMta_ = numHomeMta_;
        positions_.resize(numHomeMta_);
        atomNumbers_.resize(numHomeMta_);
        mtaToGlobalMta_.resize(numHomeMta_);
    }

    outputs->enerd_.term[InteractionFunction::MetatomicPotentialEnergy] = static_cast<real>(energy);
    outputs->forceWithVirial_.addVirialContribution(virialMatrix);
}

} // namespace gmx
