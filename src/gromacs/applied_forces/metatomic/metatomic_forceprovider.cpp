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
 * In domain decomposition, every rank gathers all ML atoms and the pairs of
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
 *  - Units: the system is handed to execute_model in the model's own length
 *    unit, so metatomic converts no lengths and the positions gradient comes
 *    back per model length unit whatever metatomic-core version is used
 *    (metatensor/metatomic#369 changes that conversion for systems in another
 *    unit). GROMACS converts it to kJ/mol/nm itself.
 *
 * \author Metatensor developers <https://github.com/metatensor>
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "metatomic_forceprovider.h"

#include <cmath>
#include <cstdint>
#include <cstdio>

#include <algorithm>
#include <array>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include "gromacs/domdec/localatomset.h"
#include "gromacs/math/boxmatrix.h"
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
#include "gromacs/utility/stringutil.h"

#include "metatomic_timer.h"

#ifdef DIM
#    undef DIM
#endif

#include <metatensor.hpp>
#include <metatensor/dlpack/dlpack.h>
#include <metatomic.hpp>


namespace gmx
{

/*! \brief Normalizes the variant string for Metatomic output selection. */
static std::optional<std::string> normalize_variant(std::string variant_string)
{
    if (variant_string == "no" || variant_string.empty())
    {
        return std::nullopt;
    }
    return variant_string;
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

static void pbc_flags(const PbcType* pbcType, std::array<uint8_t, 3>* flags)
{
    if (pbcType != nullptr && *pbcType == PbcType::Screw)
    {
        GMX_THROW(NotImplementedError("Metatomic does not support screw periodic boundary conditions."));
    }
    if (pbcType != nullptr && *pbcType == PbcType::XY)
    {
        *flags = { 1, 1, 0 };
    }
    else if (pbcType != nullptr && *pbcType == PbcType::No)
    {
        *flags = { 0, 0, 0 };
    }
    else
    {
        *flags = { 1, 1, 1 };
    }
}

//! Cartesian shift n_a a + n_b b + n_c c of an integer cell shift (box rows are the box vectors).
static RVec cellShiftVector(const matrix box, const IVec& cellShift)
{
    return { cellShift[XX] * box[XX][XX] + cellShift[YY] * box[YY][XX] + cellShift[ZZ] * box[ZZ][XX],
             cellShift[YY] * box[YY][YY] + cellShift[ZZ] * box[ZZ][YY],
             cellShift[ZZ] * box[ZZ][ZZ] };
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

//! Add `scale * values` into `forces`, indexing atoms by a sample column.
template<typename T>
static void add_atom_rows(metatensor::TensorBlock& block,
                          int                      atomColumn,
                          int32_t                  nAtoms,
                          double                   scale,
                          std::vector<double>*     forces)
{
    auto       sampleLabels = block.samples();
    const auto samples      = sampleLabels.values_cpu();
    auto       values       = block.values<T>();
    const auto shape        = values.shape();
    for (size_t row = 0; row < sampleLabels.count(); ++row)
    {
        const int32_t atom = samples(row, static_cast<size_t>(atomColumn));
        if (atom < 0 || atom >= nAtoms)
        {
            GMX_THROW(APIError("Metatomic output names an atom outside the local set"));
        }
        for (int d = 0; d < 3; ++d)
        {
            const double component = shape.size() >= 3
                                             ? static_cast<double>(values(row, static_cast<size_t>(d), 0))
                                             : static_cast<double>(values(row, static_cast<size_t>(d)));
            (*forces)[3 * atom + d] += scale * component;
        }
    }
}

//! `virial += 0.5 * strain_gradient`. Strain values are V*sigma.
template<typename T>
static void add_strain(metatensor::TensorBlock& block, matrix virial)
{
    auto       sampleLabels = block.samples();
    auto       values       = block.values<T>();
    const auto shape        = values.shape();
    const size_t nRows      = shape.size() >= 3 ? sampleLabels.count() : 1;
    for (size_t row = 0; row < nRows; ++row)
    {
        for (int a = 0; a < 3; ++a)
        {
            for (int b = 0; b < 3; ++b)
            {
                double component = 0.0;
                if (shape.size() == 4)
                {
                    component = static_cast<double>(values(row, static_cast<size_t>(a), static_cast<size_t>(b), 0));
                }
                else if (shape.size() == 3)
                {
                    component = static_cast<double>(values(row, static_cast<size_t>(a), static_cast<size_t>(b)));
                }
                else
                {
                    component = static_cast<double>(values(static_cast<size_t>(a), static_cast<size_t>(b)));
                }
                virial[a][b] += static_cast<real>(0.5 * component);
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
    //! Model length unit, used for the systems handed to the model, and its size in nm.
    std::string         lengthUnit      = "nm";
    double              lengthToNm      = 1.0;
    bool                useFloat64      = true;
    bool                checkConsistency = false;

    std::vector<std::string> nlSampleNames = {
        "first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"
    };

    std::array<uint8_t, 3> pbc = { 1, 1, 1 };

    //! Sparse/dense force exchange threshold (atom count). Env: GMX_METATOMIC_SPARSE_THRESHOLD.
    int32_t sparseThreshold = 1000;

    //! Outputs passed to execute_model, in request order.
    std::vector<metatomic::Quantity> requested;
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
    int32_t                           selectedCount = -1;

    //! Buffers used when GROMACS `real` is not the model dtype or nm is not the model length unit.
    std::vector<double> positions64;
    std::vector<float>  positions32;
    std::vector<double> cell64;
    std::vector<float>  cell32;
};

MetatomicForceProvider::MetatomicForceProvider(const MetatomicOptions& options,
                                               const MDLogger&         logger,
                                               const MpiComm&          mpiComm) :
    options_(options),
    logger_(logger),
    mpiComm_(mpiComm),
    box_{ { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } },
    data_(std::make_unique<MetatomicData>())
{
    GMX_LOG(logger_.info).asParagraph().appendText("Initializing MetatomicForceProvider...");

    if (const char* timerEnv = std::getenv("GMX_METATOMIC_TIMER"))
    {
        MetatomicTimer::enable(std::string(timerEnv) != "0");
    }

    if (const char* env = std::getenv("GMX_METATOMIC_SPARSE_THRESHOLD"))
    {
        data_->sparseThreshold = std::stoi(env);
        GMX_LOG(logger_.info)
                .asParagraph()
                .appendTextFormatted("Metatomic sparse force threshold: %d",
                                     data_->sparseThreshold);
    }

    std::string deviceName = options_.params_.device;
    if (const char* env = std::getenv("GMX_METATOMIC_DEVICE"))
    {
        deviceName = env;
    }
    if (!deviceName.empty() && deviceName != "cpu")
    {
        GMX_LOG(logger_.warning)
                .asParagraph()
                .appendTextFormatted(
                        "Metatomic device '%s' is a plugin concern. This engine "
                        "submits CPU DLPack buffers.",
                        deviceName.c_str());
    }

    try
    {
        if (!options_.params_.extensionsDirectory.empty())
        {
            try
            {
                metatomic::load_plugin(options_.params_.extensionsDirectory);
            }
            catch (const metatomic::Error& e)
            {
                // Another force provider in this process registered it already.
                const std::string message = e.what();
                if (message.find("already registered") == std::string::npos)
                {
                    throw;
                }
            }
        }
        std::optional<std::string> modelOptions;
        if (!options_.params_.modelOptions.empty())
        {
            modelOptions = options_.params_.modelOptions;
        }
        data_->model = std::make_unique<metatomic::ExternalModel>(
                metatomic::load_model(options_.params_.modelPath_, modelOptions));
        data_->capabilities = data_->model->capabilities();
        data_->lengthUnit   = data_->capabilities.length_unit();
        data_->lengthToNm   = metatomic::unit_conversion_factor(data_->lengthUnit, "nm");
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
    pbc_flags(options_.params_.pbcType_.get(), &data_->pbc);

    GMX_LOG(logger_.info)
            .asParagraph()
            .appendTextFormatted("Metatomic C API, length unit %s, model dtype %s",
                                 data_->capabilities.length_unit().c_str(),
                                 data_->useFloat64 ? "float64" : "float32");

    const auto& modelOutputs = data_->capabilities.outputs();
    const auto  vEnergy      = normalize_variant(options_.params_.variant);
    data_->energyName        = quantity_name("energy", vEnergy);
    const auto* energyCap    = find_output(modelOutputs, data_->energyName);
    if (energyCap == nullptr)
    {
        GMX_THROW(APIError(formatString(
                "The model at '%s' does not provide an '%s' output. "
                "Metatomic interface cannot proceed.",
                options_.params_.modelPath_.c_str(),
                data_->energyName.c_str())));
    }
    if (energyCap->sample_kind() != metatomic::SampleKind::Atom)
    {
        GMX_LOG(logger_.warning)
                .asParagraph()
                .appendText("Metatomic model does not support per-atom energy output "
                            "(sample_kind != \"atom\"). Energy decomposition in domain "
                            "decomposition may be less accurate.");
    }

    data_->nonConservative = options_.params_.nonConservative;
    {
        auto energy = metatomic::Quantity::builder()
                              .name(data_->energyName)
                              .unit("kJ/mol")
                              .sample_kind(energyCap->sample_kind());
        if (!data_->nonConservative)
        {
            energy.add_gradient(metatomic::Gradients::Positions)
                    .add_gradient(metatomic::Gradients::Strain);
        }
        data_->energyIndex = data_->requested.size();
        data_->requested.push_back(energy.build());
    }

    if (options_.params_.uncertaintyThreshold != "off")
    {
        const auto  vUq   = normalize_variant(options_.params_.variantEnergyUq);
        const auto  uqName = quantity_name("energy_uncertainty", vUq);
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
            if (options_.params_.uncertaintyThreshold == "auto")
            {
                data_->uncertaintyThreshold = 0.1 * metatomic::unit_conversion_factor("eV", "kJ/mol");
            }
            else
            {
                data_->uncertaintyThreshold = std::stod(options_.params_.uncertaintyThreshold);
            }
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
        const auto vForces = normalize_variant(options_.params_.variantNcForces);
        const auto vStress = normalize_variant(options_.params_.variantNcStress);
        if (vForces.has_value() && vStress.has_value() && vForces.value() != vStress.value())
        {
            GMX_THROW(APIError("if both 'variant-nc-forces' and 'variant-nc-stress' are present, "
                               "they must have the same value"));
        }
        const auto  forceName = quantity_name("non_conservative_force", vForces);
        const auto* forceCap  = find_output(modelOutputs, forceName);
        if (forceCap == nullptr)
        {
            GMX_THROW(APIError(formatString(
                    "The model does not provide '%s' output, "
                    "we can not enable non-conservative simulations",
                    forceName.c_str())));
        }
        if (forceCap->sample_kind() != metatomic::SampleKind::Atom)
        {
            GMX_THROW(APIError(formatString(
                    "The model's '%s' output can not produce per-atom output "
                    "(sample_kind != \"atom\"), we can not enable non-conservative "
                    "simulations",
                    forceName.c_str())));
        }
        data_->ncForceIndex = data_->requested.size();
        data_->requested.push_back(metatomic::Quantity::builder()
                                           .name(forceName)
                                           .unit("kJ/mol/nm")
                                           .sample_kind(metatomic::SampleKind::Atom)
                                           .build());

        const auto  stressName = quantity_name("non_conservative_stress", vStress);
        const auto* stressCap  = find_output(modelOutputs, stressName);
        if (stressCap != nullptr)
        {
            data_->ncStressIndex = data_->requested.size();
            data_->requested.push_back(metatomic::Quantity::builder()
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

    // The pair lists and extra inputs a model needs depend on the outputs it computes.
    std::vector<metatomic::Quantity> inputs;
    try
    {
        data_->pairLists = data_->model->requested_pair_lists(data_->requested);
        inputs           = data_->model->requested_inputs(data_->requested);
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
        if (!std::isfinite(interactionRange) || interactionRange == 0.0)
        {
            GMX_THROW(InconsistentInputError(
                    "The metatomic model requests no pair list and has no finite interaction "
                    "range."));
        }
        return interactionRange;
    }

    // Twice the RMS displacement over a pairlist lifetime, as recommended for
    // PlainPairlistRanges, and at least 0.1 nm for fast ML hydrogens. The normal
    // pairlist holds no pairs beyond rlist, so the buffer ends there; the cutoff
    // itself must fit (mdrun stops otherwise).
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
        int32_t                              numDuplicatesSkipped = 0;

        for (int32_t i = 0; i < numLocalPlusHalo; i++)
        {
            int32_t globalIdx = globalAtomIndices[i];
            auto    it        = globalToMtaIdx.find(globalIdx);
            if (it != globalToMtaIdx.end())
            {
                int32_t mtaIdx = it->second;

                if (mtaIdxToModelIdx.count(mtaIdx))
                {
                    // Periodic ghost: record mapping but don't create new model atom
                    numDuplicatesSkipped++;
                }
                else
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
            int32_t modelIdx  = numHomeMta_ + k;
            int32_t gmxLocal  = haloGmxLocal[k];
            int32_t globalIdx = globalAtomIndices[gmxLocal];

            mtaToGmxLocal_[modelIdx]  = gmxLocal;
            mtaToGlobalMta_[modelIdx] = haloGlobalMta[k];
            atomNumbers_[modelIdx]    = options_.params_.atoms_.atom[globalIdx].atomnumber;
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

    pbc_flags(options_.params_.pbcType_.get(), &data_->pbc);

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
 * The plain list is what the model needs inside its cutoff. The excluded
 * list is only 1-2 / 1-3 / 1-4 pairs. Reading exclusions alone drops every
 * non-bonded neighbor (the neighbor-list capacity bug).
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
    const int numRanks = mpiComm_.size();
    const int myRank   = mpiComm_.rank();
    const int sendTo   = (myRank + 1) % numRanks;
    const int recvFrom = (myRank - 1 + numRanks) % numRanks;

    // Compute box inverse once for triclinic-safe shift computation.
    matrix boxInv;
    invertBoxMatrix(box, boxInv);

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

            // Triclinic-safe minimum-image shift (matches LAMMPS cell_shifts).
            // invertBoxMatrix returns lower-triangular inverse, so upper triangle is 0.
            const double rawDx =
                    static_cast<double>(positions_[localJ][XX] - positions_[localI][XX]);
            const double rawDy =
                    static_cast<double>(positions_[localJ][YY] - positions_[localI][YY]);
            const double rawDz =
                    static_cast<double>(positions_[localJ][ZZ] - positions_[localI][ZZ]);

            IVec shift;
            shift[XX] = static_cast<int>(std::round(
                    -(boxInv[XX][XX] * rawDx + boxInv[YY][XX] * rawDy + boxInv[ZZ][XX] * rawDz)));
            shift[YY] = static_cast<int>(std::round(
                    -(boxInv[YY][YY] * rawDy + boxInv[ZZ][YY] * rawDz)));
            shift[ZZ] = static_cast<int>(std::round(-(boxInv[ZZ][ZZ] * rawDz)));

            backwardPairsMta_.push_back(localI);
            backwardPairsMta_.push_back(localJ);
            backwardShiftsMta_.push_back(shift);
            existingCanonical.insert(canonical);
        }

        // Forward received buffer for the next round.
        sendBuf.swap(recvBuf);
    }

}


void MetatomicForceProvider::distributeNonHomeForces(const double*        forces,
                                                     ForceProviderOutput* outputs)
{
    const int32_t numTotalMta = static_cast<int32_t>(options_.params_.mtaIndices_.size());

    // For small systems, dense allreduce has lower latency than the
    // sparse exchange (gather counts + allgatherv).
    const int32_t sparseThreshold = data_->sparseThreshold;

    if (numTotalMta < sparseThreshold)
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
    MPI_Gatherv(sendBuf.data(), 4 * numNonHome, MPI_DOUBLE,
                recvBuf.data(), dcounts.data(), ddispls.data(), MPI_DOUBLE,
                mpiComm_.mainRank(), mpiComm_.comm());
    MPI_Bcast(recvBuf.data(), 4 * totalNonHome, MPI_DOUBLE,
              mpiComm_.mainRank(), mpiComm_.comm());

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
    MetatomicTimer totalTimer("calculateForces", mpiComm_);

    // Fill local positions (no MPI communication)
    {
        MetatomicTimer timer("gatherAtomPositions", mpiComm_);
        gatherAtomPositions(inputs.x_);
    }
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

    // With domain decomposition every rank needs, for each home atom, all ML atoms within
    // the model's interaction range and the pairs among them (message passing), not only
    // the pairs the eighth-shell decomposition assigned to it. The GROMACS pairlist is the
    // pair source; atoms and pairs are then exchanged across ranks.
    const bool useNewtonNL = mpiComm_.isParallel();

    // The model rows appended below are removed after the evaluation, so that
    // gatherAtomPositions on the next step only sees the rows mtaToGmxLocal_ maps.
    const int32_t origNumLocalMta = numLocalMta_;

    // Save original pairlist size; backward pairs are appended temporarily.
    const std::size_t origPairlistSize = pairlistMta_.size();
    const std::size_t origShiftsSize   = cellShiftsMta_.size();

    if (useNewtonNL)
    {
        // Step 1: Make all MTA atoms local. The DD halo covers only the forward direction,
        // while each home atom energy depends on all atoms within the model interaction
        // range. Extends positions_, atomNumbers_, mtaToGlobalMta_ and numLocalMta_.
        {
            MetatomicTimer timer("gatherAllMtaAtoms", mpiComm_);
            gatherAllMtaAtoms();
        }

        // Step 2: Collect the pairs of all other ranks' pairlists, including pairs between
        // two non-home atoms. Every pair within the plain pairlist range is in exactly one
        // rank's list, so a full ring gives every rank the complete list.
        {
            MetatomicTimer timer("exchangeBackwardPairs", mpiComm_);
            exchangeBackwardPairs(inputs.box_, mpiComm_.size() - 1);
        }

        // Step 3: Temporarily extend pairlistMta_ with backward pairs
        // so the NL building loop processes all pairs in one pass.
        pairlistMta_.insert(pairlistMta_.end(),
                            backwardPairsMta_.begin(),
                            backwardPairsMta_.end());
        cellShiftsMta_.insert(cellShiftsMta_.end(),
                              backwardShiftsMta_.begin(),
                              backwardShiftsMta_.end());
    }

    // Model inference
    double energy = 0.0;
    matrix virialMatrix;
    clear_mat(virialMatrix);

    {
        MetatomicTimer modelTimer("model inference", mpiComm_);
        MetatomicTimer tensorPrepTimer("tensorPrep", mpiComm_);

        const int32_t       nAtoms       = numLocalMta_;
        const bool          modelIsDouble = data_->useFloat64;
        const DLDataType    floatType    = modelIsDouble ? DLDataType{ kDLFloat, 64, 1 }
                                                         : DLDataType{ kDLFloat, 32, 1 };
        const bool          gromacsIsDouble = std::is_same_v<real, double>;
        std::vector<int32_t> types(atomNumbers_.begin(), atomNumbers_.begin() + nAtoms);

        // Positions, cell and pair vectors go to the model in its own length unit.
        const double nmToModel = 1.0 / data_->lengthToNm;

        void* positionData = nullptr;
        void* cellData     = nullptr;
        if (modelIsDouble == gromacsIsDouble && nmToModel == 1.0)
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
                        positions[3 * i + d] = static_cast<T>(nmToModel * positions_[i][d]);
                    }
                }
                for (int a = 0; a < 3; ++a)
                {
                    for (int b = 0; b < 3; ++b)
                    {
                        cell[3 * a + b] = static_cast<T>(nmToModel * box_[a][b]);
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
                data_->lengthUnit,
                wrap_dlpack(types.data(), { nAtoms }, { kDLInt, 32, 1 }),
                wrap_dlpack(positionData, { nAtoms, 3 }, floatType),
                wrap_dlpack(cellData, { 3, 3 }, floatType),
                wrap_dlpack(data_->pbc.data(), { 3 }, { kDLBool, 8, 1 }));
        tensorPrepTimer.stop();

        MetatomicTimer buildNLTimer("buildNL", mpiComm_);
        for (size_t requestIndex = 0; requestIndex < data_->pairLists.size(); ++requestIndex)
        {
            const auto&   request = data_->pairLists[requestIndex];
            const bool    full    = request.full_list();
            const bool    strict  = request.strict();
            const double  cutoff  = data_->cutoffNm[requestIndex];
            const double  cutoff2 = cutoff * cutoff;
            const int64_t nHalf   = static_cast<int64_t>(pairlistMta_.size() / 2);
            const int64_t maxPairs = full ? 2 * nHalf : nHalf;
            nlSamplesBuffer_.resize(static_cast<size_t>(maxPairs) * 5);
            nlVectorsBuffer_.resize(static_cast<size_t>(maxPairs) * 3);

            int64_t outIdx = 0;
            for (int64_t k = 0; k < nHalf; ++k)
            {
                const int32_t ai = pairlistMta_[2 * k];
                const int32_t aj = pairlistMta_[2 * k + 1];
                const IVec&   n     = cellShiftsMta_[k];
                const RVec    shift = cellShiftVector(inputs.box_, n);
                const double dx = static_cast<double>(positions_[aj][0] - positions_[ai][0] + shift[0]);
                const double dy = static_cast<double>(positions_[aj][1] - positions_[ai][1] + shift[1]);
                const double dz = static_cast<double>(positions_[aj][2] - positions_[ai][2] + shift[2]);
                if (strict && dx * dx + dy * dy + dz * dz > cutoff2)
                {
                    continue;
                }
                nlSamplesBuffer_[5 * outIdx + 0] = ai;
                nlSamplesBuffer_[5 * outIdx + 1] = aj;
                nlSamplesBuffer_[5 * outIdx + 2] = n[0];
                nlSamplesBuffer_[5 * outIdx + 3] = n[1];
                nlSamplesBuffer_[5 * outIdx + 4] = n[2];
                nlVectorsBuffer_[3 * outIdx + 0] = dx;
                nlVectorsBuffer_[3 * outIdx + 1] = dy;
                nlVectorsBuffer_[3 * outIdx + 2] = dz;
                ++outIdx;
                if (full)
                {
                    nlSamplesBuffer_[5 * outIdx + 0] = aj;
                    nlSamplesBuffer_[5 * outIdx + 1] = ai;
                    nlSamplesBuffer_[5 * outIdx + 2] = -n[0];
                    nlSamplesBuffer_[5 * outIdx + 3] = -n[1];
                    nlSamplesBuffer_[5 * outIdx + 4] = -n[2];
                    nlVectorsBuffer_[3 * outIdx + 0] = -dx;
                    nlVectorsBuffer_[3 * outIdx + 1] = -dy;
                    nlVectorsBuffer_[3 * outIdx + 2] = -dz;
                    ++outIdx;
                }
            }
            const int64_t nPairs = outIdx;
            nlSamplesBuffer_.resize(static_cast<size_t>(nPairs) * 5);
            nlVectorsBuffer_.resize(static_cast<size_t>(nPairs) * 3);

            auto samples = metatensor::Labels(
                    data_->nlSampleNames,
                    nPairs == 0 ? nullptr : nlSamplesBuffer_.data(),
                    static_cast<size_t>(nPairs));
            const int32_t                   xyzValues[3] = { 0, 1, 2 };
            const int32_t                   distanceValue = 0;
            std::vector<metatensor::Labels> components;
            components.push_back(metatensor::Labels({ "xyz" }, xyzValues, 3));
            auto properties = metatensor::Labels({ "distance" }, &distanceValue, 1);

            // Pair vectors were built in nm, the model reads its own length unit.
            std::unique_ptr<metatensor::DataArrayBase> values;
            const auto shape = std::vector<uintptr_t>{ static_cast<uintptr_t>(nPairs), 3, 1 };
            if (modelIsDouble)
            {
                std::vector<double> packed(static_cast<size_t>(nPairs) * 3);
                for (size_t i = 0; i < packed.size(); ++i)
                {
                    packed[i] = nmToModel * nlVectorsBuffer_[i];
                }
                values = std::make_unique<metatensor::SimpleDataArray<double>>(shape, std::move(packed));
            }
            else
            {
                std::vector<float> packed(static_cast<size_t>(nPairs) * 3);
                for (size_t i = 0; i < packed.size(); ++i)
                {
                    packed[i] = static_cast<float>(nmToModel * nlVectorsBuffer_[i]);
                }
                values = std::make_unique<metatensor::SimpleDataArray<float>>(shape, std::move(packed));
            }
            system.add_pairs(request,
                             metatensor::TensorBlock(std::move(values), samples, components, properties));
        }
        buildNLTimer.stop();

        std::optional<metatensor::Labels> selected;
        if (useNewtonNL)
        {
            if (data_->selectedCount != numHomeMta_)
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
                data_->selectedCount = numHomeMta_;
            }
            selected = data_->selectedAtoms;
        }

        MetatomicTimer forwardTimer("execute_model", mpiComm_);
        std::vector<metatomic::System> systems;
        systems.push_back(std::move(system));
        std::vector<metatensor::TensorMap> modelOutputs;
        try
        {
            modelOutputs = metatomic::execute_model(
                    *data_->model, systems, selected, data_->requested, data_->checkConsistency);
        }
        catch (const std::exception& e)
        {
            GMX_THROW(APIError("[Metatomic] Model evaluation failed: " + std::string(e.what())));
        }
        forwardTimer.stop();

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

        auto&               energyMap     = modelOutputs[data_->energyIndex];
        const auto          nEnergyBlocks = energyMap.keys().count();
        std::vector<double> forces(static_cast<size_t>(nAtoms) * 3, 0.0);
        for (size_t blockId = 0; blockId < nEnergyBlocks; ++blockId)
        {
            auto block = energyMap.block_by_id(blockId);
            energy += modelIsDouble ? sum_block<double>(block) : sum_block<float>(block);
            if (data_->nonConservative)
            {
                continue;
            }
            bool hasPositions = false;
            bool hasStrain    = false;
            for (const auto& name : block.gradients_list())
            {
                hasPositions = hasPositions || name == "positions";
                hasStrain    = hasStrain || name == "strain";
            }
            if (!hasPositions || !hasStrain)
            {
                GMX_THROW(APIError(
                        "Conservative Metatomic evaluation requires 'positions' and "
                        "'strain' gradients on the energy block"));
            }
            auto posGrad    = block.gradient("positions");
            auto strainGrad = block.gradient("strain");
            // The energy is in the requested kJ/mol, and the system was in the model's
            // length unit, so the positions gradient is in kJ/mol per model length unit,
            // with or without metatensor/metatomic#369. The strain gradient is per
            // (dimensionless) strain.
            const double forceScale = -nmToModel;
            if (modelIsDouble)
            {
                add_atom_rows<double>(posGrad, 2, nAtoms, forceScale, &forces);
                add_strain<double>(strainGrad, virialMatrix);
            }
            else
            {
                add_atom_rows<float>(posGrad, 2, nAtoms, forceScale, &forces);
                add_strain<float>(strainGrad, virialMatrix);
            }
        }

        if (data_->nonConservative)
        {
            auto&      forceMap = modelOutputs[data_->ncForceIndex.value()];
            const auto nBlocks  = forceMap.keys().count();
            for (size_t blockId = 0; blockId < nBlocks; ++blockId)
            {
                auto block = forceMap.block_by_id(blockId);
                if (modelIsDouble)
                {
                    add_atom_rows<double>(block, 1, nAtoms, 1.0, &forces);
                }
                else
                {
                    add_atom_rows<float>(block, 1, nAtoms, 1.0, &forces);
                }
            }
            if (data_->ncStressIndex.has_value())
            {
                const double volume = inputs.box_[XX][XX]
                                              * (inputs.box_[YY][YY] * inputs.box_[ZZ][ZZ]
                                                 - inputs.box_[YY][ZZ] * inputs.box_[ZZ][YY])
                                      - inputs.box_[XX][YY]
                                                * (inputs.box_[YY][XX] * inputs.box_[ZZ][ZZ]
                                                   - inputs.box_[YY][ZZ] * inputs.box_[ZZ][XX])
                                      + inputs.box_[XX][ZZ]
                                                * (inputs.box_[YY][XX] * inputs.box_[ZZ][YY]
                                                   - inputs.box_[YY][YY] * inputs.box_[ZZ][XX]);
                auto&      stressMap = modelOutputs[data_->ncStressIndex.value()];
                const auto nStress   = stressMap.keys().count();
                matrix     stressVirial;
                clear_mat(stressVirial);
                for (size_t blockId = 0; blockId < nStress; ++blockId)
                {
                    auto block = stressMap.block_by_id(blockId);
                    if (modelIsDouble)
                    {
                        add_strain<double>(block, stressVirial);
                    }
                    else
                    {
                        add_strain<float>(block, stressVirial);
                    }
                }
                for (int a = 0; a < 3; ++a)
                {
                    for (int b = 0; b < 3; ++b)
                    {
                        virialMatrix[a][b] = static_cast<real>(stressVirial[a][b] * volume);
                    }
                }
            }
        }

        MetatomicTimer forceScatterTimer("forceScatter", mpiComm_);
        const double*  forceData = forces.data();
        if (data_->nonConservative || !mpiComm_.isParallel())
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
        forceScatterTimer.stop();
    }

    if (useNewtonNL)
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
