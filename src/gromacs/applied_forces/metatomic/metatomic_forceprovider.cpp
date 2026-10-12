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
 * In domain decomposition, each rank uses the GROMACS pairlist as the pair
 * source, then exchanges pair identities across ranks (backward pair exchange)
 * so every home atom has ALL its pairs. Sums home-atom energies only. Safe for
 * all model architectures (newton pair ON pattern, inspired by LAMMPS
 * pair_metatomic).
 *
 * Common design points:
 *  - Forces: home forces applied directly; non-home forces exchanged via
 *    sparse indexed communication.  ForceWithVirial is not communicated
 *    by dd_move_f, so we handle it ourselves.  Dense allreduce fallback
 *    for small systems (N_total < 1000).
 *  - Ghost deduplication: periodic ghost images share the same model index
 *    but all GROMACS local indices are mapped via gmxLocalToMtaIdx_.
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
#include <numeric>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include "gromacs/domdec/domdec_network.h"
#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/domdec/localatomset.h"
#include "gromacs/math/boxmatrix.h"
#include "gromacs/utility/vec.h"
#include "gromacs/mdlib/broadcaststructs.h"
#include "gromacs/mdrunutility/mdmodulesnotifiers.h"
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

//! The name metatomic gives a sample kind ("atom", "system", "atom_pair")
static const char* sampleKindName(metatomic::SampleKind kind)
{
    switch (kind)
    {
        case metatomic::SampleKind::Atom: return "atom";
        case metatomic::SampleKind::System: return "system";
        case metatomic::SampleKind::AtomPair: return "atom_pair";
    }
    return "unknown";
}

//! Whether a requested model input is the per-atom charge (`charge`, `charge/<variant>`).
static bool isChargeInput(const std::string& name)
{
    return name == "charge" || name.rfind("charge/", 0) == 0;
}

//! One ONIOM link cap, resolved to model rows for the current step.
struct ActiveLinkAtom
{
    //! Model rows of the embedded atom and of the boundary MM atom.
    int32_t embedded = -1;
    int32_t mm       = -1;
    //! Model row holding the cap: the MM row for the first cap on an MM atom, an extra row otherwise.
    int32_t row = -1;
    //! Minimum-image cell shift from the embedded atom to the MM atom.
    IVec mmCellShift = IVec(0, 0, 0);
    real linkDistance = 0;
};

//! Cell shift (in box vectors) bringing `dx` to its minimum image, for a triclinic GROMACS box.
static IVec minimumImageCellShift(const matrix boxInv, PbcType pbcType, const RVec& dx)
{
    IVec shift(0, 0, 0);
    if (pbcType == PbcType::No)
    {
        return shift;
    }
    shift[XX] = static_cast<int>(
            std::round(-(boxInv[XX][XX] * dx[XX] + boxInv[YY][XX] * dx[YY] + boxInv[ZZ][XX] * dx[ZZ])));
    shift[YY] = static_cast<int>(std::round(-(boxInv[YY][YY] * dx[YY] + boxInv[ZZ][YY] * dx[ZZ])));
    if (pbcType != PbcType::XY)
    {
        shift[ZZ] = static_cast<int>(std::round(-(boxInv[ZZ][ZZ] * dx[ZZ])));
    }
    return shift;
}

//! Cartesian shift n_a a + n_b b + n_c c of an integer cell shift (box rows are the box vectors).
static RVec cellShiftVector(const matrix box, const IVec& cellShift)
{
    return { cellShift[XX] * box[XX][XX] + cellShift[YY] * box[YY][XX] + cellShift[ZZ] * box[ZZ][XX],
             cellShift[YY] * box[YY][YY] + cellShift[ZZ] * box[ZZ][YY],
             cellShift[ZZ] * box[ZZ][ZZ] };
}

//! Position of a cap at `linkDistance` from the embedded atom, towards the (shifted) MM atom.
static RVec linkAtomPosition(const RVec& embedded, const RVec& mm, const RVec& mmShift, real linkDistance)
{
    const RVec bond = mm + mmShift - embedded;
    const real length = norm(bond);
    if (length == 0.0_real)
    {
        GMX_THROW(InconsistentInputError(
                "Metatomic link atom construction found a zero-length boundary bond."));
    }
    return embedded + (linkDistance / length) * bond;
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

//! Add `scale * values` into `forces`, indexing atoms by a sample column. With several systems
//! (ML sites), `systemRows[system][atom]` gives the model row of an atom of a system.
template<typename T>
static void add_atom_rows(metatensor::TensorBlock&                 block,
                          int                                      atomColumn,
                          int32_t                                  nAtoms,
                          double                                   scale,
                          std::vector<double>*                     forces,
                          const std::vector<std::vector<int32_t>>* systemRows = nullptr)
{
    auto       sampleLabels = block.samples();
    const auto samples      = sampleLabels.values_cpu();
    auto       values       = block.values<T>();
    const auto shape        = values.shape();
    for (size_t row = 0; row < sampleLabels.count(); ++row)
    {
        int32_t atom = samples(row, static_cast<size_t>(atomColumn));
        if (systemRows != nullptr)
        {
            atom = (*systemRows)[samples(row, static_cast<size_t>(atomColumn - 1))][atom];
        }
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

    //! Requested per-atom charge inputs, filled from the topology charges.
    std::vector<metatomic::Quantity> chargeInputs;
    //! Requested per-system inputs (charge, spin_multiplicity), filled from the site options.
    std::vector<metatomic::Quantity> systemInputs;

    //! Home-atom selection. Rebuilt when numHomeMta_ changes.
    std::optional<metatensor::Labels> selectedAtoms;
    int32_t                           selectedCount = -1;

    //! Buffers used when GROMACS `real` is not the model dtype.
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
                // The cutoff peek loads the same plugin during simulation setup.
                const std::string message = e.what();
                if (message.find("already registered") == std::string::npos)
                {
                    throw;
                }
            }
        }
        data_->model = std::make_unique<metatomic::ExternalModel>(
                metatomic::load_model(options_.params_.modelPath_));
        data_->capabilities = data_->model->capabilities();
        data_->pairLists    = data_->model->requested_pair_lists();
    }
    catch (const std::exception& e)
    {
        GMX_THROW(APIError("Failed to load metatomic model: " + std::string(e.what())));
    }

    data_->lengthToNm = metatomic::unit_conversion_factor(data_->capabilities.length_unit(), "nm");
    data_->useFloat64 = data_->capabilities.dtype() == metatomic::ModelCapabilities::DType::Float64;
    if (data_->capabilities.dtype() != metatomic::ModelCapabilities::DType::Float64
        && data_->capabilities.dtype() != metatomic::ModelCapabilities::DType::Float32)
    {
        GMX_THROW(APIError("Unsupported dtype from model capabilities"));
    }
    data_->checkConsistency = options_.params_.checkConsistency;
    pbc_flags(options_.params_.pbcType_.get(), &data_->pbc);

    data_->cutoffNm.reserve(data_->pairLists.size());
    for (const auto& pairs : data_->pairLists)
    {
        data_->cutoffNm.push_back(pairs.cutoff() * data_->lengthToNm);
    }

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

    for (const auto& input : data_->model->requested_inputs())
    {
        const bool perSystem = input.sample_kind() == metatomic::SampleKind::System;
        if (perSystem && (isChargeInput(input.name()) || input.name() == "spin_multiplicity"))
        {
            // the total charge and spin multiplicity of each ML site (metatomic-site-charges,
            // metatomic-site-spin-multiplicities), one system per site
            data_->systemInputs.push_back(input);
            continue;
        }
        if (!isChargeInput(input.name()) || input.sample_kind() != metatomic::SampleKind::Atom)
        {
            // the same message as the libtorch engine
            GMX_THROW(NotImplementedError(formatString(
                    "The model requests the input '%s' (per %s), which GROMACS does not provide. "
                    "Supported inputs are per-atom charges (from the topology) and per-system "
                    "charge and spin_multiplicity (metatomic-site-charges, "
                    "metatomic-site-spin-multiplicities).",
                    input.name().c_str(),
                    sampleKindName(input.sample_kind()))));
        }
        if (options_.params_.mmCharges_.empty())
        {
            GMX_THROW(InconsistentInputError(
                    "Metatomic model requests charges, but topology charges are not available."));
        }
        data_->chargeInputs.push_back(input);
    }
    if (!data_->systemInputs.empty())
    {
        GMX_LOG(logger_.info)
                .asParagraph()
                .appendTextFormatted(
                        "Metatomic: the model takes a total charge and/or spin multiplicity per "
                        "system; %d ML site(s), charge %s, multiplicity %s",
                        options_.params_.numSites(),
                        formatAndJoin(options_.params_.siteChargeValues_, " ", StringFormatter("%g"))
                                .c_str(),
                        formatAndJoin(options_.params_.siteSpinValues_, " ", StringFormatter("%d")).c_str());
    }

    linkFrontiers_ = options_.params_.linkFrontier_;
    if (!linkFrontiers_.empty())
    {
        GMX_LOG(logger_.info)
                .asParagraph()
                .appendTextFormatted("Metatomic: %zu link atoms at the ML/MM boundary",
                                     linkFrontiers_.size());
    }

    GMX_LOG(logger_.info)
            .asParagraph()
            .appendText("MetatomicForceProvider initialization complete.");
}

void MetatomicForceProvider::setLinkFrontiers(std::vector<LinkFrontierAtom> frontiers)
{
    linkFrontiers_ = std::move(frontiers);
}

MetatomicForceProvider::~MetatomicForceProvider() = default;

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


int32_t MetatomicForceProvider::exchangeBackwardGhosts(
        const gmx_domdec_t* dd, const matrix box, double cutoff)
{
    if (dd == nullptr || dd->ndim == 0)
    {
        return 0;
    }

    int32_t totalAdded = 0;
    std::unordered_set<int32_t> existingGlobalMta(
            mtaToGlobalMta_.begin(), mtaToGlobalMta_.end());

    for (int d = 0; d < dd->ndim; d++)
    {
        const int dimIndex  = dd->dim[d]; // Cartesian dimension
        const int numCellsD = dd->numCells[dimIndex];
        const int npulseD   = dd->numPulses[dimIndex];

        // The number of pulses needed to cover the "backward gap" left by GROMACS.
        // GROMACS covers npulseD cells in the forward direction only, so the
        // atoms within the cutoff behind the lower cell boundary always need
        // npulseD backward pulses, even when (with two cells) the backward
        // neighbour is the same rank as the forward one: those are different
        // atoms, on the other side of the cell. Keep the old count, all
        // domains not covered forward, as a lower bound.
        const int backwardGap = std::max(numCellsD - 1 - npulseD, numCellsD > 1 ? npulseD : 0);
        if (backwardGap <= 0)
        {
            continue;
        }

        // Forward boundary of this rank's cell (upper edge) in Cartesian coordinates.
        // DomdecZones::sizes(0) is the home zone.
        const double forwardBoundary = static_cast<double>(dd->zones.sizes(0).x1[dimIndex]);

        for (int p = 0; p < backwardGap; p++)
        {
            // Identify ALL currently local MTA atoms near the forward boundary (within cutoff).
            // We include ghosts from previous dimensions/pulses (staged communication)
            // to correctly cover diagonal and corner backward neighbors.
            std::vector<int>  sendGlobalMta;
            std::vector<RVec> sendPositions;
            std::vector<int>  sendAtomNumbers;

            for (int32_t i = 0; i < numLocalMta_; i++)
            {
                const double coord = static_cast<double>(positions_[i][dimIndex]);
                if (coord > forwardBoundary - cutoff)
                {
                    sendGlobalMta.push_back(static_cast<int>(mtaToGlobalMta_[i]));
                    sendPositions.push_back(positions_[i]);
                    sendAtomNumbers.push_back(static_cast<int>(atomNumbers_[i]));
                }
            }

            // --- Step 1: exchange counts ---
            int sendCount = static_cast<int>(sendGlobalMta.size());
            int recvCount = 0;
            ddSendrecv(dd, d, dddirForward,
                       gmx::ArrayRef<int>(&sendCount, &sendCount + 1),
                       gmx::ArrayRef<int>(&recvCount, &recvCount + 1));

            if (recvCount == 0 && sendCount == 0)
            {
                continue;
            }

            // --- Step 2: exchange data ---
            std::vector<int>  recvGlobalMta(recvCount);
            std::vector<RVec> recvPositions(recvCount);
            std::vector<int>  recvAtomNumbers(recvCount);

            ddSendrecv(dd, d, dddirForward,
                       gmx::ArrayRef<int>(sendGlobalMta),
                       gmx::ArrayRef<int>(recvGlobalMta));
            ddSendrecv(dd, d, dddirForward,
                       gmx::ArrayRef<RVec>(sendPositions),
                       gmx::ArrayRef<RVec>(recvPositions));
            ddSendrecv(dd, d, dddirForward,
                       gmx::ArrayRef<int>(sendAtomNumbers),
                       gmx::ArrayRef<int>(recvAtomNumbers));

            // PBC shift: when receiving from a rank that wrapped around the Periodic
            // Boundary (target index > our index while moving backward), shift the
            // received positions by -box[dim] to place them near our backward boundary.
            const int targetCell = (dd->ci[dimIndex] - p - 1 + numCellsD) % numCellsD;
            if (targetCell > dd->ci[dimIndex])
            {
                for (int k = 0; k < recvCount; k++)
                {
                    recvPositions[k][XX] -= box[dimIndex][XX];
                    recvPositions[k][YY] -= box[dimIndex][YY];
                    recvPositions[k][ZZ] -= box[dimIndex][ZZ];
                }
            }

            // Add non-duplicate backward ghost atoms
            int32_t addedInPulse = 0;
            for (int k = 0; k < recvCount; k++)
            {
                const int32_t globalMta = static_cast<int32_t>(recvGlobalMta[k]);
                if (existingGlobalMta.count(globalMta) == 0)
                {
                    positions_.push_back(recvPositions[k]);
                    atomNumbers_.push_back(static_cast<int32_t>(recvAtomNumbers[k]));
                    mtaToGlobalMta_.push_back(globalMta);
                    existingGlobalMta.insert(globalMta);
                    addedInPulse++;
                }
            }
            numLocalMta_ += addedInPulse;
            totalAdded += addedInPulse;
        }
    }

    return totalAdded;
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

    // Step 3: Existing canonical pairs — O(1) lookup via hashed set.
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

    // Step 4: Global MTA → local index mapping.
    std::unordered_map<int32_t, int32_t> globalToLocal;
    globalToLocal.reserve(numLocalMta_);
    for (int32_t i = 0; i < numLocalMta_; i++)
    {
        globalToLocal[mtaToGlobalMta_[i]] = i;
    }

    // Step 5: Ring exchange — P-1 rounds of MPI_Sendrecv.
    // Each round: send our pairs to rank+1, receive from rank-1.
    // Scan received pairs for those involving our home atoms.
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

        // Scan received pairs for those involving our home atoms.
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


void MetatomicForceProvider::logSiteAssignment(const std::vector<std::vector<int>>& bySite,
                                               const std::vector<int>&              siteSizes) const
{
    const auto               names = splitString(options_.params_.siteGroups);
    std::vector<std::string> perRank;
    for (size_t rank = 0; rank < bySite.size(); ++rank)
    {
        int                      numAtoms = 0;
        std::vector<std::string> rankNames;
        for (const int site : bySite[rank])
        {
            rankNames.push_back(names[site]);
            numAtoms += siteSizes[site];
        }
        perRank.push_back(rankNames.empty() ? formatString("rank %zu: none", rank)
                                            : formatString("rank %zu: %s (%d model atoms)",
                                                           rank,
                                                           joinStrings(rankNames, " ").c_str(),
                                                           numAtoms));
    }
    GMX_LOG(logger_.info)
            .asParagraph()
            .appendTextFormatted(
                    "Metatomic: %d ML sites on %zu ranks, each evaluated whole by one rank; %s",
                    options_.params_.numSites(),
                    bySite.size(),
                    joinStrings(perRank, "; ").c_str());
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

    // Newton NL mode: in parallel, each rank needs ALL pairs involving its
    // home atoms (not just the ones assigned by the eighth-shell DD
    // decomposition). Uses the GROMACS pairlist as the pair source, then
    // exchanges pair identities across ranks.
    const bool useNewtonNL = mpiComm_.isParallel();

    // Save original numLocalMta_ before potential backward ghost extension.
    // Must be restored after model evaluation so that gatherAtomPositions
    // on the next step does not access out-of-bounds mtaToGmxLocal_ entries.
    const int32_t origNumLocalMta = numLocalMta_;

    // Save original pairlist size; backward pairs are appended temporarily.
    const std::size_t origPairlistSize = pairlistMta_.size();
    const std::size_t origShiftsSize   = cellShiftsMta_.size();

    if (useNewtonNL)
    {
        // Compute max cutoff across all NL requests (used by both ghost
        // exchange and ring hop limit).
        double maxCutoff = 0.0;
        for (double cutoff : data_->cutoffNm)
        {
            maxCutoff = std::max(maxCutoff, cutoff);
        }

        GMX_UNUSED_VALUE(maxCutoff);
        // Step 1: Make all MTA atoms local. The DD halo covers only the forward direction,
        // while each home atom energy depends on all atoms within the model interaction
        // range (message passing). Extends positions_, atomNumbers_, mtaToGlobalMta_ and
        // numLocalMta_. (exchangeBackwardGhosts only covered a pairwise model.)
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

    // ONIOM link caps. The frontier holds global atom indices; resolve them to
    // model rows now, after any backward ghost exchange. The first cap on a
    // boundary MM atom takes over that atom's row, further caps on the same
    // MM atom get extra rows after the local atoms.
    const int32_t          nLocal = numLocalMta_;
    std::vector<int32_t>   modelTypes(atomNumbers_.begin(), atomNumbers_.begin() + nLocal);
    std::vector<int32_t>   chargeSource(nLocal); // row whose atom provides each row's charge
    std::iota(chargeSource.begin(), chargeSource.end(), 0);
    std::vector<ActiveLinkAtom> links;
    const PbcType pbcType = options_.params_.pbcType_ ? *options_.params_.pbcType_ : PbcType::Xyz;
    matrix        boxInv;
    clear_mat(boxInv);
    if (pbcType != PbcType::No)
    {
        invertBoxMatrix(inputs.box_, boxInv);
    }
    if (!linkFrontiers_.empty())
    {
        std::unordered_map<int32_t, int32_t> rowOfGlobalAtom;
        rowOfGlobalAtom.reserve(nLocal);
        for (int32_t i = 0; i < nLocal; ++i)
        {
            const int32_t mta = mtaToGlobalMta_[i];
            if (mta >= 0 && mta < static_cast<int32_t>(options_.params_.mtaIndices_.size()))
            {
                rowOfGlobalAtom.emplace(static_cast<int32_t>(options_.params_.mtaIndices_[mta]), i);
            }
        }
        std::unordered_set<int32_t> cappedMM;
        for (const auto& frontier : linkFrontiers_)
        {
            const auto embedded = rowOfGlobalAtom.find(frontier.getEmbeddedIndex());
            const auto mm       = rowOfGlobalAtom.find(frontier.getMMIndex());
            if (embedded == rowOfGlobalAtom.end() || mm == rowOfGlobalAtom.end())
            {
                continue; // not both on this rank
            }
            ActiveLinkAtom link;
            link.embedded     = embedded->second;
            link.mm           = mm->second;
            link.linkDistance = frontier.linkDistance();
            link.mmCellShift  = minimumImageCellShift(
                    boxInv, pbcType, positions_[link.mm] - positions_[link.embedded]);
            if (cappedMM.insert(link.mm).second)
            {
                link.row = link.mm;
            }
            else
            {
                link.row = static_cast<int32_t>(modelTypes.size());
                modelTypes.push_back(0);
                chargeSource.push_back(link.mm);
            }
            modelTypes[link.row] = frontier.linkAtomNumber();
            links.push_back(link);
        }
    }
    const int32_t nModel = static_cast<int32_t>(modelTypes.size());

    // Positions of every model row: local atoms, with caps in their rows.
    std::vector<RVec> capPositions;
    std::vector<char> isCapRow(nModel, 0);
    if (!links.empty())
    {
        capPositions.assign(positions_.begin(), positions_.begin() + nLocal);
        capPositions.resize(nModel);
        for (const auto& link : links)
        {
            capPositions[link.row] = linkAtomPosition(positions_[link.embedded],
                                                      positions_[link.mm],
                                                      cellShiftVector(inputs.box_, link.mmCellShift),
                                                      link.linkDistance);
            isCapRow[link.row] = 1;
        }
    }
    const auto modelPosition = [&](int32_t row) -> const RVec&
    { return links.empty() ? positions_[row] : capPositions[row]; };

    // Model inference
    double energy = 0.0;
    matrix virialMatrix;
    clear_mat(virialMatrix);

    {
        MetatomicTimer modelTimer("model inference", mpiComm_);
        MetatomicTimer tensorPrepTimer("tensorPrep", mpiComm_);

        const bool       modelIsDouble   = data_->useFloat64;
        const DLDataType floatType       = modelIsDouble ? DLDataType{ kDLFloat, 64, 1 }
                                                         : DLDataType{ kDLFloat, 32, 1 };
        const bool       gromacsIsDouble = std::is_same_v<real, double>;

        void* positionData = nullptr;
        void* cellData     = nullptr;
        if (modelIsDouble == gromacsIsDouble && links.empty())
        {
            positionData = positions_.data();
            cellData     = &box_[0][0];
        }
        else
        {
            const auto fill = [&](auto& positions, auto& cell)
            {
                using T = typename std::decay_t<decltype(positions)>::value_type;
                positions.resize(static_cast<size_t>(nModel) * 3);
                cell.resize(9);
                for (int32_t i = 0; i < nModel; ++i)
                {
                    for (int d = 0; d < 3; ++d)
                    {
                        positions[3 * i + d] = static_cast<T>(modelPosition(i)[d]);
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

        // ML sites (metatomic-site-groups): each site is its own system. A row's site is its
        // atom's, and a cap's that of the MM atom it replaces (which joined its ML partner's site).
        const int  numSites  = options_.params_.numSites();
        const bool multiSite = numSites > 1;
        if (multiSite && data_->nonConservative)
        {
            GMX_THROW(NotImplementedError(
                    "metatomic-site-groups with several sites needs a conservative model."));
        }
        std::vector<int>                  siteOfRow;
        std::vector<int32_t>              localRow;
        std::vector<std::vector<int32_t>> siteRows(multiSite ? numSites : 1);
        if (multiSite)
        {
            siteOfRow.resize(nModel);
            localRow.resize(nModel);
            for (int32_t row = 0; row < nModel; ++row)
            {
                const int32_t source = row < nLocal ? row : chargeSource[row];
                siteOfRow[row]       = options_.params_.mtaSites_[mtaToGlobalMta_[source]];
                localRow[row]        = static_cast<int32_t>(siteRows[siteOfRow[row]].size());
                siteRows[siteOfRow[row]].push_back(row);
            }
        }
        // The sites this rank evaluates, one system each. Serially, all of them. With domain
        // decomposition every rank holds every site (gatherAllMtaAtoms), and evaluates its share.
        std::vector<int> evaluatedSites(multiSite ? numSites : 1);
        std::iota(evaluatedSites.begin(), evaluatedSites.end(), 0);
        if (multiSite && useNewtonNL)
        {
            std::vector<int> siteSizes;
            for (const auto& rows : siteRows)
            {
                siteSizes.push_back(static_cast<int>(rows.size()));
            }
            const auto bySite = assignSitesToRanks(siteSizes, mpiComm_.size());
            evaluatedSites    = bySite[mpiComm_.rank()];
            if (!siteAssignmentLogged_)
            {
                logSiteAssignment(bySite, siteSizes);
                siteAssignmentLogged_ = true;
            }
        }
        std::vector<std::vector<int32_t>> systemRows;
        for (const int site : evaluatedSites)
        {
            systemRows.push_back(siteRows[site]);
        }
        // per-system copies of the types and positions, alive until the model has run
        std::vector<std::vector<int32_t>> siteTypes(systemRows.size());
        std::vector<std::vector<double>>  sitePositions64(systemRows.size());
        std::vector<std::vector<float>>   sitePositions32(systemRows.size());
        std::vector<metatomic::System>    systems;
        if (!multiSite)
        {
            systems.push_back(metatomic::System(
                    "nm",
                    wrap_dlpack(modelTypes.data(), { nModel }, { kDLInt, 32, 1 }),
                    wrap_dlpack(positionData, { nModel, 3 }, floatType),
                    wrap_dlpack(cellData, { 3, 3 }, floatType),
                    wrap_dlpack(data_->pbc.data(), { 3 }, { kDLBool, 8, 1 })));
        }
        else
        {
            for (size_t s = 0; s < systemRows.size(); ++s)
            {
                const auto&   rows = systemRows[s];
                const int64_t n    = static_cast<int64_t>(rows.size());
                void*         pos  = nullptr;
                const auto    copy = [&](auto* all, auto& out)
                {
                    out.resize(3 * rows.size());
                    for (size_t k = 0; k < rows.size(); ++k)
                    {
                        for (int d = 0; d < 3; ++d)
                        {
                            out[3 * k + d] = all[3 * rows[k] + d];
                        }
                    }
                    pos = out.data();
                };
                if (modelIsDouble)
                {
                    copy(static_cast<const double*>(positionData), sitePositions64[s]);
                }
                else
                {
                    copy(static_cast<const float*>(positionData), sitePositions32[s]);
                }
                for (const int32_t row : rows)
                {
                    siteTypes[s].push_back(modelTypes[row]);
                }
                systems.push_back(metatomic::System(
                        "nm",
                        wrap_dlpack(siteTypes[s].data(), { n }, { kDLInt, 32, 1 }),
                        wrap_dlpack(pos, { n, 3 }, floatType),
                        wrap_dlpack(cellData, { 3, 3 }, floatType),
                        wrap_dlpack(data_->pbc.data(), { 3 }, { kDLBool, 8, 1 })));
            }
        }

        // Topology charges; a cap carries the charge of the MM atom it replaces.
        for (const auto& input : data_->chargeInputs)
        {
            const double toUnit =
                    input.unit().empty() ? 1.0 : metatomic::unit_conversion_factor("e", input.unit());
            std::vector<double> charges(nModel);
            for (int32_t i = 0; i < nModel; ++i)
            {
                const int32_t mta = mtaToGlobalMta_[chargeSource[i]];
                if (mta < 0 || mta >= static_cast<int32_t>(options_.params_.mtaIndices_.size()))
                {
                    GMX_THROW(InconsistentInputError(
                            "Metatomic charge input contains an invalid atom index."));
                }
                const Index globalAtom = options_.params_.mtaIndices_[mta];
                if (globalAtom < 0 || globalAtom >= static_cast<Index>(options_.params_.mmCharges_.size()))
                {
                    GMX_THROW(InconsistentInputError(
                            "Metatomic charge input contains an atom without a stored charge."));
                }
                charges[i] = toUnit * options_.params_.mmCharges_[globalAtom];
            }
            for (size_t s = 0; s < systems.size(); ++s)
            {
                std::vector<double> systemCharges;
                if (!multiSite)
                {
                    systemCharges = charges;
                }
                else
                {
                    for (const int32_t row : systemRows[s])
                    {
                        systemCharges.push_back(charges[row]);
                    }
                }
                const auto           n = static_cast<int32_t>(systemCharges.size());
                std::vector<int32_t> sampleValues(static_cast<size_t>(n) * 2, 0);
                for (int32_t i = 0; i < n; ++i)
                {
                    sampleValues[2 * i + 1] = i;
                }
                const int32_t zero  = 0;
                const auto    shape = std::vector<uintptr_t>{ static_cast<uintptr_t>(n), 1 };
                std::unique_ptr<metatensor::DataArrayBase> values;
                if (modelIsDouble)
                {
                    values = std::make_unique<metatensor::SimpleDataArray<double>>(shape, std::move(systemCharges));
                }
                else
                {
                    values = std::make_unique<metatensor::SimpleDataArray<float>>(
                            shape, std::vector<float>(systemCharges.begin(), systemCharges.end()));
                }
                std::vector<metatensor::TensorBlock> blocks;
                blocks.emplace_back(std::move(values),
                                    metatensor::Labels({ "system", "atom" },
                                                       n == 0 ? nullptr : sampleValues.data(),
                                                       static_cast<size_t>(n)),
                                    std::vector<metatensor::Labels>{},
                                    metatensor::Labels({ "charge" }, &zero, 1));
                systems[s].add_custom_data(input.name(),
                                           metatensor::TensorMap(metatensor::Labels({ "_" }, &zero, 1),
                                                                 std::move(blocks)));
            }
        }

        // Per-system total charge and spin multiplicity, from the site options
        for (const auto& input : data_->systemInputs)
        {
            const bool   spin   = input.name() == "spin_multiplicity";
            const double toUnit = spin || input.unit().empty()
                                          ? 1.0
                                          : metatomic::unit_conversion_factor("e", input.unit());
            for (size_t s = 0; s < systems.size(); ++s)
            {
                const int     site  = evaluatedSites[s];
                const double  value = spin ? options_.params_.siteSpinValues_[site]
                                           : toUnit * options_.params_.siteChargeValues_[site];
                const int32_t zero  = 0;
                const auto    shape = std::vector<uintptr_t>{ 1, 1 };
                std::unique_ptr<metatensor::DataArrayBase> values;
                if (modelIsDouble)
                {
                    values = std::make_unique<metatensor::SimpleDataArray<double>>(shape, std::vector<double>{ value });
                }
                else
                {
                    values = std::make_unique<metatensor::SimpleDataArray<float>>(
                            shape, std::vector<float>{ static_cast<float>(value) });
                }
                std::vector<metatensor::TensorBlock> blocks;
                blocks.emplace_back(std::move(values),
                                    metatensor::Labels({ "system" }, &zero, 1),
                                    std::vector<metatensor::Labels>{},
                                    metatensor::Labels({ spin ? "spin_multiplicity" : "charge" }, &zero, 1));
                systems[s].add_custom_data(input.name(),
                                           metatensor::TensorMap(metatensor::Labels({ "_" }, &zero, 1),
                                                                 std::move(blocks)));
            }
        }
        tensorPrepTimer.stop();

        MetatomicTimer buildNLTimer("buildNL", mpiComm_);
        std::vector<int32_t> capRows;
        for (const auto& link : links)
        {
            capRows.push_back(link.row);
        }
        for (size_t requestIndex = 0; requestIndex < data_->pairLists.size(); ++requestIndex)
        {
            const auto&   request = data_->pairLists[requestIndex];
            const bool    full    = request.full_list();
            const bool    strict  = request.strict();
            const double  cutoff  = data_->cutoffNm[requestIndex];
            const double  cutoff2 = cutoff * cutoff;
            const int64_t nHalf   = static_cast<int64_t>(pairlistMta_.size() / 2);
            nlSamplesBuffer_.clear();
            nlVectorsBuffer_.clear();
            nlSamplesBuffer_.reserve(static_cast<size_t>((full ? 2 : 1) * nHalf) * 5);
            nlVectorsBuffer_.reserve(static_cast<size_t>((full ? 2 : 1) * nHalf) * 3);

            const auto addPair = [&](int32_t ai, int32_t aj, const IVec& n, double dx, double dy, double dz)
            {
                nlSamplesBuffer_.insert(nlSamplesBuffer_.end(), { ai, aj, n[XX], n[YY], n[ZZ] });
                nlVectorsBuffer_.insert(nlVectorsBuffer_.end(), { dx, dy, dz });
                if (full)
                {
                    nlSamplesBuffer_.insert(nlSamplesBuffer_.end(), { aj, ai, -n[XX], -n[YY], -n[ZZ] });
                    nlVectorsBuffer_.insert(nlVectorsBuffer_.end(), { -dx, -dy, -dz });
                }
            };

            for (int64_t k = 0; k < nHalf; ++k)
            {
                const int32_t ai = pairlistMta_[2 * k];
                const int32_t aj = pairlistMta_[2 * k + 1];
                if (!links.empty() && (isCapRow[ai] || isCapRow[aj]))
                {
                    continue; // the GROMACS pair is for the MM atom, not its cap
                }
                const IVec& n     = cellShiftsMta_[k];
                const RVec  shift = cellShiftVector(inputs.box_, n);
                const double dx = static_cast<double>(positions_[aj][0] - positions_[ai][0] + shift[0]);
                const double dy = static_cast<double>(positions_[aj][1] - positions_[ai][1] + shift[1]);
                const double dz = static_cast<double>(positions_[aj][2] - positions_[ai][2] + shift[2]);
                if (strict && dx * dx + dy * dy + dz * dz > cutoff2)
                {
                    continue;
                }
                addPair(ai, aj, n, dx, dy, dz);
            }

            // Cap pairs, from the cap positions (minimum image), always within the cutoff.
            for (const int32_t ai : capRows)
            {
                for (int32_t aj = 0; aj < nModel; ++aj)
                {
                    if (aj == ai || (isCapRow[aj] && aj < ai))
                    {
                        continue; // cap-cap pairs once
                    }
                    const RVec   raw   = modelPosition(aj) - modelPosition(ai);
                    const IVec   n     = minimumImageCellShift(boxInv, pbcType, raw);
                    const RVec   shift = cellShiftVector(inputs.box_, n);
                    const double dx    = static_cast<double>(raw[XX] + shift[XX]);
                    const double dy    = static_cast<double>(raw[YY] + shift[YY]);
                    const double dz    = static_cast<double>(raw[ZZ] + shift[ZZ]);
                    if (dx * dx + dy * dy + dz * dz <= cutoff2)
                    {
                        addPair(ai, aj, n, dx, dy, dz);
                    }
                }
            }
            const int32_t xyzValues[3]  = { 0, 1, 2 };
            const int32_t distanceValue = 0;
            const auto    addBlock      = [&](metatomic::System&          target,
                                       const std::vector<int32_t>& sampleValues,
                                       const std::vector<double>&  vectors)
            {
                const int64_t nPairs  = static_cast<int64_t>(vectors.size() / 3);
                auto          samples = metatensor::Labels(
                        data_->nlSampleNames,
                        nPairs == 0 ? nullptr : sampleValues.data(),
                        static_cast<size_t>(nPairs));
                std::vector<metatensor::Labels> components;
                components.push_back(metatensor::Labels({ "xyz" }, xyzValues, 3));
                auto properties = metatensor::Labels({ "distance" }, &distanceValue, 1);

                std::unique_ptr<metatensor::DataArrayBase> values;
                const auto shape = std::vector<uintptr_t>{ static_cast<uintptr_t>(nPairs), 3, 1 };
                if (modelIsDouble)
                {
                    values = std::make_unique<metatensor::SimpleDataArray<double>>(shape, vectors);
                }
                else
                {
                    values = std::make_unique<metatensor::SimpleDataArray<float>>(
                            shape, std::vector<float>(vectors.begin(), vectors.end()));
                }
                target.add_pairs(request,
                                 metatensor::TensorBlock(std::move(values), samples, components, properties));
            };
            if (!multiSite)
            {
                addBlock(systems[0], nlSamplesBuffer_, nlVectorsBuffer_);
            }
            else
            {
                // pairs within a site only, in the site's row numbering
                std::vector<std::vector<int32_t>> siteSamples(numSites);
                std::vector<std::vector<double>>  siteVectors(numSites);
                const size_t nPairs = nlVectorsBuffer_.size() / 3;
                for (size_t k = 0; k < nPairs; ++k)
                {
                    const int32_t ai = nlSamplesBuffer_[5 * k];
                    const int32_t aj = nlSamplesBuffer_[5 * k + 1];
                    const int     s  = siteOfRow[ai];
                    if (siteOfRow[aj] != s)
                    {
                        continue;
                    }
                    siteSamples[s].insert(siteSamples[s].end(),
                                          { localRow[ai], localRow[aj], nlSamplesBuffer_[5 * k + 2],
                                            nlSamplesBuffer_[5 * k + 3], nlSamplesBuffer_[5 * k + 4] });
                    siteVectors[s].insert(siteVectors[s].end(),
                                          nlVectorsBuffer_.begin() + 3 * k,
                                          nlVectorsBuffer_.begin() + 3 * k + 3);
                }
                for (size_t s = 0; s < systems.size(); ++s)
                {
                    addBlock(systems[s], siteSamples[evaluatedSites[s]], siteVectors[evaluatedSites[s]]);
                }
            }
        }
        buildNLTimer.stop();

        // With domain decomposition, each rank returns its home atoms plus the
        // caps whose embedded atom is home; a home MM row whose caps all belong
        // to non-home embedded atoms is left to the rank owning them. With
        // several sites, each rank returns whole sites instead.
        std::optional<metatensor::Labels> selected;
        if (!multiSite && useNewtonNL && links.empty())
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
        else if (!multiSite && useNewtonNL)
        {
            std::vector<char> keep(nModel, 0);
            std::fill(keep.begin(), keep.begin() + numHomeMta_, 1);
            for (const auto& link : links)
            {
                if (link.row < numHomeMta_)
                {
                    keep[link.row] = 0;
                }
            }
            for (const auto& link : links)
            {
                if (link.embedded < numHomeMta_)
                {
                    keep[link.row] = 1;
                }
            }
            std::vector<int32_t> rows;
            for (int32_t i = 0; i < nModel; ++i)
            {
                if (keep[i])
                {
                    rows.insert(rows.end(), { 0, i });
                }
            }
            selected = metatensor::Labels(
                    { "system", "atom" }, rows.empty() ? nullptr : rows.data(), rows.size() / 2);
            data_->selectedCount = -1; // the cached home-only selection is stale
        }

        // A rank without sites has nothing to evaluate, but still joins the force exchange.
        MetatomicTimer forwardTimer("execute_model", mpiComm_);
        std::vector<metatensor::TensorMap> modelOutputs;
        if (!systems.empty())
        {
            try
            {
                modelOutputs = metatomic::execute_model(
                        *data_->model, systems, selected, data_->requested, data_->checkConsistency);
            }
            catch (const std::exception& e)
            {
                GMX_THROW(APIError("[Metatomic] Model evaluation failed: " + std::string(e.what())));
            }
        }
        forwardTimer.stop();
        const bool evaluated = !modelOutputs.empty();

        if (evaluated && data_->uqIndex.has_value())
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

        std::vector<double> forces(static_cast<size_t>(nModel) * 3, 0.0);
        const auto nEnergyBlocks = evaluated ? modelOutputs[data_->energyIndex].keys().count() : 0;
        for (size_t blockId = 0; blockId < nEnergyBlocks; ++blockId)
        {
            auto block = modelOutputs[data_->energyIndex].block_by_id(blockId);
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
            // metatomic converts the energy to kJ/mol, but the positions gradient stays per
            // model length unit: kJ/mol/Angstrom for a model in Angstrom, 10x smaller than
            // kJ/mol/nm. The strain gradient is per (dimensionless) strain.
            const double forceScale = -1.0 / data_->lengthToNm;
            if (modelIsDouble)
            {
                add_atom_rows<double>(
                        posGrad, 2, nModel, forceScale, &forces, multiSite ? &systemRows : nullptr);
                add_strain<double>(strainGrad, virialMatrix);
            }
            else
            {
                add_atom_rows<float>(
                        posGrad, 2, nModel, forceScale, &forces, multiSite ? &systemRows : nullptr);
                add_strain<float>(strainGrad, virialMatrix);
            }
        }

        if (evaluated && data_->nonConservative)
        {
            auto&      forceMap = modelOutputs[data_->ncForceIndex.value()];
            const auto nBlocks  = forceMap.keys().count();
            for (size_t blockId = 0; blockId < nBlocks; ++blockId)
            {
                auto block = forceMap.block_by_id(blockId);
                if (modelIsDouble)
                {
                    add_atom_rows<double>(block, 1, nModel, 1.0, &forces);
                }
                else
                {
                    add_atom_rows<float>(block, 1, nModel, 1.0, &forces);
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

        // Spread each cap's force onto its embedded and MM atoms (exact chain
        // rule for r_cap = r_emb + d u, u the unit embedded-MM bond). A cap row
        // holds only the cap: the first cap on an MM atom replaced that atom.
        // The model's strain derivative moved the cap with the box; the cap
        // keeps its bond length d instead, which changes the virial by
        // 0.5 d (u.F_cap) u (x) u per cap.
        if (!links.empty())
        {
            std::vector<RVec> capForces(links.size());
            for (size_t k = 0; k < links.size(); ++k)
            {
                for (int d = 0; d < 3; ++d)
                {
                    capForces[k][d] = static_cast<real>(forces[3 * links[k].row + d]);
                }
            }
            for (const auto& link : links)
            {
                std::fill_n(forces.begin() + 3 * link.row, 3, 0.0);
            }
            for (size_t k = 0; k < links.size(); ++k)
            {
                const auto& link    = links[k];
                const RVec  mmShift = cellShiftVector(inputs.box_, link.mmCellShift);
                const auto [onEmbedded, onMM] = spreadLinkAtomForce(
                        capForces[k], positions_[link.embedded], positions_[link.mm], mmShift, link.linkDistance);
                for (int d = 0; d < 3; ++d)
                {
                    forces[3 * link.embedded + d] += static_cast<double>(onEmbedded[d]);
                    forces[3 * link.mm + d] += static_cast<double>(onMM[d]);
                }
                if (!data_->nonConservative)
                {
                    const RVec   u     = unitVector(positions_[link.mm] + mmShift - positions_[link.embedded]);
                    const double scale = 0.5 * link.linkDistance * static_cast<double>(dot(u, capForces[k]));
                    for (int a = 0; a < 3; ++a)
                    {
                        for (int b = 0; b < 3; ++b)
                        {
                            virialMatrix[a][b] += static_cast<real>(scale * u[a] * u[b]);
                        }
                    }
                }
            }
        }

        // Only the first nLocal rows are real atoms from here on.
        MetatomicTimer forceScatterTimer("forceScatter", mpiComm_);
        const double*  forceData = forces.data();
        if (!mpiComm_.isParallel() || (data_->nonConservative && links.empty()))
        {
            // Serial, or non-conservative outputs for home atoms only.
            const int32_t nApply = mpiComm_.isParallel() ? numHomeMta_ : nLocal;
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
            // Caps can put force on halo MM atoms, which the owning rank must receive.
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
