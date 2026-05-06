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
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "gromacs/domdec/domdec_network.h"
#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/domdec/localatomset.h"
#include "gromacs/math/boxmatrix.h"
#include "gromacs/mdlib/broadcaststructs.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
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

#include <metatensor/torch.hpp>
#include <metatomic/torch.hpp>

#if GMX_GPU_CUDA || (GMX_SYCL_ACPP && GMX_ACPP_HAVE_CUDA_TARGET)
#    include <cuda_runtime.h>
#endif


namespace gmx
{

/*! \brief Normalizes the variant string for Metatomic output selection. */
static torch::optional<std::string> normalize_variant(std::string variant_string)
{
    if (variant_string == "no" || variant_string.empty())
    {
        return torch::nullopt;
    }
    else
    {
        return variant_string;
    }
}

/*! \brief Converts GROMACS PbcType to a boolean tensor for Metatomic. */
static torch::Tensor preparePbcType(PbcType* pbcType, torch::Device device)
{
    auto options = torch::TensorOptions().dtype(torch::kBool).device(device);

    if (*pbcType == PbcType::XY)
    {
        return torch::tensor({ true, true, false }, options);
    }
    else if (*pbcType == PbcType::No)
    {
        return torch::tensor({ false, false, false }, options);
    }
    return torch::tensor({ true, true, true }, options);
}

/*! \brief Whether a requested model input uses GROMACS atom charges. */
static bool isChargeInput(const std::string& name)
{
    return name == "charges" || name.rfind("charges/", 0) == 0;
}

struct ActiveLinkAtom
{
    int32_t embeddedModelIndex = -1;
    int32_t mmModelIndex       = -1;
    int32_t linkModelIndex     = -1;
    IVec    mmCellShift        = IVec(0, 0, 0);
    real    linkDistance       = 0;
    int32_t linkAtomNumber     = 1;
};

static RVec computeLinkAtomPosition(const RVec& embeddedPosition,
                                    const RVec& mmPosition,
                                    const RVec& mmShift,
                                    real        linkDistance)
{
    const double dx = static_cast<double>(mmPosition[XX] - embeddedPosition[XX] + mmShift[XX]);
    const double dy = static_cast<double>(mmPosition[YY] - embeddedPosition[YY] + mmShift[YY]);
    const double dz = static_cast<double>(mmPosition[ZZ] - embeddedPosition[ZZ] + mmShift[ZZ]);
    const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (dist == 0.0)
    {
        GMX_THROW(InconsistentInputError(
                "Metatomic link atom construction found a zero-length boundary bond."));
    }

    RVec linkPosition;
    linkPosition[XX] = embeddedPosition[XX] + linkDistance * dx / dist;
    linkPosition[YY] = embeddedPosition[YY] + linkDistance * dy / dist;
    linkPosition[ZZ] = embeddedPosition[ZZ] + linkDistance * dz / dist;
    return linkPosition;
}

static IVec computeMinimumImageCellShift(const matrix   boxInv,
                                         PbcType        pbcType,
                                         const double   rawDx,
                                         const double   rawDy,
                                         const double   rawDz)
{
    IVec cellShift(0, 0, 0);
    if (pbcType == PbcType::No)
    {
        return cellShift;
    }

    cellShift[XX] = static_cast<int>(
            std::round(-(boxInv[XX][XX] * rawDx + boxInv[YY][XX] * rawDy
                         + boxInv[ZZ][XX] * rawDz)));
    cellShift[YY] = static_cast<int>(
            std::round(-(boxInv[YY][YY] * rawDy + boxInv[ZZ][YY] * rawDz)));
    if (pbcType != PbcType::XY)
    {
        cellShift[ZZ] = static_cast<int>(std::round(-(boxInv[ZZ][ZZ] * rawDz)));
    }
    return cellShift;
}

static RVec computeCellShiftVector(const matrix box, const IVec& cellShift)
{
    RVec shift;
    mvmul_ur0(box, cellShift.toRVec(), shift);
    return shift;
}

/*! \brief Internal data structure for Metatomic runtime states. */
struct MetatomicData
{
    metatensor_torch::Module           model = metatensor_torch::Module(torch::jit::Module());
    metatomic_torch::ModelCapabilities capabilities;
    std::vector<metatomic_torch::NeighborListOptions> nl_requests;
    metatomic_torch::ModelEvaluationOptions           evaluations_options;
    torch::ScalarType                                 dtype             = torch::kFloat32;
    bool                                              check_consistency = false;
    torch::Device                                     device            = torch::kCPU;
    //! Requested Metatomic per-atom charge input names.
    std::vector<std::string> requestedChargeInputs;

    //! Cached NL Labels that are identical every step (created once in constructor).
    metatensor_torch::Labels cachedNLComponent;
    metatensor_torch::Labels cachedNLProperties;
    //! Cached sample column names (avoids heap-allocating string vector every step).
    std::vector<std::string> nlSampleNames = {
            "first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"
    };

    //! Cached types and PBC tensors (re-created on AtomsRedistributed).
    torch::Tensor cachedTypes;
    torch::Tensor cachedPbc;

    //! Sparse/dense force exchange threshold (atom count). Env: GMX_METATOMIC_SPARSE_THRESHOLD.
    int32_t sparseThreshold = 1000;

    //! Energy uncertainty output key (empty if disabled or model lacks it).
    std::string energy_uq_key;
    //! Requested uncertainty output (nullptr if disabled).
    metatomic_torch::ModelOutput uncertainty_output;
    //! Uncertainty threshold in kJ/mol. Atoms above this trigger a warning.
    double uncertaintyThreshold = 0.0;

    //! Link frontier atoms for ONIOM link atom support.
    std::vector<LinkFrontierAtom> linkFrontier;

    //! Non-conservative mode: forces/stress predicted directly, no backward pass.
    bool nonConservative = false;
    //! Output keys for non-conservative forces and stress.
    std::string nc_forces_key;
    std::string nc_stress_key;
    //! Requested outputs for non-conservative mode.
    metatomic_torch::ModelOutput nc_forces_output;
    metatomic_torch::ModelOutput nc_stress_output;
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

    try
    {
        torch::optional<std::string> extensions_directory = torch::nullopt;
        if (!options_.params_.extensionsDirectory.empty())
        {
            extensions_directory = options_.params_.extensionsDirectory;
        }

        data_->model = metatomic_torch::load_atomistic_model(options_.params_.modelPath_,
                                                             extensions_directory);
    }
    catch (const std::exception& e)
    {
        GMX_THROW(APIError("Failed to load metatomic model: " + std::string(e.what())));
    }

    data_->capabilities =
            data_->model.run_method("capabilities").toCustomClass<metatomic_torch::ModelCapabilitiesHolder>();

    torch::optional<std::string> desiredDevice = torch::nullopt;
    if (!options_.params_.device.empty())
    {
        desiredDevice = options_.params_.device;
    }
    if (const char* env = std::getenv("GMX_METATOMIC_DEVICE"))
    {
        desiredDevice = std::string(env);
    }

    const auto deviceType =
            metatomic_torch::pick_device(data_->capabilities->supported_devices, desiredDevice);
    data_->device = torch::Device(deviceType);

    // Set PyTorch intra-op thread count based on device and MPI mode.
    // For GPU devices, CPU overhead is minimal so we keep 1 thread to avoid
    // oversubscription with GROMACS threads.  For CPU devices, model inference
    // (matmuls, convolutions) benefits from multi-threading.
    // Synchronize PyTorch's thread count with GROMACS's ntomp.
    //
    // at::set_num_threads updates PyTorch's cached thread count AND
    // MKL's thread pool (via mkl_set_num_threads), while GROMACS's
    // omp_set_num_threads only updates the OpenMP ICV.  Without this
    // call, PyTorch/MKL may retain the init-time default (all cores).
    //
    // For real MPI, ntomp is the GROMACS-selected per-rank thread count.
    {
        int ntomp = gmx_omp_nthreads_get(ModuleMultiThread::Default);
        at::set_num_threads(std::max(1, ntomp));
    }

    // JIT fusion: dynamic strategy with depth limit of 10 improves CPU
    // inference throughput (matches LAMMPS pair_metatomic).
    torch::jit::FusionStrategy strategy = { { torch::jit::FusionBehavior::DYNAMIC, 10 } };
    torch::jit::setFusionStrategy(strategy);

    // Allow disabling graph optimization when it is counterproductive.
    if (const char* jitEnv = std::getenv("GMX_METATOMIC_DISABLE_TORCH_JIT_OPTIMIZATION"))
    {
        if (std::string(jitEnv) == "1")
        {
            torch::jit::setGraphExecutorOptimize(false);
            GMX_LOG(logger_.info)
                    .asParagraph()
                    .appendText("Metatomic TorchScript graph optimization disabled");
        }
    }

    GMX_LOG(logger_.info)
            .asParagraph()
            .appendTextFormatted("Metatomic PyTorch threads: %d", at::get_num_threads());

    // Cache NL Labels that are constant across steps (avoids per-step
    // string vector + tensor allocation for component and properties).
    auto devIntOpts = torch::TensorOptions().dtype(torch::kInt32).device(data_->device);
    data_->cachedNLComponent = torch::make_intrusive<metatensor_torch::LabelsHolder>(
            std::vector<std::string>{ "xyz" },
            torch::tensor({ 0, 1, 2 }, devIntOpts).reshape({ 3, 1 }));
    data_->cachedNLProperties = torch::make_intrusive<metatensor_torch::LabelsHolder>(
            std::vector<std::string>{ "distance" },
            torch::zeros({ 1, 1 }, devIntOpts));

    GMX_LOG(logger_.info)
            .asParagraph()
            .appendTextFormatted("Metatomic using device: %s", data_->device.str().c_str());

    auto requests_ivalue = data_->model.run_method("requested_neighbor_lists");
    for (const auto& request_ivalue : requests_ivalue.toList())
    {
        auto nl_opt = request_ivalue.get().toCustomClass<metatomic_torch::NeighborListOptionsHolder>();
        data_->nl_requests.push_back(nl_opt);
    }

    data_->model.to(data_->device);

    if (data_->capabilities->dtype() == "float64")
    {
        data_->dtype = torch::kFloat64;
    }
    else if (data_->capabilities->dtype() == "float32")
    {
        data_->dtype = torch::kFloat32;
    }
    else
    {
        GMX_THROW(APIError("Unsupported dtype from model capabilities: " + data_->capabilities->dtype()));
    }

    data_->evaluations_options = torch::make_intrusive<metatomic_torch::ModelEvaluationOptionsHolder>();
    data_->evaluations_options->set_length_unit("nm");

    auto requestedInputs = data_->model.run_method("requested_inputs").toGenericDict();
    for (const auto& entry : requestedInputs)
    {
        std::string inputName = entry.key().toStringRef();
        if (isChargeInput(inputName))
        {
            data_->requestedChargeInputs.push_back(inputName);
        }
    }

    auto outputs    = data_->capabilities->outputs();
    auto v_energy   = normalize_variant(options_.params_.variant);
    auto energy_key = pick_output("energy", outputs, v_energy);

    if (!outputs.contains(energy_key))
    {
        GMX_THROW(
                APIError(formatString("The model at '%s' does not provide an '%s' output. "
                                      "Metatomic interface cannot proceed.",
                                      options_.params_.modelPath_.c_str(),
                                      energy_key.c_str())));
    }

    auto model_output     = outputs.at(energy_key);
    auto requested_output = torch::make_intrusive<metatomic_torch::ModelOutputHolder>();
    // Use the model's declared per_atom capability (needed for correct energy
    // decomposition when using the GROMACS pairlist in domain decomposition)
    requested_output->per_atom           = model_output->per_atom;
    if (!model_output->per_atom)
    {
        GMX_LOG(logger_.warning)
                .asParagraph()
                .appendText(
                        "Metatomic model does not support per_atom energy output. "
                        "Energy decomposition in domain decomposition may be less accurate.");
    }
    requested_output->explicit_gradients = {};
    requested_output->set_quantity("energy");
    requested_output->set_unit("kJ/mol");

    data_->evaluations_options->outputs.insert(energy_key, requested_output);
    data_->check_consistency = options_.params_.checkConsistency;

    // Uncertainty checking: auto-detect energy_uncertainty output from model
    if (options_.params_.uncertaintyThreshold != "off")
    {
        auto v_energy_uq = normalize_variant(options_.params_.variantEnergyUq);
        bool has_uncertainty = false;
        for (const auto& entry : outputs)
        {
            if (entry.key().find("energy_uncertainty") == 0)
            {
                has_uncertainty = true;
                break;
            }
        }

        if (has_uncertainty)
        {
            data_->energy_uq_key = pick_output("energy_uncertainty", outputs, v_energy_uq);
            auto uq_cap = outputs.at(data_->energy_uq_key);

            if (uq_cap->per_atom)
            {
                data_->uncertainty_output =
                        torch::make_intrusive<metatomic_torch::ModelOutputHolder>();
                data_->uncertainty_output->set_quantity("energy");
                data_->uncertainty_output->set_unit("kJ/mol");
                data_->uncertainty_output->per_atom = true;

                if (options_.params_.uncertaintyThreshold == "auto")
                {
                    // Default: 100 meV/atom converted to kJ/mol
                    data_->uncertaintyThreshold =
                            0.1 * metatomic_torch::unit_conversion_factor("energy", "eV", "kJ/mol");
                }
                else
                {
                    data_->uncertaintyThreshold =
                            std::stod(options_.params_.uncertaintyThreshold);
                }

                data_->evaluations_options->outputs.insert(
                        data_->energy_uq_key, data_->uncertainty_output);

                GMX_LOG(logger_.info)
                        .asParagraph()
                        .appendTextFormatted(
                                "Metatomic: found '%s' output, will check for atoms with "
                                "high uncertainty (threshold: %.4f kJ/mol)",
                                data_->energy_uq_key.c_str(),
                                data_->uncertaintyThreshold);
            }
        }
    }

    // Non-conservative mode: model predicts forces/stress directly
    data_->nonConservative = options_.params_.nonConservative;
    if (data_->nonConservative)
    {
        auto v_nc_forces = normalize_variant(options_.params_.variantNcForces);
        auto v_nc_stress = normalize_variant(options_.params_.variantNcStress);

        // Both variant overrides must match if both are set (LAMMPS convention)
        if (v_nc_forces.has_value() && v_nc_stress.has_value()
            && v_nc_forces.value() != v_nc_stress.value())
        {
            GMX_THROW(APIError(
                    "if both 'variant-nc-forces' and 'variant-nc-stress' are present, "
                    "they must have the same value"));
        }

        data_->nc_forces_key = pick_output("non_conservative_forces", outputs, v_nc_forces);
        if (!outputs.contains(data_->nc_forces_key))
        {
            GMX_THROW(APIError(formatString(
                    "The model does not provide '%s' output, "
                    "we can not enable non-conservative simulations",
                    data_->nc_forces_key.c_str())));
        }
        auto nc_forces_cap = outputs.at(data_->nc_forces_key);
        if (!nc_forces_cap->per_atom)
        {
            GMX_THROW(APIError(formatString(
                    "The model's '%s' output can not produce per-atom output, "
                    "we can not enable non-conservative simulations",
                    data_->nc_forces_key.c_str())));
        }

        data_->nc_forces_output = torch::make_intrusive<metatomic_torch::ModelOutputHolder>();
        data_->nc_forces_output->set_quantity("force");
        data_->nc_forces_output->set_unit("kJ/mol/nm");
        data_->nc_forces_output->per_atom = true;

        data_->evaluations_options->outputs.insert(
                data_->nc_forces_key, data_->nc_forces_output);

        data_->nc_stress_key = pick_output("non_conservative_stress", outputs, v_nc_stress);
        if (outputs.contains(data_->nc_stress_key))
        {
            data_->nc_stress_output = torch::make_intrusive<metatomic_torch::ModelOutputHolder>();
            data_->nc_stress_output->set_quantity("stress");
            data_->nc_stress_output->set_unit("kJ/mol/nm^3");
            data_->nc_stress_output->per_atom = false;

            data_->evaluations_options->outputs.insert(
                    data_->nc_stress_key, data_->nc_stress_output);
        }

        GMX_LOG(logger_.info)
                .asParagraph()
                .appendTextFormatted(
                        "Metatomic: non-conservative mode enabled. Forces from '%s', "
                        "stress from '%s'",
                        data_->nc_forces_key.c_str(),
                        data_->nc_stress_key.c_str());
    }

    // Store link frontier from preprocessing
    data_->linkFrontier = options_.params_.linkFrontier_;
    if (!data_->linkFrontier.empty())
    {
        GMX_LOG(logger_.info)
                .asParagraph()
                .appendTextFormatted("Metatomic: %zu link atoms at ML/MM boundary",
                                     data_->linkFrontier.size());
    }

    GMX_LOG(logger_.info)
            .asParagraph()
            .appendText("MetatomicForceProvider initialization complete.");
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

    // Update cached tensors for the new atom distribution
    data_->cachedTypes =
            torch::tensor(atomNumbers_, torch::TensorOptions().dtype(torch::kInt32)).to(data_->device);
    data_->cachedPbc = preparePbcType(options_.params_.pbcType_.get(), data_->device);

}

void MetatomicForceProvider::gatherAtomPositions(ArrayRef<const RVec> pos)
{
    positions_.resize(numLocalMta_);
    for (int32_t i = 0; i < numLocalMta_; i++)
    {
        positions_[i] = pos[mtaToGmxLocal_[i]];
    }
}

/*! \brief Convert GROMACS excluded pairlist to MTA model indices.
 *
 * Called on every PairlistConstructed signal. Maps GROMACS local buffer
 * indices in excludedPairlist_ to MTA model indices via gmxLocalToMtaIdx_,
 * and negates cell shifts (GROMACS shifts first atom, metatensor shifts
 * second atom).
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
    for (const auto& entry : signal.excludedPairlist_)
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
        // GROMACS covers npulseD in the forward direction. The remaining domains
        // in that dimension must be filled by us.
        const int backwardGap = numCellsD - 1 - npulseD;
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
            // We include staged ghosts from dimensions and pulses already processed
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

    // Step 2: Build set of my home atoms.
    std::unordered_set<int32_t> myHomeGlobalMta;
    for (int32_t i = 0; i < numHomeMta_; i++)
    {
        myHomeGlobalMta.insert(mtaToGlobalMta_[i]);
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

            const bool iIsMyHome = myHomeGlobalMta.count(gI) > 0;
            const bool jIsMyHome = myHomeGlobalMta.count(gJ) > 0;
            if (!iIsMyHome && !jIsMyHome)
            {
                continue;
            }

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
        for (const auto& req : data_->nl_requests)
        {
            maxCutoff = std::max(maxCutoff, req->engine_cutoff("nm"));
        }

        // Step 1: Exchange backward ghost atoms to fill the backward gap
        // in the DD halo.  Extends positions_, atomNumbers_, mtaToGlobalMta_
        // and numLocalMta_ with atoms from the backward PBC neighbor.
        {
            MetatomicTimer timer("exchangeBackwardGhosts", mpiComm_);
            exchangeBackwardGhosts(inputs.dd_, inputs.box_, maxCutoff);
        }

        // Step 2: Exchange backward-direction pairs.  Discovers pairs from
        // other ranks' pairlists that involve this rank's home atoms.
        // Limit ring rounds to ceil(cutoff/minCellSize) when DD is available.
        {
            MetatomicTimer timer("exchangeBackwardPairs", mpiComm_);
            int maxRounds = mpiComm_.size() - 1;
            if (inputs.dd_ != nullptr && inputs.dd_->ndim > 0)
            {
                double minCellSize = 1e30;
                for (int d = 0; d < inputs.dd_->ndim; d++)
                {
                    const int    dim = inputs.dd_->dim[d];
                    const double cs  = static_cast<double>(inputs.box_[dim][dim])
                                      / inputs.dd_->numCells[dim];
                    minCellSize = std::min(minCellSize, cs);
                }
                if (minCellSize > 0.0)
                {
                    maxRounds = std::min(maxRounds,
                                         static_cast<int>(std::ceil(maxCutoff / minCellSize)));
                }
            }
            exchangeBackwardPairs(inputs.box_, maxRounds);
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

    std::vector<ActiveLinkAtom> activeLinkAtoms;
    std::vector<int32_t>        modelAtomNumbers(atomNumbers_.begin(), atomNumbers_.end());
    std::vector<int32_t>        modelChargeSourceModelIndex(numLocalMta_);
    for (int32_t i = 0; i < numLocalMta_; ++i)
    {
        modelChargeSourceModelIndex[i] = i;
    }

    bool needTypesRebuild = useNewtonNL;
    // Boundary MM atoms are represented as hydrogen link atoms in the model
    // input. The frontier stores GROMACS global atom indices, while runtime
    // model arrays are indexed by local MTA model index. The conversion goes
    // through mtaIndices_ so it works for serial execution, DD home atoms,
    // halo atoms, and backward ghosts.
    if (!data_->linkFrontier.empty())
    {
        std::unordered_map<int32_t, int32_t> globalAtomToModelIdx;
        globalAtomToModelIdx.reserve(numLocalMta_);
        for (int32_t i = 0; i < numLocalMta_; ++i)
        {
            const int32_t globalMtaIdx = mtaToGlobalMta_[i];
            if (globalMtaIdx >= 0
                && globalMtaIdx < static_cast<int32_t>(options_.params_.mtaIndices_.size()))
            {
                globalAtomToModelIdx.emplace(
                        static_cast<int32_t>(options_.params_.mtaIndices_[globalMtaIdx]), i);
            }
        }

        std::unordered_map<int32_t, int32_t> boundaryMmToPrimaryCap;
        boundaryMmToPrimaryCap.reserve(data_->linkFrontier.size());

        const PbcType pbcType = *options_.params_.pbcType_;
        matrix        boxInv;
        if (pbcType != PbcType::No)
        {
            invertBoxMatrix(inputs.box_, boxInv);
        }

        for (auto& link : data_->linkFrontier)
        {
            const auto embIt = globalAtomToModelIdx.find(link.getEmbeddedIndex());
            const auto mmIt  = globalAtomToModelIdx.find(link.getMMIndex());

            const int32_t embMtaIdx =
                    (embIt != globalAtomToModelIdx.end()) ? embIt->second : -1;
            const int32_t mmMtaIdx = (mmIt != globalAtomToModelIdx.end()) ? mmIt->second : -1;

            link.setInputIndices(embMtaIdx, mmMtaIdx);
            if (embMtaIdx >= 0 && mmMtaIdx >= 0)
            {
                auto [it, inserted] = boundaryMmToPrimaryCap.emplace(mmMtaIdx, mmMtaIdx);
                int32_t linkModelIdx = it->second;
                if (!inserted)
                {
                    linkModelIdx = static_cast<int32_t>(modelAtomNumbers.size());
                    modelAtomNumbers.push_back(link.linkAtomNumber());
                    modelChargeSourceModelIndex.push_back(mmMtaIdx);
                }
                else
                {
                    modelAtomNumbers[mmMtaIdx] = link.linkAtomNumber();
                }

                const double rawDx = static_cast<double>(positions_[mmMtaIdx][XX]
                                                         - positions_[embMtaIdx][XX]);
                const double rawDy = static_cast<double>(positions_[mmMtaIdx][YY]
                                                         - positions_[embMtaIdx][YY]);
                const double rawDz = static_cast<double>(positions_[mmMtaIdx][ZZ]
                                                         - positions_[embMtaIdx][ZZ]);
                const IVec mmCellShift =
                        computeMinimumImageCellShift(boxInv, pbcType, rawDx, rawDy, rawDz);

                activeLinkAtoms.push_back({ embMtaIdx,
                                            mmMtaIdx,
                                            linkModelIdx,
                                            mmCellShift,
                                            link.linkDistance(),
                                            link.linkAtomNumber() });
                needTypesRebuild = true;
            }
        }
    }

    const int32_t numModelAtoms = static_cast<int32_t>(modelAtomNumbers.size());
    torch::Tensor modelTypes    = data_->cachedTypes;
    if (needTypesRebuild)
    {
        modelTypes = torch::tensor(modelAtomNumbers, torch::TensorOptions().dtype(torch::kInt32))
                             .to(data_->device);
    }

    std::vector<RVec> modelPositionsForNl(positions_.begin(), positions_.end());
    std::unordered_set<int32_t> linkModelIndices;
    std::vector<int32_t> linkModelIndexList;
    linkModelIndices.reserve(activeLinkAtoms.size());
    linkModelIndexList.reserve(activeLinkAtoms.size());
    for (const auto& link : activeLinkAtoms)
    {
        const bool inserted = linkModelIndices.insert(link.linkModelIndex).second;
        if (inserted)
        {
            linkModelIndexList.push_back(link.linkModelIndex);
        }
        const RVec mmShift = computeCellShiftVector(inputs.box_, link.mmCellShift);
        const RVec linkPosition = computeLinkAtomPosition(
                positions_[link.embeddedModelIndex],
                positions_[link.mmModelIndex],
                mmShift,
                link.linkDistance);
        if (link.linkModelIndex < numLocalMta_)
        {
            modelPositionsForNl[link.linkModelIndex] = linkPosition;
        }
        else
        {
            modelPositionsForNl.push_back(linkPosition);
        }
    }

    // Model inference
    torch::Tensor forceTensor;
    torch::Tensor virialTensor;
    double        energy = 0.0;

    {
        MetatomicTimer modelTimer("model inference", mpiComm_);

        MetatomicTimer tensorPrepTimer("tensorPrep", mpiComm_);

        auto gromacs_scalar_type = torch::kFloat32;
        if (std::is_same_v<real, double>)
        {
            gromacs_scalar_type = torch::kFloat64;
        }
        auto cpu_blob_options = torch::TensorOptions().dtype(gromacs_scalar_type).device(torch::kCPU);

        auto torch_positions = torch::from_blob(positions_.data()->as_vec(), { static_cast<int64_t>(numLocalMta_), 3 }, cpu_blob_options)
                                       .to(data_->device, data_->dtype)
                                       .set_requires_grad(!data_->nonConservative);

        auto torch_cell =
                torch::from_blob(&box_, { 3, 3 }, cpu_blob_options).to(data_->device, data_->dtype);

        auto strain = torch::eye(
                3, torch::TensorOptions().dtype(data_->dtype).device(data_->device)
                           .requires_grad(!data_->nonConservative));

        auto strained_cell           = torch::matmul(torch_cell, strain);
        auto real_strained_positions = torch::matmul(torch_positions, strain);
        auto strained_positions      = real_strained_positions;

        // Link atom position replacement INSIDE the autograd graph.
        // r_link = r_emb + d_link * (r_MM - r_emb) / |r_MM - r_emb|
        // By computing this with torch operations, autograd automatically
        // computes dE/dr_emb and dE/dr_MM via the chain rule through r_link.
        // No manual spreadForce redistribution needed.
        if (!activeLinkAtoms.empty())
        {
            strained_positions = real_strained_positions.clone();
            std::vector<torch::Tensor> extraLinkPositions;

            for (const auto& link : activeLinkAtoms)
            {
                auto r_emb = real_strained_positions.index({ link.embeddedModelIndex });
                auto r_mm  = real_strained_positions.index({ link.mmModelIndex });
                auto cellShift = torch::tensor({ static_cast<double>(link.mmCellShift[XX]),
                                                 static_cast<double>(link.mmCellShift[YY]),
                                                 static_cast<double>(link.mmCellShift[ZZ]) },
                                               torch::TensorOptions()
                                                       .dtype(data_->dtype)
                                                       .device(data_->device));
                auto direction = r_mm + torch::matmul(cellShift, strained_cell) - r_emb;
                auto dist = direction.norm();
                auto r_link = r_emb + link.linkDistance * direction / dist;

                if (link.linkModelIndex < numLocalMta_)
                {
                    // Replace the boundary MM row with the primary cap.
                    strained_positions.index_put_({ link.linkModelIndex }, r_link);
                }
                else
                {
                    extraLinkPositions.push_back(r_link.unsqueeze(0));
                }
            }

            if (!extraLinkPositions.empty())
            {
                strained_positions = torch::cat(
                        { strained_positions, torch::cat(extraLinkPositions, 0) }, 0);
            }
        }

        auto system = torch::make_intrusive<metatomic_torch::SystemHolder>(
                modelTypes, strained_positions, strained_cell, data_->cachedPbc);

        if (!data_->requestedChargeInputs.empty())
        {
            if (options_.params_.mmCharges_.empty())
            {
                GMX_THROW(InconsistentInputError(
                        "Metatomic model requests charges, but topology charges are not available."));
            }

            std::vector<real> chargeValues;
            chargeValues.reserve(numModelAtoms);
            for (int32_t i = 0; i < numModelAtoms; ++i)
            {
                const int32_t sourceModelIndex = modelChargeSourceModelIndex[i];
                const int32_t mtaIndex = mtaToGlobalMta_[sourceModelIndex];
                if (mtaIndex < 0
                    || mtaIndex >= static_cast<int32_t>(options_.params_.mtaIndices_.size()))
                {
                    GMX_THROW(InconsistentInputError(
                            "Metatomic charge input contains an invalid atom index."));
                }
                const Index globalAtom = options_.params_.mtaIndices_[mtaIndex];
                if (globalAtom < 0
                    || globalAtom >= static_cast<Index>(options_.params_.mmCharges_.size()))
                {
                    GMX_THROW(InconsistentInputError(
                            "Metatomic charge input contains an atom without a stored charge."));
                }
                chargeValues.push_back(options_.params_.mmCharges_[globalAtom]);
            }

            auto charges = torch::tensor(chargeValues, cpu_blob_options)
                                   .reshape({ static_cast<int64_t>(numModelAtoms), 1 })
                                   .to(data_->device, data_->dtype);
            auto intOptions = torch::TensorOptions().dtype(torch::kInt32).device(data_->device);
            auto samplesTensor = torch::zeros({ numModelAtoms, 2 }, intOptions);
            samplesTensor.index_put_({ torch::indexing::Slice(), 1 },
                                     torch::arange(numModelAtoms, intOptions));

            auto samples = torch::make_intrusive<metatensor_torch::LabelsHolder>(
                    std::vector<std::string>{ "system", "atom" }, samplesTensor);
            auto properties = torch::make_intrusive<metatensor_torch::LabelsHolder>(
                    std::vector<std::string>{ "charge" },
                    torch::zeros({ 1, 1 }, intOptions));
            auto keys = torch::make_intrusive<metatensor_torch::LabelsHolder>(
                    std::vector<std::string>{ "_" },
                    torch::zeros({ 1, 1 }, intOptions));

            auto block = torch::make_intrusive<metatensor_torch::TensorBlockHolder>(
                    charges, samples, std::vector<metatensor_torch::Labels>{}, properties);
            auto chargeMap = torch::make_intrusive<metatensor_torch::TensorMapHolder>(
                    keys, std::vector<metatensor_torch::TensorBlock>{ block });
            chargeMap->set_info("quantity", "charge");
            chargeMap->set_info("unit", "e");

            for (const auto& inputName : data_->requestedChargeInputs)
            {
                system->add_data(inputName, chargeMap);
            }
        }

        tensorPrepTimer.stop();

        // Build NL directly into raw buffers, then wrap with from_blob.
        // Component and properties Labels are cached (identical every step).
        MetatomicTimer buildNLTimer("buildNL", mpiComm_);

        for (const auto& request : data_->nl_requests)
        {
            int64_t nPairs;

            {
                // Build NL from GROMACS pairlist (+ backward pairs in newton mode).
                // Both modes use the same code path; newton mode just has extra
                // pairs appended to pairlistMta_ from exchangeBackwardPairs().
                //
                // GROMACS pairlist uses rlist (verlet buffer) which is typically
                // larger than the model's cutoff. Filter pairs by the model's
                // cutoff to avoid sending excess pairs that waste GPU memory.
                const int64_t nHalf  = static_cast<int64_t>(pairlistMta_.size() / 2);
                const bool    full   = request->full_list();
                const double  cutoff = request->engine_cutoff("nm");
                const double  cutoff2 = cutoff * cutoff;

                const int64_t maxSyntheticPairs =
                        static_cast<int64_t>(linkModelIndexList.size()) * numModelAtoms;
                const int64_t reservePairs =
                        (full ? 2 : 1) * (nHalf + maxSyntheticPairs);
                nlSamplesBuffer_.clear();
                nlVectorsBuffer_.clear();
                nlSamplesBuffer_.reserve(reservePairs * 5);
                nlVectorsBuffer_.reserve(reservePairs * 3);

                auto appendPair = [&](int32_t ai,
                                      int32_t aj,
                                      const IVec& cellShift,
                                      double dx,
                                      double dy,
                                      double dz)
                {
                    const double dist2 = dx * dx + dy * dy + dz * dz;
                    if (dist2 > cutoff2)
                    {
                        return;
                    }

                    nlSamplesBuffer_.push_back(ai);
                    nlSamplesBuffer_.push_back(aj);
                    nlSamplesBuffer_.push_back(cellShift[XX]);
                    nlSamplesBuffer_.push_back(cellShift[YY]);
                    nlSamplesBuffer_.push_back(cellShift[ZZ]);
                    nlVectorsBuffer_.push_back(dx);
                    nlVectorsBuffer_.push_back(dy);
                    nlVectorsBuffer_.push_back(dz);

                    if (full)
                    {
                        nlSamplesBuffer_.push_back(aj);
                        nlSamplesBuffer_.push_back(ai);
                        nlSamplesBuffer_.push_back(-cellShift[XX]);
                        nlSamplesBuffer_.push_back(-cellShift[YY]);
                        nlSamplesBuffer_.push_back(-cellShift[ZZ]);
                        nlVectorsBuffer_.push_back(-dx);
                        nlVectorsBuffer_.push_back(-dy);
                        nlVectorsBuffer_.push_back(-dz);
                    }
                };

                for (int64_t k = 0; k < nHalf; k++)
                {
                    const int32_t ai = pairlistMta_[2 * k];
                    const int32_t aj = pairlistMta_[2 * k + 1];
                    if (linkModelIndices.count(ai) > 0 || linkModelIndices.count(aj) > 0)
                    {
                        continue;
                    }

                    // Compute shift vector from cell shift and current box
                    RVec shift;
                    mvmul_ur0(inputs.box_, cellShiftsMta_[k].toRVec(), shift);

                    // Displacement: r_ij = pos[j] - pos[i] + shift  (metatensor convention)
                    const double dx = static_cast<double>(positions_[aj][0] - positions_[ai][0] + shift[0]);
                    const double dy = static_cast<double>(positions_[aj][1] - positions_[ai][1] + shift[1]);
                    const double dz = static_cast<double>(positions_[aj][2] - positions_[ai][2] + shift[2]);

                    appendPair(ai, aj, cellShiftsMta_[k], dx, dy, dz);
                }

                if (!linkModelIndexList.empty())
                {
                    const PbcType pbcType = *options_.params_.pbcType_;
                    matrix boxInv;
                    if (pbcType != PbcType::No)
                    {
                        invertBoxMatrix(inputs.box_, boxInv);
                    }

                    for (const int32_t ai : linkModelIndexList)
                    {
                        for (int32_t aj = 0; aj < numModelAtoms; ++aj)
                        {
                            if (ai == aj)
                            {
                                continue;
                            }
                            if (linkModelIndices.count(aj) > 0 && ai > aj)
                            {
                                continue;
                            }

                            const double rawDx = static_cast<double>(
                                    modelPositionsForNl[aj][XX] - modelPositionsForNl[ai][XX]);
                            const double rawDy = static_cast<double>(
                                    modelPositionsForNl[aj][YY] - modelPositionsForNl[ai][YY]);
                            const double rawDz = static_cast<double>(
                                    modelPositionsForNl[aj][ZZ] - modelPositionsForNl[ai][ZZ]);

                            const IVec cellShift =
                                    computeMinimumImageCellShift(boxInv, pbcType, rawDx, rawDy, rawDz);

                            const RVec shift = computeCellShiftVector(inputs.box_, cellShift);

                            const double dx = rawDx + shift[XX];
                            const double dy = rawDy + shift[YY];
                            const double dz = rawDz + shift[ZZ];
                            appendPair(ai, aj, cellShift, dx, dy, dz);
                        }
                    }
                }

                nPairs = static_cast<int64_t>(nlSamplesBuffer_.size() / 5);
            }

            // Wrap raw buffers as tensors (zero-copy on CPU, then move to device)
            MetatomicTimer fromBlobTimer("fromBlob", mpiComm_);
            auto samples_tensor = torch::from_blob(
                    nlSamplesBuffer_.data(), { nPairs, 5 },
                    torch::TensorOptions().dtype(torch::kInt32)).to(data_->device);
            auto vectors_tensor = torch::from_blob(
                    nlVectorsBuffer_.data(), { nPairs, 3, 1 },
                    torch::TensorOptions().dtype(torch::kFloat64)).to(data_->device, data_->dtype);
            fromBlobTimer.stop();

            MetatomicTimer labelsTimer("makeSampleLabels", mpiComm_);
            metatensor_torch::Labels neighbor_samples;
            if (data_->check_consistency)
            {
                neighbor_samples = torch::make_intrusive<metatensor_torch::LabelsHolder>(
                        data_->nlSampleNames, samples_tensor);
            }
            else
            {
                neighbor_samples = torch::make_intrusive<metatensor_torch::LabelsHolder>(
                        data_->nlSampleNames, samples_tensor,
                        metatensor::assume_unique{});
            }
            labelsTimer.stop();

            MetatomicTimer blockTimer("makeTensorBlock", mpiComm_);
            auto neighbors = torch::make_intrusive<metatensor_torch::TensorBlockHolder>(
                    vectors_tensor,
                    neighbor_samples,
                    std::vector<metatensor_torch::Labels>{ data_->cachedNLComponent },
                    data_->cachedNLProperties);
            blockTimer.stop();

            MetatomicTimer autogradTimer("registerAutograd", mpiComm_);
            metatomic_torch::register_autograd_neighbors(system, neighbors, data_->check_consistency);
            autogradTimer.stop();

            MetatomicTimer addNLTimer("addNeighborList", mpiComm_);
            system->add_neighbor_list(request, neighbors);
            addNLTimer.stop();
        }

        buildNLTimer.stop();

        if (useNewtonNL)
        {
            // Restrict output to home atoms and link caps attached to home atoms.
            // Following the LAMMPS pair_metatomic pattern (selected_atoms = nlocal).
            // The model computes per-atom energies for ALL local atoms internally,
            // but only returns results for home atoms.  This is important because
            // models are free to return output samples in arbitrary order when
            // selected_atoms is nullopt, but the order is deterministic when
            // selected_atoms is set.
            std::vector<int32_t> selectedModelIndices;
            selectedModelIndices.reserve(numHomeMta_ + activeLinkAtoms.size());
            std::unordered_set<int32_t> selectedSet;
            selectedSet.reserve(numHomeMta_ + activeLinkAtoms.size());
            for (int32_t i = 0; i < numHomeMta_; ++i)
            {
                selectedModelIndices.push_back(i);
                selectedSet.insert(i);
            }
            for (const auto& link : activeLinkAtoms)
            {
                if (link.embeddedModelIndex < numHomeMta_
                    && selectedSet.insert(link.linkModelIndex).second)
                {
                    selectedModelIndices.push_back(link.linkModelIndex);
                }
            }

            auto sa_values = torch::zeros({ static_cast<int64_t>(selectedModelIndices.size()), 2 },
                                          torch::TensorOptions().dtype(torch::kInt32));
            sa_values.index_put_({ torch::indexing::Slice(), 1 },
                                 torch::tensor(selectedModelIndices,
                                               torch::TensorOptions().dtype(torch::kInt32)));
            sa_values = sa_values.to(data_->device);
            auto selected = torch::make_intrusive<metatensor_torch::LabelsHolder>(
                    std::vector<std::string>{ "system", "atom" }, sa_values);
            data_->evaluations_options->set_selected_atoms(selected);
        }

        MetatomicTimer forwardTimer("forward", mpiComm_);

        metatensor_torch::TensorMap output_map;
        c10::IValue                 ivalue_output;
        try
        {
            std::vector<metatomic_torch::System> systems;
            systems.push_back(system);

            ivalue_output = data_->model.forward(
                    { systems, data_->evaluations_options, data_->check_consistency });
            auto dict_output = ivalue_output.toGenericDict();
            output_map = dict_output.at("energy").toCustomClass<metatensor_torch::TensorMapHolder>();
        }
        catch (const std::exception& e)
        {
            GMX_THROW(APIError("[Metatomic] Model evaluation failed: " + std::string(e.what())));
        }
        // Re-extract dict for uncertainty and NC access (auto type avoids GenericDict issues)
        auto dict_output = ivalue_output.toGenericDict();

        forwardTimer.stop();

        // Check uncertainty if the model provides it
        if (data_->uncertainty_output != nullptr
            && dict_output.contains(data_->energy_uq_key))
        {
            auto uq_map = dict_output.at(data_->energy_uq_key)
                                  .toCustomClass<metatensor_torch::TensorMapHolder>();
            auto uq_block = metatensor_torch::TensorMapHolder::block_by_id(uq_map, 0);
            auto uq_values = uq_block->values().reshape({ -1 });
            auto atoms_above = uq_values > data_->uncertaintyThreshold;

            if (torch::any(atoms_above).to(torch::kCPU).item<bool>())
            {
                int64_t nAbove = torch::sum(atoms_above.to(torch::kInt64))
                                         .to(torch::kCPU).item<int64_t>();
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

        auto energy_block  = metatensor_torch::TensorMapHolder::block_by_id(output_map, 0);
        auto energy_tensor = energy_block->values();

        // Sum all returned per-atom energies.
        // In parallel, selected_atoms restricts output to home atoms only,
        // so this sums only home atom energies (each home atom has a complete NL).
        energy = energy_tensor.sum().item<double>();

        if (data_->nonConservative)
        {
            // Non-conservative: extract forces directly from model output
            MetatomicTimer ncTimer("ncExtract", mpiComm_);

            auto forces_map =
                    dict_output.at(data_->nc_forces_key)
                            .toCustomClass<metatensor_torch::TensorMapHolder>();
            auto forces_block =
                    metatensor_torch::TensorMapHolder::block_by_id(forces_map, 0);
            forceTensor = forces_block->values().squeeze(-1)
                                  .to(torch::kCPU).to(torch::kFloat64);

            // Virial from stress if available
            if (data_->nc_stress_output != nullptr)
            {
                auto stress_map =
                        dict_output.at(data_->nc_stress_key)
                                .toCustomClass<metatensor_torch::TensorMapHolder>();
                auto stress_block =
                        metatensor_torch::TensorMapHolder::block_by_id(stress_map, 0);
                auto stress_tensor = stress_block->values().squeeze(0).squeeze(-1);

                // Compute volume from box
                double volume = inputs.box_[XX][XX]
                                * (inputs.box_[YY][YY] * inputs.box_[ZZ][ZZ]
                                   - inputs.box_[YY][ZZ] * inputs.box_[ZZ][YY])
                                - inputs.box_[XX][YY]
                                          * (inputs.box_[YY][XX] * inputs.box_[ZZ][ZZ]
                                             - inputs.box_[YY][ZZ] * inputs.box_[ZZ][XX])
                                + inputs.box_[XX][ZZ]
                                          * (inputs.box_[YY][XX] * inputs.box_[ZZ][YY]
                                             - inputs.box_[YY][YY] * inputs.box_[ZZ][XX]);

                virialTensor = (-stress_tensor * volume)
                                       .to(torch::kCPU).to(torch::kFloat64);
            }
            else
            {
                virialTensor = torch::zeros({ 3, 3 }, torch::kFloat64);
            }

            ncTimer.stop();
        }
        else
        {
            // Conservative: backward pass for forces and virial via autograd
            MetatomicTimer backwardTimer("backward", mpiComm_);

            torch_positions.mutable_grad() = torch::Tensor();
            strain.mutable_grad()          = torch::Tensor();

            // Backpropagate through all returned per-atom energies.
            // In parallel, output is restricted to home atoms via selected_atoms.
            // Forces propagate to ALL local atoms (home + halo) via the NL autograd.
            energy_tensor.backward(-torch::ones_like(energy_tensor));

            backwardTimer.stop();

            MetatomicTimer toCPUTimer("toCPU", mpiComm_);

            forceTensor  = torch_positions.grad().to(torch::kCPU).to(torch::kFloat64);
            virialTensor = strain.grad().to(torch::kCPU).to(torch::kFloat64);

            toCPUTimer.stop();
        }
    }

    // Force distribution: home forces applied directly, non-home forces
    // exchanged via sparse indexed communication (or dense fallback for
    // small systems). ForceWithVirial is NOT communicated by dd_move_f.
    // In non-conservative mode, the model returns forces for home atoms
    // only (via selected_atoms), so we skip halo force exchange.
    MetatomicTimer forceScatterTimer("forceScatter", mpiComm_);

    const double* forceData = forceTensor.data_ptr<double>();
    const int32_t nForceAtoms = static_cast<int32_t>(forceTensor.size(0));

    if (data_->nonConservative)
    {
        // NC mode: forces are for home atoms only (or all atoms in serial).
        // Apply directly, no halo exchange needed.
        for (int32_t i = 0; i < nForceAtoms; i++)
        {
            int32_t gmxIdx = mtaToGmxLocal_[i];
            outputs->forceWithVirial_.force_[gmxIdx][0] += static_cast<real>(forceData[3 * i]);
            outputs->forceWithVirial_.force_[gmxIdx][1] += static_cast<real>(forceData[3 * i + 1]);
            outputs->forceWithVirial_.force_[gmxIdx][2] += static_cast<real>(forceData[3 * i + 2]);
        }
    }
    else if (mpiComm_.isParallel())
    {
        distributeNonHomeForces(forceData, outputs);
    }
    else
    {
        // Apply forces directly. When link atoms are used, autograd has
        // already computed the correct forces on r_emb and r_MM via the
        // chain rule through r_link (because the link position computation
        // is in the autograd graph). No manual spreadForce needed.
        for (int32_t i = 0; i < numLocalMta_; i++)
        {
            int32_t gmxIdx = mtaToGmxLocal_[i];
            outputs->forceWithVirial_.force_[gmxIdx][0] += static_cast<real>(forceData[3 * i]);
            outputs->forceWithVirial_.force_[gmxIdx][1] += static_cast<real>(forceData[3 * i + 1]);
            outputs->forceWithVirial_.force_[gmxIdx][2] += static_cast<real>(forceData[3 * i + 2]);
        }
    }

    forceScatterTimer.stop();

    // Restore original state.  Backward ghost atoms have no GROMACS local
    // buffer index, and backward pairs should not persist across steps.
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

    // Energy: each rank's energy is the sum of per-atom energies for all local
    // atoms (home + halo) from the pairs assigned to this rank.  GROMACS
    // global_stat sums across ranks to get the system total.
    outputs->enerd_.term[InteractionFunction::MetatomicPotentialEnergy] = static_cast<real>(energy);

    // Virial: same decomposition as energy — per-rank portion, summed by GROMACS.
    matrix virialMatrix;
    auto   virialAccessor = virialTensor.accessor<double, 2>();
    for (int32_t i = 0; i < 3; ++i)
    {
        for (int32_t j = 0; j < 3; ++j)
        {
            virialMatrix[i][j] = virialAccessor[i][j];
        }
    }
    outputs->forceWithVirial_.addVirialContribution(virialMatrix);
}

} // namespace gmx
