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
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>

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
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/gmxmpi.h"
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

// ModelOutput sample granularity: per-atom when sample_kind == "atom"
// (get_per_atom/set_per_atom are being removed from metatomic-torch).
template<typename OutputPtr>
bool modelOutputIsAtomSample(const OutputPtr& out)
{
    return out->sample_kind() == "atom";
}

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

    //! Energy output key, including the variant suffix (e.g. "energy/pbe0").
    std::string energy_key;
    //! Energy uncertainty output key (empty if disabled or model lacks it).
    std::string energy_uq_key;
    //! Requested uncertainty output (nullptr if disabled).
    metatomic_torch::ModelOutput uncertainty_output;
    //! Uncertainty threshold in kJ/mol. Atoms above this trigger a warning.
    double uncertaintyThreshold = 0.0;

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
    if (data_->device.is_cpu())
    {
#if GMX_THREAD_MPI
        // Thread-MPI: ranks share a process.  PyTorch's global thread pool
        // would be contended by all ranks calling forward() concurrently,
        // so keep at 1 to avoid oversubscription.
        if (mpiComm_.isParallel())
        {
            at::set_num_threads(1);
        }
#else
        // Real MPI (or no MPI): each rank is a separate process.
        // Use the GROMACS-assigned OpenMP thread count so that PyTorch
        // can parallelize matrix operations within each rank's allocation.
        if (mpiComm_.isParallel())
        {
            int ntomp = gmx_omp_nthreads_get(ModuleMultiThread::Default);
            at::set_num_threads(std::max(1, ntomp));
        }
        // Serial: let PyTorch use its default (all cores)
#endif
    }
    else
    {
        // GPU/other device: model runs on accelerator, CPU work is minimal.
        if (mpiComm_.isParallel())
        {
            at::set_num_threads(1);
        }
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

    auto outputs    = data_->capabilities->outputs();
    auto v_energy   = normalize_variant(options_.params_.variant);
    data_->energy_key = pick_output("energy", outputs, v_energy);
    const auto& energy_key = data_->energy_key;

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
    // Request the same sample granularity the model declares (atom vs system).
    requested_output->set_sample_kind(model_output->sample_kind());
    if (!modelOutputIsAtomSample(model_output))
    {
        GMX_LOG(logger_.warning)
                .asParagraph()
                .appendText(
                        "Metatomic model does not support per-atom energy output "
                        "(sample_kind != \"atom\"). Energy decomposition in domain "
                        "decomposition may be less accurate.");
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

            if (modelOutputIsAtomSample(uq_cap))
            {
                data_->uncertainty_output =
                        torch::make_intrusive<metatomic_torch::ModelOutputHolder>();
                data_->uncertainty_output->set_quantity("energy");
                data_->uncertainty_output->set_unit("kJ/mol");
                data_->uncertainty_output->set_sample_kind("atom");

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
        if (!modelOutputIsAtomSample(nc_forces_cap))
        {
            GMX_THROW(APIError(formatString(
                    "The model's '%s' output can not produce per-atom output "
                    "(sample_kind != \"atom\"), we can not enable non-conservative "
                    "simulations",
                    data_->nc_forces_key.c_str())));
        }

        data_->nc_forces_output = torch::make_intrusive<metatomic_torch::ModelOutputHolder>();
        data_->nc_forces_output->set_quantity("force");
        data_->nc_forces_output->set_unit("kJ/mol/nm");
        data_->nc_forces_output->set_sample_kind("atom");

        data_->evaluations_options->outputs.insert(
                data_->nc_forces_key, data_->nc_forces_output);

        data_->nc_stress_key = pick_output("non_conservative_stress", outputs, v_nc_stress);
        if (outputs.contains(data_->nc_stress_key))
        {
            data_->nc_stress_output = torch::make_intrusive<metatomic_torch::ModelOutputHolder>();
            data_->nc_stress_output->set_quantity("stress");
            data_->nc_stress_output->set_unit("kJ/mol/nm^3");
            // System-level stress samples (not per-atom).
            data_->nc_stress_output->set_sample_kind("system");

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
}


#if GMX_MPI
namespace
{

/*! \brief Returns, on every rank, the concatenation of one buffer per rank.
 *
 * The buffers are concatenated in rank order, and \p ownOffset, when given,
 * receives the index at which the buffer of this rank starts. This is MPI_Allgatherv, written
 * as a gather to the main rank and a broadcast because thread-MPI does not
 * provide MPI_Allgatherv.
 */
template<typename T>
std::vector<T> gatherFromAllRanks(const MpiComm& mpiComm,
                                  std::vector<T>* send,
                                  MPI_Datatype    type,
                                  std::size_t*    ownOffset = nullptr)
{
    const int numRanks = mpiComm.size();
    GMX_RELEASE_ASSERT(send->size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                       "Buffer too large for an MPI count");
    std::vector<int> counts(numRanks, 0);
    counts[mpiComm.rank()] = static_cast<int>(send->size());
    mpiComm.sumReduce(ArrayRef<int>(counts));

    std::vector<int> displacements(numRanks, 0);
    std::int64_t     total = 0;
    for (int rank = 0; rank < numRanks; rank++)
    {
        displacements[rank] = static_cast<int>(total);
        total += counts[rank];
    }
    GMX_RELEASE_ASSERT(total <= std::numeric_limits<int>::max(),
                       "Gathered buffer too large for an MPI count");
    if (ownOffset != nullptr)
    {
        *ownOffset = static_cast<std::size_t>(displacements[mpiComm.rank()]);
    }

    std::vector<T> received(total);
    if (total == 0)
    {
        // thread-MPI cannot gather a buffer of zero total extent
        return received;
    }
    MPI_Gatherv(send->data(),
                counts[mpiComm.rank()],
                type,
                received.data(),
                counts.data(),
                displacements.data(),
                type,
                mpiComm.mainRank(),
                mpiComm.comm());
    MPI_Bcast(received.data(), static_cast<int>(total), type, mpiComm.mainRank(), mpiComm.comm());
    return received;
}

} // namespace
#endif

void MetatomicForceProvider::gatherRemoteAtoms()
{
#if GMX_MPI
    if (!mpiComm_.isParallel())
    {
        return;
    }

    // Every rank contributes the atoms it owns, so each atom is sent once.
    std::vector<int>  sendGlobalMta(numHomeMta_);
    std::vector<int>  sendAtomNumbers(numHomeMta_);
    std::vector<real> sendPositions(3 * static_cast<std::size_t>(numHomeMta_));
    for (int32_t i = 0; i < numHomeMta_; i++)
    {
        sendGlobalMta[i]         = mtaToGlobalMta_[i];
        sendAtomNumbers[i]       = atomNumbers_[i];
        sendPositions[3 * i]     = positions_[i][XX];
        sendPositions[3 * i + 1] = positions_[i][YY];
        sendPositions[3 * i + 2] = positions_[i][ZZ];
    }

    const std::vector<int> allGlobalMta   = gatherFromAllRanks(mpiComm_, &sendGlobalMta, MPI_INT);
    const std::vector<int> allAtomNumbers = gatherFromAllRanks(mpiComm_, &sendAtomNumbers, MPI_INT);
    const std::vector<real> allPositions = gatherFromAllRanks(mpiComm_, &sendPositions, GMX_MPI_REAL);

    // Keep the image GROMACS gave for the atoms already here. The atoms added
    // below can be in any periodic image: gatherRemotePairs() finds the
    // shift of each of their pairs from the pair vector the owner sends.
    std::unordered_set<int32_t> known(mtaToGlobalMta_.begin(), mtaToGlobalMta_.begin() + numLocalMta_);
    for (std::size_t k = 0; k < allGlobalMta.size(); k++)
    {
        if (known.insert(allGlobalMta[k]).second)
        {
            positions_.push_back({ allPositions[3 * k], allPositions[3 * k + 1], allPositions[3 * k + 2] });
            atomNumbers_.push_back(allAtomNumbers[k]);
            mtaToGlobalMta_.push_back(allGlobalMta[k]);
            numLocalMta_++;
        }
    }
#endif
}


void MetatomicForceProvider::gatherRemotePairs(const matrix box)
{
    backwardPairsMta_.clear();
    backwardShiftsMta_.clear();

#if GMX_MPI
    if (!mpiComm_.isParallel())
    {
        return;
    }

    // The GROMACS pairlist holds each pair, with its periodic image, on
    // exactly one rank, so the lists of all ranks together are the pairs of
    // the whole system. A rank sends the two atoms and the pair vector it
    // gets from its own positions and the shift of the pairlist. The receiver
    // then finds the shift that gives the same vector between its copies of
    // the two atoms, which can be other periodic images. Nothing is decided
    // again by a minimum image search.
    const std::size_t numMyPairs = pairlistMta_.size() / 2;
    std::vector<int>  sendPairs(2 * numMyPairs);
    std::vector<real> sendVectors(3 * numMyPairs);
    for (std::size_t k = 0; k < numMyPairs; k++)
    {
        const int32_t ai     = pairlistMta_[2 * k];
        const int32_t aj     = pairlistMta_[2 * k + 1];
        const IVec&   n      = cellShiftsMta_[k];
        sendPairs[2 * k]     = mtaToGlobalMta_[ai];
        sendPairs[2 * k + 1] = mtaToGlobalMta_[aj];
        // r_ij = x[j] + n * box - x[i], the convention of cellShiftsMta_
        sendVectors[3 * k] = positions_[aj][XX] - positions_[ai][XX] + n[XX] * box[XX][XX]
                             + n[YY] * box[YY][XX] + n[ZZ] * box[ZZ][XX];
        sendVectors[3 * k + 1] = positions_[aj][YY] - positions_[ai][YY] + n[YY] * box[YY][YY]
                                 + n[ZZ] * box[ZZ][YY];
        sendVectors[3 * k + 2] = positions_[aj][ZZ] - positions_[ai][ZZ] + n[ZZ] * box[ZZ][ZZ];
    }

    std::size_t             ownOffset = 0;
    const std::vector<int>  allPairs = gatherFromAllRanks(mpiComm_, &sendPairs, MPI_INT, &ownOffset);
    const std::vector<real> allVectors = gatherFromAllRanks(mpiComm_, &sendVectors, GMX_MPI_REAL);
    const std::size_t       ownFirst = ownOffset / 2;

    std::unordered_map<int32_t, int32_t> globalToLocal;
    globalToLocal.reserve(numLocalMta_);
    for (int32_t i = 0; i < numLocalMta_; i++)
    {
        globalToLocal.emplace(mtaToGlobalMta_[i], i);
    }

    matrix boxInv;
    invertBoxMatrix(box, boxInv);

    for (std::size_t k = 0; k < allPairs.size() / 2; k++)
    {
        if (k >= ownFirst && k < ownFirst + numMyPairs)
        {
            // The pairs of this rank are in pairlistMta_ already
            continue;
        }
        const auto itI = globalToLocal.find(allPairs[2 * k]);
        const auto itJ = globalToLocal.find(allPairs[2 * k + 1]);
        GMX_RELEASE_ASSERT(itI != globalToLocal.end() && itJ != globalToLocal.end(),
                           "gatherRemoteAtoms() must have added every atom of every pair");
        const int32_t localI = itI->second;
        const int32_t localJ = itJ->second;

        // What is left of the pair vector after the vector between the local
        // copies is a lattice vector. invertBoxMatrix returns a
        // lower-triangular inverse.
        const double dx = static_cast<double>(allVectors[3 * k])
                          - static_cast<double>(positions_[localJ][XX] - positions_[localI][XX]);
        const double dy = static_cast<double>(allVectors[3 * k + 1])
                          - static_cast<double>(positions_[localJ][YY] - positions_[localI][YY]);
        const double dz = static_cast<double>(allVectors[3 * k + 2])
                          - static_cast<double>(positions_[localJ][ZZ] - positions_[localI][ZZ]);

        IVec shift;
        shift[XX] = static_cast<int>(
                std::round(boxInv[XX][XX] * dx + boxInv[YY][XX] * dy + boxInv[ZZ][XX] * dz));
        shift[YY] = static_cast<int>(std::round(boxInv[YY][YY] * dy + boxInv[ZZ][YY] * dz));
        shift[ZZ] = static_cast<int>(std::round(boxInv[ZZ][ZZ] * dz));

        backwardPairsMta_.push_back(localI);
        backwardPairsMta_.push_back(localJ);
        backwardShiftsMta_.push_back(shift);
    }
#else
    GMX_UNUSED_VALUE(box);
#endif
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

    // Step 3: Gather the tuples of all ranks on every rank.
#if GMX_MPI
    const std::vector<double> recvBuf = gatherFromAllRanks(mpiComm_, &sendBuf, MPI_DOUBLE);
#else
    const std::vector<double> recvBuf = sendBuf;
#endif
    const int totalNonHome = static_cast<int>(recvBuf.size() / 4);

    // Step 4: Scan received tuples for forces destined for our home atoms.
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

    // In parallel every rank evaluates the model on the whole system and
    // keeps the energy of its home atoms. A model can depend on atoms several
    // cutoffs away, so a rank needs all atoms and all pairs, which no halo of
    // fixed width provides. The pairs come from the GROMACS pairlists of all
    // ranks.
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
        // Largest cutoff over the requested neighbor lists
        double maxCutoff = 0.0;
        for (const auto& req : data_->nl_requests)
        {
            maxCutoff = std::max(maxCutoff, req->engine_cutoff("nm"));
        }

        // Step 1: Add the atoms the other ranks own. Extends positions_,
        // atomNumbers_, mtaToGlobalMta_ and numLocalMta_.
        {
            MetatomicTimer timer("gatherRemoteAtoms", mpiComm_);
            gatherRemoteAtoms();
        }

        // Rebuild cachedTypes now that atomNumbers_ has grown
        data_->cachedTypes =
                torch::tensor(atomNumbers_, torch::TensorOptions().dtype(torch::kInt32)).to(data_->device);

        // Step 2: Add the pairs the other ranks hold.
        {
            MetatomicTimer timer("gatherRemotePairs", mpiComm_);
            gatherRemotePairs(inputs.box_);
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
    torch::Tensor forceTensor;
    torch::Tensor virialTensor;
    double        energy = 0.0;

    // A rank without home atoms of the model has no energy to compute, and
    // models are not required to accept an empty selection. It still takes
    // part in the gathers above and in the force distribution below, which
    // are collective.
    if (!useNewtonNL || numHomeMta_ > 0)
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

        auto strained_cell      = torch::matmul(torch_cell, strain);
        auto strained_positions = torch::matmul(torch_positions, strain);

        auto system = torch::make_intrusive<metatomic_torch::SystemHolder>(
                data_->cachedTypes, strained_positions, strained_cell, data_->cachedPbc);

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
                // pairs appended to pairlistMta_ from gatherRemotePairs().
                //
                // GROMACS pairlist uses rlist (verlet buffer) which is typically
                // larger than the model's cutoff. Filter pairs by the model's
                // cutoff to avoid sending excess pairs that waste GPU memory.
                const int64_t nHalf  = static_cast<int64_t>(pairlistMta_.size() / 2);
                const bool    full   = request->full_list();
                const double  cutoff = request->engine_cutoff("nm");
                const double  cutoff2 = cutoff * cutoff;

                // Pre-allocate for worst case (all pairs within cutoff)
                const int64_t maxPairs = full ? 2 * nHalf : nHalf;
                nlSamplesBuffer_.resize(maxPairs * 5);
                nlVectorsBuffer_.resize(maxPairs * 3);

                int64_t outIdx = 0; // running output index for filtered pairs

                for (int64_t k = 0; k < nHalf; k++)
                {
                    const int32_t ai = pairlistMta_[2 * k];
                    const int32_t aj = pairlistMta_[2 * k + 1];

                    // Compute shift vector from cell shift and current box
                    RVec shift;
                    const IVec& n = cellShiftsMta_[k];
                    shift[XX] = n[XX] * inputs.box_[XX][XX] + n[YY] * inputs.box_[YY][XX]
                                + n[ZZ] * inputs.box_[ZZ][XX];
                    shift[YY] = n[YY] * inputs.box_[YY][YY] + n[ZZ] * inputs.box_[ZZ][YY];
                    shift[ZZ] = n[ZZ] * inputs.box_[ZZ][ZZ];

                    // Displacement: r_ij = pos[j] - pos[i] + shift  (metatensor convention)
                    const double dx = static_cast<double>(positions_[aj][0] - positions_[ai][0] + shift[0]);
                    const double dy = static_cast<double>(positions_[aj][1] - positions_[ai][1] + shift[1]);
                    const double dz = static_cast<double>(positions_[aj][2] - positions_[ai][2] + shift[2]);

                    const double dist2 = dx * dx + dy * dy + dz * dz;
                    if (dist2 > cutoff2)
                    {
                        continue;
                    }

                    nlSamplesBuffer_[5 * outIdx + 0] = ai;
                    nlSamplesBuffer_[5 * outIdx + 1] = aj;
                    nlSamplesBuffer_[5 * outIdx + 2] = cellShiftsMta_[k][0];
                    nlSamplesBuffer_[5 * outIdx + 3] = cellShiftsMta_[k][1];
                    nlSamplesBuffer_[5 * outIdx + 4] = cellShiftsMta_[k][2];
                    nlVectorsBuffer_[3 * outIdx + 0] = dx;
                    nlVectorsBuffer_[3 * outIdx + 1] = dy;
                    nlVectorsBuffer_[3 * outIdx + 2] = dz;
                    outIdx++;

                    if (full)
                    {
                        // Reverse pair (j,i) with negated shifts and displacement
                        nlSamplesBuffer_[5 * outIdx + 0] = aj;
                        nlSamplesBuffer_[5 * outIdx + 1] = ai;
                        nlSamplesBuffer_[5 * outIdx + 2] = -cellShiftsMta_[k][0];
                        nlSamplesBuffer_[5 * outIdx + 3] = -cellShiftsMta_[k][1];
                        nlSamplesBuffer_[5 * outIdx + 4] = -cellShiftsMta_[k][2];
                        nlVectorsBuffer_[3 * outIdx + 0] = -dx;
                        nlVectorsBuffer_[3 * outIdx + 1] = -dy;
                        nlVectorsBuffer_[3 * outIdx + 2] = -dz;
                        outIdx++;
                    }
                }

                nPairs = outIdx;
                nlSamplesBuffer_.resize(nPairs * 5);
                nlVectorsBuffer_.resize(nPairs * 3);
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

        // Without a selection a model may return its per-atom rows in any
        // order, and the non-conservative forces are read by row.
        if (useNewtonNL || data_->nonConservative)
        {
            // Restrict output to home atoms only [0, numHomeMta_).
            // Following the LAMMPS pair_metatomic pattern (selected_atoms = nlocal).
            // The model computes per-atom energies for ALL local atoms internally,
            // but only returns results for home atoms.  This is important because
            // models are free to return output samples in arbitrary order when
            // selected_atoms is nullopt, but the order is deterministic when
            // selected_atoms is set.
            auto sa_values = torch::zeros({ numHomeMta_, 2 },
                                          torch::TensorOptions().dtype(torch::kInt32));
            sa_values.index_put_({ torch::indexing::Slice(), 1 },
                                 torch::arange(numHomeMta_, torch::kInt32));
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
            output_map = dict_output.at(data_->energy_key).toCustomClass<metatensor_torch::TensorMapHolder>();
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

                // The stress is that of the whole input group and every rank
                // gets the same tensor, while the virials of the ranks are
                // summed. Each rank adds the share of its home atoms.
                const double homeShare =
                        static_cast<double>(numHomeMta_)
                        / static_cast<double>(options_.params_.mtaIndices_.size());

                // 1/2 for GROMACS' virial convention
                virialTensor = (0.5 * homeShare * stress_tensor * volume).to(torch::kCPU).to(torch::kFloat64);
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

            forceTensor = torch_positions.grad().to(torch::kCPU).to(torch::kFloat64);
            // 1/2 for GROMACS' virial convention; "-" since backward ran on -E
            virialTensor = (-0.5 * strain.grad()).to(torch::kCPU).to(torch::kFloat64);

            toCPUTimer.stop();
        }
    }
    else
    {
        forceTensor  = torch::zeros({ numLocalMta_, 3 }, torch::kFloat64);
        virialTensor = torch::zeros({ 3, 3 }, torch::kFloat64);
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
        // Serial: apply forces directly
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
