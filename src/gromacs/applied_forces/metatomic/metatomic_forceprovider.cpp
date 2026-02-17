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
 * Uses the GROMACS plain pairlist (excludedPairlist) as the neighbor list
 * source.  MTA-MTA nonbonded pairs are excluded from the classical force
 * calculation via intermolecularExclusionGroup, causing them to appear in
 * excludedPairlist_ instead of pairlist_.  Each pair is assigned to exactly
 * one rank by the GROMACS nbnxm pairlist builder.
 *
 * Key design points:
 *  - Neighbor list: built from excludedPairlist_, not AnalysisNeighborhood.
 *    Cell shifts are negated (GROMACS shifts first atom, metatensor shifts
 *    second atom).  Models requesting full_list get both (i,j) and (j,i).
 *  - Energy: selected_atoms = nullopt (sum all per-atom energies, home +
 *    halo).  No double counting because each pair is on one rank.
 *    NOTE: this only works for GNN-style models that exclusively use the
 *    provided neighbor list.  Models with global attention or internal pair
 *    recomputation would double-count; for those, selected_atoms should be
 *    set to home atoms only (not yet implemented).
 *  - Forces: all-reduce on a global buffer because ForceWithVirial is not
 *    communicated by dd_move_f.  Only home atom forces are applied.
 *  - Ghost deduplication: periodic ghost images share the same model index
 *    but all GROMACS local indices are mapped via gmxLocalToMtaIdx_.
 *
 * \author Metatensor developers <https://github.com/metatensor>
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "metatomic_forceprovider.h"

#include <cstdint>
#include <cstdio>

#include <algorithm>
#include <string>
#include <unordered_map>

#include "gromacs/domdec/localatomset.h"
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

    //! Whether debug logging to per-rank files is enabled (GMX_METATOMIC_DEBUG).
    bool debugEnabled = false;
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

    data_->debugEnabled = (std::getenv("GMX_METATOMIC_DEBUG") != nullptr);

    // With thread-MPI, each rank is a thread sharing the same process.
    // PyTorch's internal OpenMP would spawn N threads per rank, causing
    // massive oversubscription (e.g. 12 ranks × 12 OMP threads = 144
    // threads on 12 cores).  Force single-threaded torch operations.
    if (GMX_THREAD_MPI && mpiComm_.isParallel())
    {
        at::set_num_threads(1);
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

    if (data_->debugEnabled)
    {
        double interactionRange = data_->capabilities->engine_interaction_range("nm");
        std::string fname = "metatomic_debug_rank_" + std::to_string(mpiComm_.rank()) + ".log";
        FILE*       fp    = std::fopen(fname.c_str(), "w");
        if (fp)
        {
            std::fprintf(fp,
                         "=== Metatomic init (rank %d) ===\n"
                         "interaction_range(nm)=%.6f\n",
                         mpiComm_.rank(),
                         interactionRange);
            std::fclose(fp);
        }
    }

    auto requests_ivalue = data_->model.run_method("requested_neighbor_lists");
    for (const auto& request_ivalue : requests_ivalue.toList())
    {
        auto nl_opt = request_ivalue.get().toCustomClass<metatomic_torch::NeighborListOptionsHolder>();
        if (data_->debugEnabled)
        {
            std::string fname = "metatomic_debug_rank_" + std::to_string(mpiComm_.rank()) + ".log";
            FILE*       fp    = std::fopen(fname.c_str(), "a");
            if (fp)
            {
                std::fprintf(fp,
                             "NL request: cutoff()=%.6f, engine_cutoff(nm)=%.6f, full_list=%s\n",
                             nl_opt->cutoff(),
                             nl_opt->engine_cutoff("nm"),
                             nl_opt->full_list() ? "true" : "false");
                std::fclose(fp);
            }
        }
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
    requested_output->set_unit("kJ/mol");

    data_->evaluations_options->outputs.insert(energy_key, requested_output);
    data_->check_consistency = options_.params_.checkConsistency;

    // Allocate global force buffer sized to total MTA atoms
    const auto&   mtaIndices = options_.params_.mtaIndices_;
    const int32_t n_total    = static_cast<int32_t>(mtaIndices.size());
    globalForceBuffer_.resize(n_total, RVec({ 0.0, 0.0, 0.0 }));

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

        // Second pass: build complete gmxLocal → modelIdx mapping for ALL images
        for (int32_t i = 0; i < numLocalPlusHalo; i++)
        {
            int32_t globalIdx = globalAtomIndices[i];
            auto    it        = globalToMtaIdx.find(globalIdx);
            if (it != globalToMtaIdx.end())
            {
                gmxLocalToMtaIdx_[i] = mtaIdxToModelIdx[it->second];
            }
        }

        if (data_->debugEnabled)
        {
            std::string fname = "metatomic_debug_rank_" + std::to_string(mpiComm_.rank()) + ".log";
            FILE*       fp    = std::fopen(fname.c_str(), "a");
            if (fp)
            {
                std::fprintf(fp,
                             "gatherAtoms: home=%zu halo=%zu duplicatesSkipped=%d gmxLocalEntries=%zu\n",
                             homeGmxLocal.size(),
                             haloGmxLocal.size(),
                             numDuplicatesSkipped,
                             gmxLocalToMtaIdx_.size());
                std::fclose(fp);
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
        auto itA = gmxLocalToMtaIdx_.find(atomPair.first);
        auto itB = gmxLocalToMtaIdx_.find(atomPair.second);
        if (itA != gmxLocalToMtaIdx_.end() && itB != gmxLocalToMtaIdx_.end())
        {
            pairlistMta_.push_back(itA->second);
            pairlistMta_.push_back(itB->second);
            const IVec gmxShift = shiftIndexToXYZ(shiftIndex);
            cellShiftsMta_.push_back(IVec(-gmxShift[XX], -gmxShift[YY], -gmxShift[ZZ]));
        }
    }
}


void MetatomicForceProvider::calculateForces(const ForceProviderInput& inputs, ForceProviderOutput* outputs)
{
    MetatomicTimer totalTimer("calculateForces", mpiComm_);

    const int32_t numTotalMta = static_cast<int32_t>(options_.params_.mtaIndices_.size());

    // Fill local positions (no MPI communication)
    {
        MetatomicTimer timer("gatherAtomPositions", mpiComm_);
        gatherAtomPositions(inputs.x_);
    }
    copy_mat(inputs.box_, box_);

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
                                       .to(data_->dtype)
                                       .to(data_->device)
                                       .set_requires_grad(true);

        auto torch_cell =
                torch::from_blob(&box_, { 3, 3 }, cpu_blob_options).to(data_->dtype).to(data_->device);

        auto strain = torch::eye(
                3, torch::TensorOptions().dtype(data_->dtype).device(data_->device).requires_grad(true));

        auto strained_cell      = torch::matmul(torch_cell, strain);
        auto strained_positions = torch::matmul(torch_positions, strain);

        auto torch_pbc = preparePbcType(options_.params_.pbcType_.get(), data_->device);
        auto torch_types =
                torch::tensor(atomNumbers_, torch::TensorOptions().dtype(torch::kInt32)).to(data_->device);

        auto system = torch::make_intrusive<metatomic_torch::SystemHolder>(
                torch_types, strained_positions, strained_cell, torch_pbc);

        tensorPrepTimer.stop();

        // Build NL directly into raw buffers, then wrap with from_blob.
        // Shift vectors are computed inline (no separate prepareNL pass).
        // Component and properties Labels are cached (identical every step).
        MetatomicTimer buildNLTimer("buildNL", mpiComm_);

        for (const auto& request : data_->nl_requests)
        {
            const int64_t nHalf  = static_cast<int64_t>(pairlistMta_.size() / 2);
            const bool    full   = request->full_list();
            const int64_t nPairs = full ? 2 * nHalf : nHalf;

            nlSamplesBuffer_.resize(nPairs * 5);
            nlVectorsBuffer_.resize(nPairs * 3);

            for (int64_t k = 0; k < nHalf; k++)
            {
                const int32_t ai = pairlistMta_[2 * k];
                const int32_t aj = pairlistMta_[2 * k + 1];

                // Compute shift vector from cell shift and current box
                RVec shift;
                mvmul_ur0(inputs.box_, cellShiftsMta_[k].toRVec(), shift);

                // Displacement: r_ij = pos[j] - pos[i] + shift  (metatensor convention)
                const double dx = static_cast<double>(positions_[aj][0] - positions_[ai][0] + shift[0]);
                const double dy = static_cast<double>(positions_[aj][1] - positions_[ai][1] + shift[1]);
                const double dz = static_cast<double>(positions_[aj][2] - positions_[ai][2] + shift[2]);

                const int64_t fwd = full ? 2 * k : k;
                nlSamplesBuffer_[5 * fwd + 0] = ai;
                nlSamplesBuffer_[5 * fwd + 1] = aj;
                nlSamplesBuffer_[5 * fwd + 2] = cellShiftsMta_[k][0];
                nlSamplesBuffer_[5 * fwd + 3] = cellShiftsMta_[k][1];
                nlSamplesBuffer_[5 * fwd + 4] = cellShiftsMta_[k][2];
                nlVectorsBuffer_[3 * fwd + 0] = dx;
                nlVectorsBuffer_[3 * fwd + 1] = dy;
                nlVectorsBuffer_[3 * fwd + 2] = dz;

                if (full)
                {
                    // Reverse pair (j,i) with negated shifts and displacement
                    const int64_t rev = 2 * k + 1;
                    nlSamplesBuffer_[5 * rev + 0] = aj;
                    nlSamplesBuffer_[5 * rev + 1] = ai;
                    nlSamplesBuffer_[5 * rev + 2] = -cellShiftsMta_[k][0];
                    nlSamplesBuffer_[5 * rev + 3] = -cellShiftsMta_[k][1];
                    nlSamplesBuffer_[5 * rev + 4] = -cellShiftsMta_[k][2];
                    nlVectorsBuffer_[3 * rev + 0] = -dx;
                    nlVectorsBuffer_[3 * rev + 1] = -dy;
                    nlVectorsBuffer_[3 * rev + 2] = -dz;
                }
            }

            MetatomicTimer fromBlobTimer("fromBlob", mpiComm_);
            auto samples_tensor = torch::from_blob(
                    nlSamplesBuffer_.data(), { nPairs, 5 },
                    torch::TensorOptions().dtype(torch::kInt32)).to(data_->device);
            auto vectors_tensor = torch::from_blob(
                    nlVectorsBuffer_.data(), { nPairs, 3, 1 },
                    torch::TensorOptions().dtype(torch::kFloat64)).to(data_->dtype).to(data_->device);
            fromBlobTimer.stop();

            MetatomicTimer labelsTimer("makeSampleLabels", mpiComm_);
            auto neighbor_samples = torch::make_intrusive<metatensor_torch::LabelsHolder>(
                    data_->nlSampleNames, samples_tensor);
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

        // TODO: For non-local models (global attention, internal pair recomputation),
        // selected_atoms should be set to home atoms only to avoid double counting.
        // Currently assumes GNN-style models that only use the provided neighbor list.
        // No selected_atoms: each pair is on exactly one rank, so summing
        // all per-atom energies (home + halo) gives the correct pair energy.
        // GROMACS sums across ranks via global_stat.
        data_->evaluations_options->set_selected_atoms(torch::nullopt);

        MetatomicTimer forwardTimer("forward", mpiComm_);

        metatensor_torch::TensorMap output_map;
        try
        {
            std::vector<metatomic_torch::System> systems;
            systems.push_back(system);

            auto ivalue_output = data_->model.forward(
                    { systems, data_->evaluations_options, data_->check_consistency });
            auto dict_output = ivalue_output.toGenericDict();
            output_map = dict_output.at("energy").toCustomClass<metatensor_torch::TensorMapHolder>();
        }
        catch (const std::exception& e)
        {
            GMX_THROW(APIError("[Metatomic] Model evaluation failed: " + std::string(e.what())));
        }

        forwardTimer.stop();

        auto energy_block  = metatensor_torch::TensorMapHolder::block_by_id(output_map, 0);
        auto energy_tensor = energy_block->values();

        energy = energy_tensor.sum().item<double>();

        // Diagnostic: log pairlist size, per-rank energy and MPI sum
        if (data_->debugEnabled)
        {
            double mpiSumEnergy = energy;
            if (mpiComm_.isParallel())
            {
                mpiComm_.sumReduce(1, &mpiSumEnergy);
            }
            std::string fname =
                    "metatomic_debug_rank_" + std::to_string(mpiComm_.rank()) + ".log";
            FILE* fp = std::fopen(fname.c_str(), "a");
            if (fp)
            {
                std::fprintf(fp,
                             "pairlistPairs=%zu, numLocalMta=%d, numHomeMta=%d, "
                             "energy: perRank=%.6f, mpiSum=%.6f\n",
                             pairlistMta_.size() / 2,
                             numLocalMta_,
                             numHomeMta_,
                             energy,
                             mpiSumEnergy);
                std::fclose(fp);
            }
        }

        MetatomicTimer backwardTimer("backward", mpiComm_);

        torch_positions.mutable_grad() = torch::Tensor();
        strain.mutable_grad()          = torch::Tensor();

        energy_tensor.backward(-torch::ones_like(energy_tensor));

        backwardTimer.stop();

        MetatomicTimer toCPUTimer("toCPU", mpiComm_);

        forceTensor  = torch_positions.grad().to(torch::kCPU).to(torch::kFloat64);
        virialTensor = strain.grad().to(torch::kCPU).to(torch::kFloat64);

        toCPUTimer.stop();
    }

    // Force distribution via all-reduce.
    // backward() produces forces on ALL local atoms (home + halo).  Since
    // ForceWithVirial forces are NOT communicated by dd_move_f (which only
    // handles ForceWithShiftForces), we must all-reduce ourselves.  Each rank
    // scatters its local forces into a global buffer indexed by global MTA
    // index.  After all-reduce, each rank reads back only its home atoms.
    MetatomicTimer forceScatterTimer("forceScatter", mpiComm_);

    auto forceAccessor = forceTensor.accessor<double, 2>();

    if (mpiComm_.isParallel())
    {
        globalForceBuffer_.assign(numTotalMta, RVec({ 0.0, 0.0, 0.0 }));

        for (int32_t i = 0; i < numLocalMta_; i++)
        {
            int32_t globalMtaIdx                = mtaToGlobalMta_[i];
            globalForceBuffer_[globalMtaIdx][0] = static_cast<real>(forceAccessor[i][0]);
            globalForceBuffer_[globalMtaIdx][1] = static_cast<real>(forceAccessor[i][1]);
            globalForceBuffer_[globalMtaIdx][2] = static_cast<real>(forceAccessor[i][2]);
        }

        mpiComm_.sumReduce(3 * numTotalMta, globalForceBuffer_.data()->as_vec());

        // Apply forces only to home MTA atoms from the reduced buffer
        for (int32_t i = 0; i < numHomeMta_; i++)
        {
            int32_t gmxIdx       = mtaToGmxLocal_[i];
            int32_t globalMtaIdx = mtaToGlobalMta_[i];
            outputs->forceWithVirial_.force_[gmxIdx][0] += globalForceBuffer_[globalMtaIdx][0];
            outputs->forceWithVirial_.force_[gmxIdx][1] += globalForceBuffer_[globalMtaIdx][1];
            outputs->forceWithVirial_.force_[gmxIdx][2] += globalForceBuffer_[globalMtaIdx][2];
        }
    }
    else
    {
        // Serial: apply forces directly
        for (int32_t i = 0; i < numLocalMta_; i++)
        {
            int32_t gmxIdx = mtaToGmxLocal_[i];
            outputs->forceWithVirial_.force_[gmxIdx][0] += static_cast<real>(forceAccessor[i][0]);
            outputs->forceWithVirial_.force_[gmxIdx][1] += static_cast<real>(forceAccessor[i][1]);
            outputs->forceWithVirial_.force_[gmxIdx][2] += static_cast<real>(forceAccessor[i][2]);
        }
    }

    forceScatterTimer.stop();

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
