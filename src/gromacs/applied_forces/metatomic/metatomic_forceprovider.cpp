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
 * Two NL modes are supported (controlled by MDP `metatomic-nl-mode`):
 *
 *  - **full** (default): uses the GROMACS pairlist as the pair source, then
 *    exchanges pair identities across ranks (backward pair exchange) so every
 *    home atom has ALL its pairs.  Sums home-atom energies only.  Safe for all
 *    model architectures (newton pair ON pattern).
 *
 *  - **pairlist**: uses the GROMACS excluded pairlist (excludedPairlist_).
 *    MTA-MTA nonbonded pairs are excluded from the classical force calculation
 *    via intermolecularExclusionGroup.  Sets `selected_atoms = nullopt` — each
 *    pair is on exactly one rank, so summing all per-atom energies is correct.
 *    Only valid for models that exclusively use the provided neighbor list.
 *
 * Common design points:
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

#include <cmath>
#include <cstdint>
#include <cstdio>

#include <algorithm>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "gromacs/domdec/domdec_network.h"
#include "gromacs/domdec/domdec_struct.h"
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

    //! Cached types and PBC tensors (re-created on AtomsRedistributed).
    torch::Tensor cachedTypes;
    torch::Tensor cachedPbc;

    //! Effective NL mode: "full" (pairlist + backward pair exchange) or
    //! "pairlist" (excluded pairlist only, selected_atoms=nullopt).
    std::string nlMode = "full";
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

    // NL mode: MDP setting, overridable by env var
    data_->nlMode = options_.params_.nlMode;
    if (const char* env = std::getenv("GMX_METATOMIC_NL_MODE"))
    {
        data_->nlMode = std::string(env);
    }
    if (data_->nlMode != "full" && data_->nlMode != "pairlist")
    {
        GMX_THROW(InvalidInputError(
                formatString("Invalid metatomic nl-mode '%s'. Must be 'full' or 'pairlist'.",
                             data_->nlMode.c_str())));
    }
    GMX_LOG(logger_.info)
            .asParagraph()
            .appendTextFormatted("Metatomic NL mode: %s", data_->nlMode.c_str());

    // Force single-threaded PyTorch operations in all parallel runs.
    // In thread-MPI, ranks share a process; in real MPI, each rank is a process.
    // In both cases, GROMACS manages CPU affinity, and having PyTorch spawn its
    // own internal thread pool (defaulting to all cores) leads to catastrophic
    // oversubscription and context switching overhead.
    if (mpiComm_.isParallel())
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

    GMX_RELEASE_ASSERT(std::count(atomNumbers_.begin(), atomNumbers_.end(), 0) == 0,
                       "Some atom numbers not set.");

    // Update cached tensors for the new atom distribution
    data_->cachedTypes =
            torch::tensor(atomNumbers_, torch::TensorOptions().dtype(torch::kInt32)).to(data_->device);
    data_->cachedPbc = preparePbcType(options_.params_.pbcType_.get(), data_->device);

    // Debug: dump local-to-global mapping
    if (data_->debugEnabled)
    {
        std::string fname =
                "metatomic_atoms_rank_" + std::to_string(mpiComm_.rank()) + ".csv";
        FILE* fp = std::fopen(fname.c_str(), "w");
        if (fp)
        {
            std::fprintf(fp, "localIdx,globalMta,isHome\n");
            for (int32_t i = 0; i < numLocalMta_; i++)
            {
                std::fprintf(fp, "%d,%d,%d\n", i, mtaToGlobalMta_[i], i < numHomeMta_ ? 1 : 0);
            }
            std::fclose(fp);
        }
    }
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

    if (data_->debugEnabled && totalAdded > 0)
    {
        std::string fname =
                "metatomic_debug_rank_" + std::to_string(mpiComm_.rank()) + ".log";
        FILE* fp = std::fopen(fname.c_str(), "a");
        if (fp)
        {
            std::fprintf(fp,
                         "exchangeBackwardGhosts: added %d, numLocalMta=%d (home=%d), "
                         "uniqueGlobalMta=%zu\n",
                         totalAdded, numLocalMta_, numHomeMta_,
                         existingGlobalMta.size());
            std::fclose(fp);
        }
    }

    return totalAdded;
}


void MetatomicForceProvider::exchangeBackwardPairs(const matrix box)
{
    backwardPairsMta_.clear();
    backwardShiftsMta_.clear();

    if (!mpiComm_.isParallel())
    {
        return;
    }

    const int32_t numTotalMta = static_cast<int32_t>(options_.params_.mtaIndices_.size());

    // Step 1: Build global pair existence table via allreduce.
    // pairTable[gI * numTotalMta + gJ] = 1 if any rank has pair (gI, gJ).
    // After sumReduce, entries > 0 indicate pairs that exist somewhere.
    std::vector<int> pairTable(numTotalMta * numTotalMta, 0);
    const int nMyPairs = static_cast<int>(pairlistMta_.size() / 2);
    for (int k = 0; k < nMyPairs; k++)
    {
        const int32_t gI = mtaToGlobalMta_[pairlistMta_[2 * k]];
        const int32_t gJ = mtaToGlobalMta_[pairlistMta_[2 * k + 1]];
        pairTable[gI * numTotalMta + gJ] = 1;
    }
    mpiComm_.sumReduce(static_cast<std::size_t>(numTotalMta * numTotalMta),
                       pairTable.data());

    // Step 2: Build set of my home atoms and existing canonical pairs.
    std::unordered_set<int32_t> myHomeGlobalMta;
    for (int32_t i = 0; i < numHomeMta_; i++)
    {
        myHomeGlobalMta.insert(mtaToGlobalMta_[i]);
    }

    // Canonical pair set {min(gI,gJ), max(gI,gJ)} to avoid half-list duplication.
    std::set<std::pair<int32_t, int32_t>> existingCanonical;
    for (int k = 0; k < nMyPairs; k++)
    {
        const int32_t gI = mtaToGlobalMta_[pairlistMta_[2 * k]];
        const int32_t gJ = mtaToGlobalMta_[pairlistMta_[2 * k + 1]];
        existingCanonical.insert({ std::min(gI, gJ), std::max(gI, gJ) });
    }

    // Step 3: Build global MTA → local index mapping.
    std::unordered_map<int32_t, int32_t> globalToLocal;
    for (int32_t i = 0; i < numLocalMta_; i++)
    {
        globalToLocal[mtaToGlobalMta_[i]] = i;
    }

    // Step 4: Find pairs I need but don't have.
    for (int32_t gI = 0; gI < numTotalMta; gI++)
    {
        for (int32_t gJ = 0; gJ < numTotalMta; gJ++)
        {
            if (pairTable[gI * numTotalMta + gJ] == 0)
            {
                continue;
            }

            // Pair must involve one of my home atoms.
            const bool iIsMyHome = myHomeGlobalMta.count(gI) > 0;
            const bool jIsMyHome = myHomeGlobalMta.count(gJ) > 0;
            if (!iIsMyHome && !jIsMyHome)
            {
                continue;
            }

            // Skip if I already have this pair (in either direction).
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

            // Compute minimum-image shift from local positions (orthorhombic).
            const double rawDx =
                    static_cast<double>(positions_[localJ][XX] - positions_[localI][XX]);
            const double rawDy =
                    static_cast<double>(positions_[localJ][YY] - positions_[localI][YY]);
            const double rawDz =
                    static_cast<double>(positions_[localJ][ZZ] - positions_[localI][ZZ]);

            IVec shift;
            shift[XX] = static_cast<int>(
                    std::round(-rawDx / static_cast<double>(box[XX][XX])));
            shift[YY] = static_cast<int>(
                    std::round(-rawDy / static_cast<double>(box[YY][YY])));
            shift[ZZ] = static_cast<int>(
                    std::round(-rawDz / static_cast<double>(box[ZZ][ZZ])));

            backwardPairsMta_.push_back(localI);
            backwardPairsMta_.push_back(localJ);
            backwardShiftsMta_.push_back(shift);
            existingCanonical.insert(canonical);
        }
    }

    if (data_->debugEnabled)
    {
        std::string fname =
                "metatomic_debug_rank_" + std::to_string(mpiComm_.rank()) + ".log";
        FILE* fp = std::fopen(fname.c_str(), "a");
        if (fp)
        {
            std::fprintf(fp,
                         "exchangeBackwardPairs: added %zu pairs "
                         "(pairlist=%d, total=%zu)\n",
                         backwardPairsMta_.size() / 2,
                         nMyPairs,
                         pairlistMta_.size() / 2 + backwardPairsMta_.size() / 2);
            std::fclose(fp);
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

    // Newton NL mode: in parallel with nl-mode=full, each rank needs ALL
    // pairs involving its home atoms (not just the ones assigned by the
    // eighth-shell DD decomposition). Uses the GROMACS pairlist as the pair
    // source, then exchanges pair identities across ranks.
    const bool useNewtonNL = mpiComm_.isParallel() && data_->nlMode == "full";

    // Save original numLocalMta_ before potential backward ghost extension.
    // Must be restored after model evaluation so that gatherAtomPositions
    // on the next step does not access out-of-bounds mtaToGmxLocal_ entries.
    const int32_t origNumLocalMta = numLocalMta_;

    // Save original pairlist size; backward pairs are appended temporarily.
    const std::size_t origPairlistSize = pairlistMta_.size();
    const std::size_t origShiftsSize   = cellShiftsMta_.size();

    if (useNewtonNL)
    {
        // Step 1: Exchange backward ghost atoms to fill the backward gap
        // in the DD halo.  Extends positions_, atomNumbers_, mtaToGlobalMta_
        // and numLocalMta_ with atoms from the backward PBC neighbor.
        {
            MetatomicTimer timer("exchangeBackwardGhosts", mpiComm_);
            double maxCutoff = 0.0;
            for (const auto& req : data_->nl_requests)
            {
                maxCutoff = std::max(maxCutoff, req->engine_cutoff("nm"));
            }
            exchangeBackwardGhosts(inputs.dd_, inputs.box_, maxCutoff);
        }

        // Rebuild cachedTypes after backward ghost exchange extended atomNumbers_
        data_->cachedTypes =
                torch::tensor(atomNumbers_, torch::TensorOptions().dtype(torch::kInt32)).to(data_->device);

        // Step 2: Exchange backward-direction pairs.  Discovers pairs from
        // other ranks' pairlists that involve this rank's home atoms.
        {
            MetatomicTimer timer("exchangeBackwardPairs", mpiComm_);
            exchangeBackwardPairs(inputs.box_);
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
                                       .set_requires_grad(true);

        auto torch_cell =
                torch::from_blob(&box_, { 3, 3 }, cpu_blob_options).to(data_->device, data_->dtype);

        auto strain = torch::eye(
                3, torch::TensorOptions().dtype(data_->dtype).device(data_->device).requires_grad(true));

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
                // pairs appended to pairlistMta_ from exchangeBackwardPairs().
                const int64_t nHalf = static_cast<int64_t>(pairlistMta_.size() / 2);
                const bool    full  = request->full_list();
                nPairs              = full ? 2 * nHalf : nHalf;

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
            }

            // Debug: dump global pair indices + distances for comparison
            if (data_->debugEnabled)
            {
                std::string fname =
                        "metatomic_pairs_rank_" + std::to_string(mpiComm_.rank()) + ".csv";
                FILE* fp2 = std::fopen(fname.c_str(), "w");
                if (fp2)
                {
                    std::fprintf(fp2, "globalI,globalJ,shift_a,shift_b,shift_c,dist\n");
                    for (int64_t p = 0; p < nPairs; p++)
                    {
                        const int32_t lI = nlSamplesBuffer_[5 * p + 0];
                        const int32_t lJ = nlSamplesBuffer_[5 * p + 1];
                        const double ddx = nlVectorsBuffer_[3 * p + 0];
                        const double ddy = nlVectorsBuffer_[3 * p + 1];
                        const double ddz = nlVectorsBuffer_[3 * p + 2];
                        const double dist = std::sqrt(ddx * ddx + ddy * ddy + ddz * ddz);
                        std::fprintf(fp2, "%d,%d,%d,%d,%d,%.8f\n",
                                     mtaToGlobalMta_[lI],
                                     mtaToGlobalMta_[lJ],
                                     nlSamplesBuffer_[5 * p + 2],
                                     nlSamplesBuffer_[5 * p + 3],
                                     nlSamplesBuffer_[5 * p + 4],
                                     dist);
                    }
                    std::fclose(fp2);
                }
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

        if (useNewtonNL)
        {
            // Newton mode: restrict output to home atoms only [0, numHomeMta_).
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
        else
        {
            // Pairlist mode: each pair is on exactly one rank, so summing all
            // per-atom energies (home + halo) gives the correct pair energy.
            data_->evaluations_options->set_selected_atoms(torch::nullopt);
        }

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

        // Sum all returned per-atom energies.
        // In Newton mode, selected_atoms restricts output to home atoms only,
        // so this sums only home atom energies.
        // In pairlist mode, selected_atoms is nullopt, so this sums all atoms.
        // Both are correct: Newton mode has complete NL per home atom,
        // pairlist mode has each pair on exactly one rank.
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
                             "nlMode=%s, nlPairs=%zu, numLocalMta=%d, numHomeMta=%d, "
                             "energy: perRank=%.6f, mpiSum=%.6f\n",
                             data_->nlMode.c_str(),
                             nlSamplesBuffer_.size() / 5,
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

        // Backpropagate through all returned per-atom energies.
        // In Newton mode, output is restricted to home atoms via selected_atoms.
        // In pairlist mode, output includes all local atoms.
        // Forces propagate to ALL local atoms (home + halo) via the NL autograd.
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
        globalForceBuffer_.resize(numTotalMta);
        std::fill(globalForceBuffer_.begin(), globalForceBuffer_.end(), RVec({ 0.0, 0.0, 0.0 }));

        // Scatter local forces into the global buffer.
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
