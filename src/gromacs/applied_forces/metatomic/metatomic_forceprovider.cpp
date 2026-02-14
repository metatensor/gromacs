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
 * Implements the Metatomic Force Provider class with proper domain decomposition support.
 *
 * \author Metatensor developers <https://github.com/metatensor>
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "metatomic_forceprovider.h"

#include <cmath>
#include <cstdint>

#include <algorithm>
#include <optional>
#include <set>
#include <tuple>

#include "gromacs/domdec/localatomset.h"
#include "gromacs/mdlib/broadcaststructs.h"
#include "gromacs/mdrunutility/mdmodulesnotifiers.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/selection/nbsearch.h"
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

/*! \brief Normalizes the variant string for Metatomic output selection.
 *
 * \param[in] variant_string The raw variant string from options.
 * \return A torch::optional containing the string if valid, or nullopt if empty/"no".
 */
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

/*! \brief Converts GROMACS PbcType to a boolean tensor for Metatomic.
 *
 * \param[in] pbcType The GROMACS periodic boundary condition type.
 * \param[in] device  The torch device where the tensor should reside.
 * \return A boolean tensor of shape {3} indicating periodicity in X, Y, Z.
 */
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
    else if (*pbcType != PbcType::Xyz)
    {
        GMX_THROW(InconsistentInputError("PBC type not supported by Metatomic interface."));
    }
    return torch::tensor({ true, true, true }, options);
}

/*! \brief Constructs a Metatensor TensorBlock representing the neighbor list.
 *
 * This function takes the filtered pairlist (atoms participating in the model interaction)
 * and constructs the corresponding neighbor list in the format required by Metatensor/Torch.
 * It computes the interatomic vectors, applying periodic boundary shifts where necessary.
 *
 * \param[in] pairlist     Flat array of atom pairs (indices into the model's atom list).
 * \param[in] shiftVectors Geometric shift vectors (RVec) for each pair.
 * \param[in] cellShifts   Integer cell shift indices for each pair (for metadata).
 * \param[in] positions    Positions of the atoms (ordered by model index).
 * \param[in] device       The torch device for the output tensors.
 * \param[in] dtype        The torch scalar type (float32/float64).
 * \return A TensorBlockHolder containing the neighbor list data.
 */
static metatensor_torch::TensorBlock buildNeighborListFromPairlist(ArrayRef<const int32_t> pairlist,
                                                                   ArrayRef<const RVec> shiftVectors,
                                                                   ArrayRef<const IVec> cellShifts,
                                                                   ArrayRef<const RVec> positions,
                                                                   torch::Device        device,
                                                                   torch::ScalarType    dtype)
{
    const int64_t n_pairs = static_cast<int64_t>(pairlist.size() / 2);

    // Prepare CPU tensors first to facilitate efficient element access
    auto cpu_int_options   = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto cpu_float_options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);

    // Samples: [first_atom, second_atom, cell_shift_a, cell_shift_b, cell_shift_c]
    auto pair_samples_values = torch::zeros({ n_pairs, 5 }, cpu_int_options);
    auto pair_samples_ptr    = pair_samples_values.accessor<int32_t, 2>();

    // Values: Full interatomic vectors (rj - ri + shift)
    auto vectors_cpu      = torch::zeros({ n_pairs, 3, 1 }, cpu_float_options);
    auto vectors_accessor = vectors_cpu.accessor<double, 3>();

    for (int64_t i = 0; i < n_pairs; i++)
    {
        const int32_t atom_i = pairlist[2 * i];
        const int32_t atom_j = pairlist[2 * i + 1];

        // Fill sample metadata
        pair_samples_ptr[i][0] = static_cast<int32_t>(atom_i);
        pair_samples_ptr[i][1] = static_cast<int32_t>(atom_j);
        pair_samples_ptr[i][2] = cellShifts[i][0];
        pair_samples_ptr[i][3] = cellShifts[i][1];
        pair_samples_ptr[i][4] = cellShifts[i][2];

        // Calculate r_ij = r_j - r_i + shift
        const double r_ij_x =
                static_cast<double>(positions[atom_j][0] - positions[atom_i][0] + shiftVectors[i][0]);
        const double r_ij_y =
                static_cast<double>(positions[atom_j][1] - positions[atom_i][1] + shiftVectors[i][1]);
        const double r_ij_z =
                static_cast<double>(positions[atom_j][2] - positions[atom_i][2] + shiftVectors[i][2]);

        vectors_accessor[i][0][0] = r_ij_x;
        vectors_accessor[i][1][0] = r_ij_y;
        vectors_accessor[i][2][0] = r_ij_z;
    }

    // Move data to target device and type
    auto final_samples_values = pair_samples_values.to(device);
    auto final_vectors        = vectors_cpu.to(dtype).to(device);

    auto neighbor_samples = torch::make_intrusive<metatensor_torch::LabelsHolder>(
            std::vector<std::string>{
                    "first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c" },
            final_samples_values);

    auto neighbor_component = torch::make_intrusive<metatensor_torch::LabelsHolder>(
            std::vector<std::string>{ "xyz" },
            torch::tensor({ 0, 1, 2 }, torch::TensorOptions().dtype(torch::kInt32).device(device))
                    .reshape({ 3, 1 }));

    auto neighbor_properties = torch::make_intrusive<metatensor_torch::LabelsHolder>(
            std::vector<std::string>{ "distance" },
            torch::zeros({ 1, 1 }, torch::TensorOptions().dtype(torch::kInt32).device(device)));

    return torch::make_intrusive<metatensor_torch::TensorBlockHolder>(
            final_vectors,
            neighbor_samples,
            std::vector<metatensor_torch::Labels>{ neighbor_component },
            neighbor_properties);
}


/*! \brief Internal data structure for Metatomic runtime states.
 *
 * Encapsulates the Torch model, device configurations, capabilities, and options
 * required for model evaluation.
 */
struct MetatomicData
{
    metatensor_torch::Module           model = metatensor_torch::Module(torch::jit::Module());
    metatomic_torch::ModelCapabilities capabilities;
    std::vector<metatomic_torch::NeighborListOptions> nl_requests;
    metatomic_torch::ModelEvaluationOptions           evaluations_options;
    torch::ScalarType                                 dtype             = torch::kFloat32;
    bool                                              check_consistency = false;
    torch::Device                                     device            = torch::kCPU;
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

    // Enable profiling via environment variable GMX_METATOMIC_TIMER=1
    if (const char* timerEnv = std::getenv("GMX_METATOMIC_TIMER"))
    {
        MetatomicTimer::enable(std::string(timerEnv) == "1");
    }

    // Pairlist-based neighbor lists don't work with domain decomposition yet (indices are local)
    // Matches NNPot's limitation
    if (mpiComm_.isParallel())
    {
        GMX_LOG(logger_.warning)

                .asParagraph()

                .appendText(
                        "Metatomic support domain decomposition is EXPERIMENTAL (MPI). "
                        "Please use thread-MPI (gmx mdrun -ntmpi X) instead of MPI "
                        "(mpirun -np X gmx_mpi mdrun).");
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

    // Determine computation device
    torch::optional<std::string> desiredDevice = torch::nullopt;
    if (const char* env = std::getenv("GMX_METATOMIC_DEVICE"))
    {
        desiredDevice = std::string(env);
    }

    const auto deviceType =
            metatomic_torch::pick_device(data_->capabilities->supported_devices, desiredDevice);
    data_->device = torch::Device(deviceType);

    GMX_LOG(logger_.info)
            .asParagraph()
            .appendTextFormatted("Metatomic using device: %s", data_->device.str().c_str());

    // Process neighbor list requests from the model
    auto requests_ivalue = data_->model.run_method("requested_neighbor_lists");
    for (const auto& request_ivalue : requests_ivalue.toList())
    {
        data_->nl_requests.push_back(
                request_ivalue.get().toCustomClass<metatomic_torch::NeighborListOptionsHolder>());
    }

    data_->model.to(data_->device);

    // Configure precision
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

    // Validate energy output existence
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

    auto requested_output = torch::make_intrusive<metatomic_torch::ModelOutputHolder>();
    // TODO: take from the user
    requested_output->per_atom           = false;
    requested_output->explicit_gradients = {};
    requested_output->set_unit("kJ/mol");

    data_->evaluations_options->outputs.insert(energy_key, requested_output);
    data_->check_consistency = options_.params_.checkConsistency;


    // Initialize vectors for atom mapping
    const auto&   mtaIndices = options_.params_.mtaIndices_;
    const int32_t n_atoms    = static_cast<int32_t>(mtaIndices.size());

    positions_.resize(n_atoms);
    atomNumbers_.resize(n_atoms, 0);
    inputToLocalIndex_.resize(n_atoms, -1);
    inputToGlobalIndex_.resize(n_atoms, -1);

    GMX_LOG(logger_.info)
            .asParagraph()
            .appendText("MetatomicForceProvider initialization complete.");
}

MetatomicForceProvider::~MetatomicForceProvider() = default;

/*! \brief Updates the mapping between GROMACS local/global atom indices and the Metatomic model's input atoms.
 *
 * This function is subscribed to the `MDModulesAtomsRedistributedSignal`. It is called whenever
 * atoms are redistributed across MPI ranks (Domain Decomposition) or reordered in memory (sorting).
 *
 * Its primary responsibilities are:
 * 1. **Locate Input Atoms**: It iterates through the local atoms on the current rank to find
 * which atoms correspond to the "input atoms" defined for the Metatomic model (via `mtaIndices`).
 * 2. **Update Maps**: It populates `inputToLocalIndex_` (mapping model input index -> GROMACS local index)
 * and `inputToGlobalIndex_` (mapping model input index -> GROMACS global tag).
 * 3. **Gather Atomic Numbers**: It ensures `atomNumbers_` (Z numbers) are correctly associated with
 * the current local atoms, performing an MPI reduction if necessary to gather data from
 * distributed ranks.
 *
 * \param[in] signal Contains the new mapping of global atom indices to local buffer indices after redistribution.
 */
void MetatomicForceProvider::gatherAtomNumbersIndices(const MDModulesAtomsRedistributedSignal& signal)
{
    const auto&   mtaIndices = options_.params_.mtaIndices_;
    const int32_t numInput   = static_cast<int32_t>(mtaIndices.size());

    // Reset mappings
    inputToLocalIndex_.assign(numInput, -1);
    inputToGlobalIndex_.assign(numInput, -1);
    atomNumbers_.assign(numInput, 0);

    // GROMACS domain decomposition logic
    if (mpiComm_.isParallel())
    {
        GMX_RELEASE_ASSERT(signal.globalAtomIndices_.has_value(),
                           "Global atom indices required for domain decomposition.");
        auto          globalAtomIndices = signal.globalAtomIndices_.value();
        const int32_t numLocal          = signal.x_.size();
        const int32_t numLocalPlusHalo  = globalAtomIndices.size();

        // Size to include both home and halo atoms
        localToModelIndex_.assign(numLocalPlusHalo, -1);
        numLocalAtoms_ = numLocal;

        for (int32_t i = 0; i < numLocalPlusHalo; i++)
        {
            int32_t globalIdx = globalAtomIndices[i];
            for (int32_t j = 0; j < numInput; j++)
            {
                // Match current local atom to one of the requested Metatomic input atoms
                if (options_.params_.mtaAtoms_->globalIndex()[j] == globalIdx)
                {
                    localToModelIndex_[i] = j;
                    std::fprintf(stderr,
                                 "Rank %d: Found ModelAtom %d (Global %d) at Local %d (%s)\n",
                                 mpiComm_.rank(),
                                 j,
                                 globalIdx,
                                 i,
                                 (i < numLocal) ? "HOME" : "HALO");
                    if (i < numLocal)
                    {
                        inputToLocalIndex_[j]  = i;
                        inputToGlobalIndex_[j] = globalIdx;
                        atomNumbers_[j]        = options_.params_.atoms_.atom[globalIdx].atomnumber;
                    }
                    break;
                }
            }
        }
        // Reduce atomic numbers across ranks to ensure the main rank has the full set
        mpiComm_.sumReduce(numInput, atomNumbers_.data());

        // Debug logging for domain decomposition distribution
        int32_t homeCount = 0;
        int32_t haloCount = 0;
        for (int32_t i = 0; i < numLocalPlusHalo; i++)
        {
            if (localToModelIndex_[i] != -1)
            {
                if (i < numLocal)
                {
                    homeCount++;
                }
                else
                {
                    haloCount++;
                }
            }
        }
        std::fprintf(stderr,
                     "Rank %d: Mapped %d HOME + %d HALO = %d / %d Metatomic atoms.\n",
                     mpiComm_.rank(),
                     homeCount,
                     haloCount,
                     homeCount + haloCount,
                     numInput);
    }
    else
    {
        // Thread-MPI or Serial execution
        const auto* mtaAtoms = options_.params_.mtaAtoms_.get();
        localToModelIndex_.clear();
        numLocalAtoms_ = 0;
        for (int32_t i = 0; i < numInput; i++)
        {
            int32_t localIndex     = mtaAtoms->localIndex()[i];
            int32_t globalIdx      = mtaAtoms->globalIndex()[mtaAtoms->collectiveIndex()[i]];
            inputToLocalIndex_[i]  = localIndex;
            inputToGlobalIndex_[i] = globalIdx;
            atomNumbers_[i]        = options_.params_.atoms_.atom[globalIdx].atomnumber;
        }
    }

    GMX_RELEASE_ASSERT(std::count(atomNumbers_.begin(), atomNumbers_.end(), 0) == 0,
                       "Some atom numbers not set.");
}

/*! \brief Updates the internal position buffer with the coordinates of the Metatomic atoms.
 *
 * This function extracts the coordinates of the atoms relevant to the Metatomic model
 * from the full GROMACS local atom array (`pos`).
 *
 * 1. **Filtering:** `pos` contains all local atoms (solute, solvent, ions).
 * This function copies only the atoms defined in `mtaIndices` to `positions_`.
 * 2. **Ordering/Packing:** GROMACS reorders atoms dynamically for Domain Decomposition.
 * This function uses `inputToLocalIndex_` to collect atoms and pack them into a contiguous, ordered buffer
 *
 * \param[in] pos The array of all local atom coordinates for the current step.
 */
void MetatomicForceProvider::gatherAtomPositions(ArrayRef<const RVec> pos)
{
    const size_t numInput = inputToLocalIndex_.size();
    positions_.assign(numInput, RVec({ 0.0, 0.0, 0.0 }));

    for (size_t i = 0; i < numInput; i++)
    {
        if (inputToLocalIndex_[i] != -1)
        {
            positions_[i] = pos[inputToLocalIndex_[i]];
        }
    }

    if (mpiComm_.isParallel())
    {
        mpiComm_.sumReduce(3 * numInput, positions_.data()->as_vec());
    }
}

void MetatomicForceProvider::setPairlist(const MDModulesPairlistConstructedSignal& signal)
{
    // Capture the pairlist signal. Processing is deferred to
    // calculateForces/preparePairlistInput to keep this callback fast.
    fullPairlist_.assign(signal.excludedPairlist_.begin(), signal.excludedPairlist_.end());
    doPairlist_ = true;
}

/*! \brief Converts the GROMACS neighbor list to a model-compatible list.
 *
 * This function iterates over the full GROMACS excluded pairlist (which contains pairs in
 * GROMACS local atom indices). It filters this list to retain only pairs where *both* atoms
 * are part of the Metatomic model's input set.
 *
 * It populates `pairlistForModel_` (using model-relative indices), `shiftVectors_`,
 * and `cellShifts_`.
 */
void MetatomicForceProvider::preparePairlistInput()
{
    if (!doPairlist_)
    {
        return;
    }

    // Although the assert catches empty pairlists, in a real simulation with a very large cutoff,
    // this might happen legitimately if only 1 atom exists. However, for standard MD, it indicates
    // an issue. Assert here to catch initialization ordering bugs.
    GMX_ASSERT(!fullPairlist_.empty(), "Pairlist for Metatomic is empty!");

    const int32_t numPairs = gmx::ssize(fullPairlist_);
    pairlistForModel_.clear();
    pairlistForModel_.reserve(2 * numPairs);
    shiftVectors_.clear();
    shiftVectors_.reserve(numPairs);
    cellShifts_.clear();
    cellShifts_.reserve(numPairs);

    for (int32_t i = 0; i < numPairs; i++)
    {
        const auto [atomPair, shiftIndex] = fullPairlist_[i];

        // GROMACS pairlists use local atom indices.
        // Map these local indices back to the model's input indices [0, N_model_atoms).
        // `inputToLocalIndex_` maps ModelIdx -> LocalIdx.
        // indexOf reverses the map: Find ModelIdx k such that inputToLocalIndex_[k] == LocalIdx.
        const int32_t inputIdxA = localToModelIndex_[atomPair.first];

        if (inputIdxA != -1)
        {
            const int32_t inputIdxB = localToModelIndex_[atomPair.second];

            if (inputIdxB != -1)
            {
                // Both atoms belong to the Metatomic subsystem.
                // Calculate the shift vector due to PBC.
                RVec       shift;
                const IVec unitShift = shiftIndexToXYZ(shiftIndex);
                mvmul_ur0(box_, unitShift.toRVec(), shift);

                pairlistForModel_.push_back(inputIdxA);
                pairlistForModel_.push_back(inputIdxB);
                std::fprintf(stderr,
                             "Rank %d: Signal pair (Local %d, %d) -> Model (%d, %d)\n",
                             mpiComm_.rank(),
                             atomPair.first,
                             atomPair.second,
                             inputIdxA,
                             inputIdxB);
                shiftVectors_.push_back(shift);
                cellShifts_.push_back(unitShift);
            }
        }
    }

    GMX_RELEASE_ASSERT(pairlistForModel_.size() == shiftVectors_.size() * 2,
                       "Pairlist/shift size mismatch.");
    doPairlist_ = false;
}

void MetatomicForceProvider::augmentGhostPairs(const ArrayRef<const RVec> x, const matrix box)
{
    if (!mpiComm_.isParallel())
    {
        return;
    }

    // Identify halo MTA atoms: atoms in localToModelIndex_ that have a valid model index
    // but are NOT home atoms on this rank (i.e. inputToLocalIndex_[modelIdx] == -1).
    // These are atoms in the halo zone (local index >= numLocalAtoms_).
    std::vector<int32_t> haloLocalIndices;
    std::vector<RVec>    haloCoords;

    for (int32_t i = numLocalAtoms_; i < static_cast<int32_t>(localToModelIndex_.size()); i++)
    {
        if (localToModelIndex_[i] != -1)
        {
            haloLocalIndices.push_back(i);
            haloCoords.push_back(x[i]);
        }
    }

    std::fprintf(stderr,
                 "Rank %d: augmentGhostPairs found %zu halo MTA atoms\n",
                 mpiComm_.rank(),
                 haloCoords.size());

    if (haloCoords.size() < 2)
    {
        return;
    }

    t_pbc pbc;
    set_pbc(&pbc, *options_.params_.pbcType_, box);

    gmx::AnalysisNeighborhood nb;
    nb.setCutoff(data_->nl_requests[0]->cutoff());

    gmx::AnalysisNeighborhoodPositions ghostPositions(as_rvec_array(haloCoords.data()), haloCoords.size());

    gmx::AnalysisNeighborhoodSearch     search      = nb.initSearch(&pbc, ghostPositions);
    gmx::AnalysisNeighborhoodPairSearch ghostSearch = search.startSelfPairSearch();
    gmx::AnalysisNeighborhoodPair       pair;

    int32_t augmentedCount = 0;
    while (ghostSearch.findNextPair(&pair))
    {
        const int32_t localIdxA = haloLocalIndices[pair.refIndex()];
        const int32_t localIdxB = haloLocalIndices[pair.testIndex()];

        const int32_t inputIdxA = localToModelIndex_[localIdxA];
        const int32_t inputIdxB = localToModelIndex_[localIdxB];

        // Both should be valid since we pre-filtered, but guard anyway
        if (inputIdxA != -1 && inputIdxB != -1)
        {
            // pair.dx() returns the PBC-correct vector from ref to test.
            // buildNeighborListFromPairlist computes: r_ij = positions_[B] - positions_[A] + shift
            // We need: shift = pair.dx() - (positions_[B] - positions_[A])
            rvec modelDiff;
            rvec_sub(positions_[inputIdxB].as_vec(), positions_[inputIdxA].as_vec(), modelDiff);

            rvec shift;
            rvec_sub(pair.dx(), modelDiff, shift);

            // Compute integer cell shifts via box matrix inversion
            double det = box[0][0] * (box[1][1] * box[2][2] - box[1][2] * box[2][1])
                         - box[0][1] * (box[1][0] * box[2][2] - box[1][2] * box[2][0])
                         + box[0][2] * (box[1][0] * box[2][1] - box[1][1] * box[2][0]);

            IVec unitShift;
            if (std::abs(det) > 1e-10)
            {
                double invDet = 1.0 / det;
                rvec   unitShiftRvec;
                unitShiftRvec[0] = invDet
                                   * (shift[0] * (box[1][1] * box[2][2] - box[1][2] * box[2][1])
                                      + shift[1] * (box[0][2] * box[2][1] - box[0][1] * box[2][2])
                                      + shift[2] * (box[0][1] * box[1][2] - box[0][2] * box[1][1]));
                unitShiftRvec[1] = invDet
                                   * (shift[0] * (box[1][2] * box[2][0] - box[1][0] * box[2][2])
                                      + shift[1] * (box[0][0] * box[2][2] - box[0][2] * box[2][0])
                                      + shift[2] * (box[0][2] * box[1][0] - box[0][0] * box[1][2]));
                unitShiftRvec[2] = invDet
                                   * (shift[0] * (box[1][0] * box[2][1] - box[1][1] * box[2][0])
                                      + shift[1] * (box[0][1] * box[2][0] - box[0][0] * box[2][1])
                                      + shift[2] * (box[0][0] * box[1][1] - box[0][1] * box[1][0]));

                unitShift[0] = static_cast<int>(std::round(unitShiftRvec[0]));
                unitShift[1] = static_cast<int>(std::round(unitShiftRvec[1]));
                unitShift[2] = static_cast<int>(std::round(unitShiftRvec[2]));
            }
            else
            {
                unitShift = { 0, 0, 0 };
            }

            // Recompute shift from integer cell shifts for consistency with preparePairlistInput
            RVec finalShift;
            mvmul_ur0(box, unitShift.toRVec(), finalShift);

            pairlistForModel_.push_back(inputIdxA);
            pairlistForModel_.push_back(inputIdxB);
            shiftVectors_.push_back(finalShift);
            cellShifts_.push_back(unitShift);
            augmentedCount++;

            std::fprintf(stderr,
                         "[Augmented] Rank %d: Halo pair (Local %d, %d) -> Model (%d, %d) "
                         "shift=(%d,%d,%d)\n",
                         mpiComm_.rank(),
                         localIdxA,
                         localIdxB,
                         inputIdxA,
                         inputIdxB,
                         unitShift[0],
                         unitShift[1],
                         unitShift[2]);
        }
    }

    std::fprintf(stderr, "Rank %d: augmentGhostPairs added %d halo-halo pairs\n", mpiComm_.rank(), augmentedCount);
}


void MetatomicForceProvider::calculateForces(const ForceProviderInput& inputs, ForceProviderOutput* outputs)
{
    MetatomicTimer totalTimer("calculateForces", mpiComm_);

    const int32_t n_atoms = static_cast<int32_t>(options_.params_.mtaIndices_.size());

    // Update positions and box for the current step
    {
        MetatomicTimer timer("gatherAtomPositions", mpiComm_);
        gatherAtomPositions(inputs.x_);
    }
    copy_mat(inputs.box_, box_);

    {
        MetatomicTimer timer("preparePairlistInput", mpiComm_);
        preparePairlistInput();
    }

    const int32_t signalPairs = static_cast<int32_t>(pairlistForModel_.size() / 2);

    {
        MetatomicTimer timer("augmentGhostPairs", mpiComm_);
        augmentGhostPairs(inputs.x_, inputs.box_);
    }

    const int32_t totalPairsBeforeDedup = static_cast<int32_t>(pairlistForModel_.size() / 2);

    // Deduplicate pairs: the signal may already include some halo-halo pairs
    // that augmentGhostPairs also finds. Metatensor requires unique labels.
    {
        MetatomicTimer timer("deduplicatePairs", mpiComm_);

        using PairKey = std::tuple<int32_t, int32_t, int, int, int>;
        std::set<PairKey>    seen;
        std::vector<int32_t> dedupPairlist;
        std::vector<RVec>    dedupShifts;
        std::vector<IVec>    dedupCellShifts;

        const int32_t nPairs = static_cast<int32_t>(pairlistForModel_.size() / 2);
        dedupPairlist.reserve(pairlistForModel_.size());
        dedupShifts.reserve(nPairs);
        dedupCellShifts.reserve(nPairs);

        for (int32_t i = 0; i < nPairs; i++)
        {
            int32_t a = pairlistForModel_[2 * i];
            int32_t b = pairlistForModel_[2 * i + 1];
            PairKey key(a, b, cellShifts_[i][0], cellShifts_[i][1], cellShifts_[i][2]);

            if (seen.insert(key).second)
            {
                dedupPairlist.push_back(a);
                dedupPairlist.push_back(b);
                dedupShifts.push_back(shiftVectors_[i]);
                dedupCellShifts.push_back(cellShifts_[i]);
            }
        }

        const int32_t removed = nPairs - static_cast<int32_t>(dedupShifts.size());
        if (removed > 0)
        {
            std::fprintf(stderr, "Rank %d: Removed %d duplicate pairs\n", mpiComm_.rank(), removed);
        }

        pairlistForModel_ = std::move(dedupPairlist);
        shiftVectors_     = std::move(dedupShifts);
        cellShifts_       = std::move(dedupCellShifts);
    }

    const int32_t totalPairs = static_cast<int32_t>(pairlistForModel_.size() / 2);
    std::fprintf(stderr,
                 "Rank %d Step %ld: %d signal + %d augmented - %d dupes = %d unique pairs, "
                 "%d model atoms, homenr=%d, x.size=%zu\n",
                 mpiComm_.rank(),
                 inputs.step_,
                 signalPairs,
                 totalPairsBeforeDedup - signalPairs,
                 totalPairsBeforeDedup - totalPairs,
                 totalPairs,
                 n_atoms,
                 inputs.homenr_,
                 inputs.x_.size());

    // Force tensor - main rank fills this, others hold zeros until reduction
    torch::Tensor forceTensor = torch::zeros(
            { n_atoms, 3 }, torch::TensorOptions().dtype(torch::kFloat64).device(data_->device));

    // Virial tensor for pressure/stress calculations
    torch::Tensor virialTensor = torch::zeros({ 3, 3 }, torch::TensorOptions().dtype(torch::kFloat64));

    if (mpiComm_.isMainRank())
    {
        MetatomicTimer modelTimer("model inference (main rank)", mpiComm_);

        // Select appropriate precision for GROMACS data conversion
        auto gromacs_scalar_type = torch::kFloat32;
        if (std::is_same_v<real, double>)
        {
            gromacs_scalar_type = torch::kFloat64;
        }
        auto cpu_blob_options = torch::TensorOptions().dtype(gromacs_scalar_type).device(torch::kCPU);

        auto torch_positions = torch::from_blob(positions_.data()->as_vec(), { n_atoms, 3 }, cpu_blob_options)
                                       .to(data_->dtype)
                                       .to(data_->device)
                                       .set_requires_grad(true);

        auto torch_cell =
                torch::from_blob(&box_, { 3, 3 }, cpu_blob_options).to(data_->dtype).to(data_->device);

        // Create strain tensor (identity matrix) for virial computation via autodiff
        auto strain = torch::eye(
                3, torch::TensorOptions().dtype(data_->dtype).device(data_->device).requires_grad(true));

        // Apply strain to cell: strained_cell = cell @ strain
        auto strained_cell = torch::matmul(torch_cell, strain);
        // Apply strain to positions: r' = r @ strain
        auto strained_positions = torch::matmul(torch_positions, strain);

        auto torch_pbc = preparePbcType(options_.params_.pbcType_.get(), data_->device);
        auto torch_types =
                torch::tensor(atomNumbers_, torch::TensorOptions().dtype(torch::kInt32)).to(data_->device);

        auto system = torch::make_intrusive<metatomic_torch::SystemHolder>(
                torch_types, strained_positions, strained_cell, torch_pbc);

        // Build neighbor list from GROMACS pairlist
        for (const auto& request : data_->nl_requests)
        {
            auto neighbors = buildNeighborListFromPairlist(
                    pairlistForModel_, shiftVectors_, cellShifts_, positions_, data_->device, data_->dtype);
            metatomic_torch::register_autograd_neighbors(system, neighbors, data_->check_consistency);
            system->add_neighbor_list(request, neighbors);
        }

        metatensor_torch::TensorMap output_map;
        try
        {
            std::vector<metatomic_torch::System> systems;
            systems.push_back(system);

            // Forward pass
            auto ivalue_output = data_->model.forward(
                    { systems, data_->evaluations_options, data_->check_consistency });
            auto dict_output = ivalue_output.toGenericDict();
            output_map = dict_output.at("energy").toCustomClass<metatensor_torch::TensorMapHolder>();
        }
        catch (const std::exception& e)
        {
            GMX_THROW(APIError("[Metatomic] Model evaluation failed: " + std::string(e.what())));
        }

        // Extract Energy
        auto energy_block  = metatensor_torch::TensorMapHolder::block_by_id(output_map, 0);
        auto energy_tensor = energy_block->values();

        outputs->enerd_.term[InteractionFunction::MetatomicPotentialEnergy] =
                static_cast<real>(energy_tensor.sum().item<double>());

        // Reset gradients before backward
        torch_positions.mutable_grad() = torch::Tensor();
        strain.mutable_grad()          = torch::Tensor();

        // Backward pass: Compute forces (-dE/dr) and virial (-dE/dStrain)
        energy_tensor.backward(-torch::ones_like(energy_tensor));

        auto grad   = torch_positions.grad();
        forceTensor = grad.to(torch::kCPU).to(torch::kFloat64);

        // Get virial from strain gradient
        virialTensor = strain.grad().to(torch::kCPU).to(torch::kFloat64);
    }

    // Distribute results to all ranks if necessary (sumReduce broadcasts if ranks > 1)
    if (mpiComm_.isParallel())
    {
        MetatomicTimer mpiTimer("MPI force/virial reduction", mpiComm_);
        mpiComm_.sumReduce(n_atoms * 3, static_cast<double*>(forceTensor.data_ptr()));
        mpiComm_.sumReduce(9, static_cast<double*>(virialTensor.data_ptr()));
    }

    // Accumulate forces into the GROMACS force output
    auto forceAccessor = forceTensor.accessor<double, 2>();
    for (int32_t i = 0; i < n_atoms; ++i)
    {
        // Only apply force if this atom is local to this rank
        if (inputToLocalIndex_[i] != -1)
        {
            outputs->forceWithVirial_.force_[inputToLocalIndex_[i]][0] += forceAccessor[i][0];
            outputs->forceWithVirial_.force_[inputToLocalIndex_[i]][1] += forceAccessor[i][1];
            outputs->forceWithVirial_.force_[inputToLocalIndex_[i]][2] += forceAccessor[i][2];
        }
    }

    // Apply virial contribution
    // GROMACS uses a 3x3 virial tensor in forceWithVirial_
    // Copy the tensor data into a GROMACS matrix and use the public API
    matrix virialMatrix;
    auto   virialAccessor = virialTensor.accessor<double, 2>();
    // TODO: technically this is DIM, not 3...
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
