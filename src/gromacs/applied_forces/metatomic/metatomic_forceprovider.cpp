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
 * Implements the Metatomic Force Provider class
 *
 * \author Metatensor developers <https://github.com/metatensor>
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "metatomic_forceprovider.h"

#include <vesin.h>

#include <cstdint>

#include <filesystem>

#ifndef DIM
#    define DIM 3
#endif
#include "gromacs/domdec/localatomset.h"
#include "gromacs/mdlib/broadcaststructs.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/mpicomm.h"

using namespace std::string_literals; // For ""s
namespace gmx
{

MetatomicForceProvider::MetatomicForceProvider(const MetatomicOptions& options,
                                               const MDLogger&         logger,
                                               const MpiComm&          mpiComm) :
    options_(options),
    logger_(logger),
    mpiComm_(mpiComm),
    device_(torch::Device(torch::kCPU)),
    box_{ { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } }
{
    // NOTE: do NOT return early on non-main ranks. Only perform file I/O / GPU init
    // on the main rank, but still run the remainder of the constructor on all ranks
    // so that all ranks call the same MPI collectives.
    if (mpiComm_.isMainRank())
    {
        GMX_LOG(logger_.info).asParagraph().appendText("Initializing MetatomicForceProvider...");

        if (!std::filesystem::exists(options_.params_.modelPath_))
        {
            GMX_THROW(FileIOError("Metatomic model file does not exist: " + options_.params_.modelPath_));
        }

        // Load the model
        try
        {
            torch::optional<std::string> extensions_directory = torch::nullopt;
            if (!options_.params_.extensionsDirectory.empty())
            {
                extensions_directory = options_.params_.extensionsDirectory;
            }

            model_ = metatomic_torch::load_atomistic_model(options_.params_.modelPath_, extensions_directory);
        }
        catch (const std::exception& e)
        {
            GMX_THROW(APIError("Failed to load metatomic model: " + std::string(e.what())));
        }

        // Query model capabilities
        capabilities_ =
                model_.run_method("capabilities").toCustomClass<metatomic_torch::ModelCapabilitiesHolder>();
        auto requests_ivalue = model_.run_method("requested_neighbor_lists");
        for (const auto& request_ivalue : requests_ivalue.toList())
        {
            nl_requests_.push_back(
                    request_ivalue.get().toCustomClass<metatomic_torch::NeighborListOptionsHolder>());
        }

        // TODO(rg): determine device
        model_.to(device_);

        // Set data type locally on main rank
        if (capabilities_->dtype() == "float64")
        {
            dtype_ = torch::kFloat64;
        }
        else if (capabilities_->dtype() == "float32")
        {
            dtype_ = torch::kFloat32;
        }
        else
        {
            GMX_THROW(APIError("Unsupported dtype from model: " + capabilities_->dtype()));
        }

        // 5. Set up evaluation options (only on main rank)
        evaluations_options_ = torch::make_intrusive<metatomic_torch::ModelEvaluationOptionsHolder>();
        evaluations_options_->set_length_unit("nm");

        auto outputs = capabilities_->outputs();
        if (!outputs.contains("energy"))
        {
            GMX_THROW(APIError("Metatomic model must provide an 'energy' output."));
        }

        auto requested_output      = torch::make_intrusive<metatomic_torch::ModelOutputHolder>();
        requested_output->per_atom = false;
        requested_output->explicit_gradients = {}; // Use autograd for forces

        evaluations_options_->outputs.insert("energy", requested_output);
    }
    else
    {
        // non-main ranks do not load model or query capabilities, but they must still participate
        // in collectives and must know dtype_ afterwards; dtype_ will be broadcast below.
    }

    // Initialize ML atom group data structures (must run on all ranks)
    const auto& mtaIndices = options_.params_.mtaIndices_;
    const int   n_atoms    = static_cast<int>(mtaIndices.size());

    // Resize vectors that will be used on all ranks.
    positions_.resize(n_atoms);
    atomNumbers_.resize(n_atoms);
    idxLookup_.resize(n_atoms); // Filled by gatherAtomNumbersIndices

    // Atom numbers are static. We will populate them on the main rank
    // and "broadcast" using sumReduce.

    // Ensure non-main ranks have zeroed atomNumbers_ so a sumReduce acts like a broadcast.
    std::fill(atomNumbers_.begin(), atomNumbers_.end(), 0);

    // Now, only the main rank populates its version of the data.
    if (mpiComm_.isMainRank())
    {
        for (int i = 0; i < n_atoms; ++i)
        {
            const int gIdx = mtaIndices[i];

            if (gIdx >= options_.params_.numAtoms_)
            {
                GMX_THROW(APIError("Metatomic atom index " + std::to_string(gIdx) + " is out of bounds for topology numAtoms="
                                   + std::to_string(options_.params_.numAtoms_)));
            }
            atomNumbers_[i] = options_.params_.atoms_.atom[gIdx].atomnumber;
        }
    }

    // "Broadcast" the static atom numbers to all ranks: all ranks must call this.
    if (mpiComm_.isParallel())
    {
        // resize/allocate on non-main ranks and broadcast raw bytes of vector
        nblock_abc(mpiComm_.isMainRank(), mpiComm_.comm(), static_cast<std::size_t>(n_atoms), &atomNumbers_);
    }

    // Broadcast dtype info (so all ranks know whether the model uses float32 or float64).
    // 0 = float64, 1 = float32
    int dtype_code_local = 0;
    if (mpiComm_.isMainRank())
    {
        dtype_code_local = (dtype_ == torch::kFloat32) ? 1 : 0;
    }
    // comm() may be MPI_COMM_NULL for SingleRank so..
    if (mpiComm_.isParallel())
    {
        block_bc(mpiComm_.comm(), dtype_code_local);
    }
    // Set dtype_ on all ranks
    dtype_ = (dtype_code_local == 1) ? torch::kFloat32 : torch::kFloat64;

    // Initialize the lookup table. It will be populated correctly
    // by the first call from the SimulationRunNotifier.
    std::fill(idxLookup_.begin(), idxLookup_.end(), -1);

    if (mpiComm_.isMainRank())
    {
        GMX_LOG(logger_.info)
                .asParagraph()
                .appendText("MetatomicForceProvider initialization complete.");
    }
}

void MetatomicForceProvider::gatherAtomNumbersIndices()
{
    // This function is called on domain decomposition.
    // Its ONLY job is to update the `idxLookup_` map, which maps
    // the contiguous ML atom index (0..n_atoms-1) to the
    // sparse GROMACS local index.
    //
    // `atomNumbers_` is static and was set/broadcast in the constructor.

    const auto& mtaIndices = options_.params_.mtaIndices_;
    const int   n_atoms    = static_cast<int>(mtaIndices.size());

    // Reset the lookup table to -1 (sentinel for "not local")
    std::fill(idxLookup_.begin(), idxLookup_.end(), -1);

    // Build a reverse lookup map for efficiently finding ML atoms.
    // Map: (Global GROMACS Index) -> (ML Group Index, 0..n_atoms-1)
    std::unordered_map<int, int> globalToMlIndex;
    globalToMlIndex.reserve(n_atoms);
    for (int i = 0; i < n_atoms; ++i)
    {
        globalToMlIndex[mtaIndices[i]] = i;
    }

    // Iterate over this rank's LOCAL atoms and populate the lookup table if a
    // local atom is part of our ML group.
    const auto* mtaAtoms = options_.params_.mtaAtoms_.get();
    for (size_t i = 0; i < mtaAtoms->numAtomsLocal(); ++i)
    {
        const int lIdx = mtaAtoms->localIndex()[i];
        const int gIdx = mtaAtoms->globalIndex()[mtaAtoms->collectiveIndex()[i]];

        // Check if this local atom is one of our ML atoms
        if (auto it = globalToMlIndex.find(gIdx); it != globalToMlIndex.end())
        {
            const int mlIdx   = it->second; // This is the index (0..n_atoms-1)
            idxLookup_[mlIdx] = lIdx;
        }
    }

    // Done. No MPI is needed. `gatherAtomPositions` will now correctly
    // use this rank-local `idxLookup_` to contribute its positions.
}

MetatomicForceProvider::~MetatomicForceProvider() = default;

void MetatomicForceProvider::gatherAtomPositions(ArrayRef<const RVec> globalPositions)
{
    const int n_atoms = static_cast<int>(options_.params_.mtaIndices_.size());
    positions_.assign(n_atoms, RVec{ 0.0, 0.0, 0.0 });

    for (int i = 0; i < n_atoms; ++i)
    {
        // If idxLookup_[i] is not -1, this atom is local to this rank
        if (idxLookup_[i] != -1)
        {
            positions_[i] = globalPositions[idxLookup_[i]];
        }
    }

    // All ranks need the complete, contiguous list of positions for the metatomic group.
    if (mpiComm_.isParallel())
    {
        // Flatten into a double or float buffer depending on dtype_
        const size_t total = static_cast<size_t>(n_atoms) * 3;
        if (dtype_ == torch::kFloat64)
        {
            std::vector<double> flat(total, 0.0);
            for (int i = 0; i < n_atoms; ++i)
            {
                flat[3 * i + 0] = static_cast<double>(positions_[i][0]);
                flat[3 * i + 1] = static_cast<double>(positions_[i][1]);
                flat[3 * i + 2] = static_cast<double>(positions_[i][2]);
            }
            mpiComm_.sumReduce(flat);
            // unpack back to positions_ (optional; code that reads positions_ expects them set)
            for (int i = 0; i < n_atoms; ++i)
            {
                positions_[i][0] = static_cast<float>(flat[3 * i + 0]);
                positions_[i][1] = static_cast<float>(flat[3 * i + 1]);
                positions_[i][2] = static_cast<float>(flat[3 * i + 2]);
            }
        }
        else
        {
            std::vector<float> flat(total, 0.0f);
            for (int i = 0; i < n_atoms; ++i)
            {
                flat[3 * i + 0] = static_cast<float>(positions_[i][0]);
                flat[3 * i + 1] = static_cast<float>(positions_[i][1]);
                flat[3 * i + 2] = static_cast<float>(positions_[i][2]);
            }
            mpiComm_.sumReduce(flat);
            for (int i = 0; i < n_atoms; ++i)
            {
                positions_[i][0] = static_cast<double>(flat[3 * i + 0]);
                positions_[i][1] = static_cast<double>(flat[3 * i + 1]);
                positions_[i][2] = static_cast<double>(flat[3 * i + 2]);
            }
        }
    }
}


void MetatomicForceProvider::calculateForces(const ForceProviderInput& inputs, ForceProviderOutput* outputs)
{
    const int n_atoms = static_cast<int>(options_.params_.mtaIndices_.size());

    // Gather all required data so it is available on every rank
    this->gatherAtomPositions(inputs.x_);
    copy_mat(inputs.box_, box_);
    torch::Tensor forceTensor;

    // Prepare a host-side buffer for forces. We will fill it on main rank and then sumReduce it.
    const size_t total = static_cast<size_t>(n_atoms) * 3;
    MPI_Comm     comm  = mpiComm_.comm();

    // Container that will hold final forces on all ranks after the collective.
    // We choose float or double according to dtype_.
    if (dtype_ == torch::kFloat64)
    {
        std::vector<double> global_force(total, 0.0);
        // main rank computes forces into a tensor, then we'll copy into global_force and sumReduce
        if (mpiComm_.isMainRank())
        {
            // Build torch inputs and run model on main rank
            auto dtype_options = torch::TensorOptions().dtype(dtype_).device(torch::kCPU);

            auto coerced_positions = makeArrayRef(positions_);
            auto torch_positions =
                    torch::from_blob(coerced_positions.data()->as_vec(), { n_atoms, 3 }, dtype_options)
                            .to(this->dtype_)
                            .to(this->device_)
                            .set_requires_grad(true);

            auto torch_cell = torch::from_blob(&box_, { 3, 3 }, dtype_options);

            auto cell_norms = torch::norm(torch_cell, 2, /*dim=*/1);
            auto torch_pbc =
                    torch::tensor(std::vector<uint8_t>{ 1, 1, 1 },
                                  torch::TensorOptions().dtype(torch::kBool).device(this->device_));
            auto torch_types =
                    torch::tensor(atomNumbers_, torch::TensorOptions().dtype(torch::kInt32)).to(this->device_);

            auto system = torch::make_intrusive<metatomic_torch::SystemHolder>(
                    torch_types, torch_positions, torch_cell, torch_pbc);

            bool periodic = torch::all(torch_pbc).item<bool>();
            // Compute and add neighbor lists
            for (const auto& request : nl_requests_)
            {
                auto neighbors = computeNeighbors(
                        request, n_atoms, coerced_positions.data()->as_vec(), box_, periodic);
                metatomic_torch::register_autograd_neighbors(system, neighbors, false);
                system->add_neighbor_list(request, neighbors);
            }

            // Run the model
            metatensor_torch::TensorMap output_map;
            try
            {
                auto ivalue_output = this->model_.forward({
                        std::vector<metatomic_torch::System>{ system },
                        evaluations_options_,
                        this->check_consistency_,
                });
                auto dict_output   = ivalue_output.toGenericDict();
                output_map = dict_output.at("energy").toCustomClass<metatensor_torch::TensorMapHolder>();
            }
            catch (const std::exception& e)
            {
                GMX_THROW(APIError("[MetatomicPotential] Model evaluation failed"s));
            }

            // Extract energy and compute forces via autograd
            auto energy_block  = metatensor_torch::TensorMapHolder::block_by_id(output_map, 0);
            auto energy_tensor = energy_block->values(); // This should be a [1, 1] tensor

            if (energy_tensor.sizes().vec() != std::vector<int64_t>{ 1, 1 })
            {
                GMX_THROW(APIError("Model did not return a single scalar energy value."s));
            }

            // Set energy output
            outputs->enerd_.term[F_EMETATOMICPOT] = energy_tensor.item<double>();

            // Compute gradients
            energy_tensor.backward();
            auto grad = system->positions().grad();

            forceTensor = -grad.to(torch::kCPU).to(dtype_);

            // Copy into host double buffer
            auto accessor = forceTensor.accessor<double, 2>();
            for (int i = 0; i < n_atoms; ++i)
            {
                global_force[3 * i + 0] = accessor[i][0];
                global_force[3 * i + 1] = accessor[i][1];
                global_force[3 * i + 2] = accessor[i][2];
            }
        } // main rank block

        // All ranks participate in this collective. After this call, global_force contains the final forces.
        if (mpiComm_.isParallel())
        {
            nblock_abc(mpiComm_.isMainRank(), comm, total, &global_force);
        }

        // 4. Each rank accumulates forces for its local atoms
        for (int i = 0; i < n_atoms; ++i)
        {
            const int localIndex = idxLookup_[i];
            if (localIndex != -1) // This atom is local to the current rank
            {
                for (int m = 0; m < DIM; ++m)
                {
                    outputs->forceWithVirial_.force_[localIndex][m] += global_force[3 * i + m];
                }
            }
        }
    }
    else // float32 case
    {
        std::vector<float> global_force(static_cast<size_t>(total), 0.0f);
        if (mpiComm_.isMainRank())
        {
            auto dtype_options = torch::TensorOptions().dtype(dtype_).device(torch::kCPU);

            auto coerced_positions = makeArrayRef(positions_);
            auto torch_positions =
                    torch::from_blob(coerced_positions.data()->as_vec(), { n_atoms, 3 }, dtype_options)
                            .to(this->dtype_)
                            .to(this->device_)
                            .set_requires_grad(true);

            auto torch_cell = torch::from_blob(&box_, { 3, 3 }, dtype_options);

            auto cell_norms = torch::norm(torch_cell, 2, /*dim=*/1);
            auto torch_pbc =
                    torch::tensor(std::vector<uint8_t>{ 1, 1, 1 },
                                  torch::TensorOptions().dtype(torch::kBool).device(this->device_));
            auto torch_types =
                    torch::tensor(atomNumbers_, torch::TensorOptions().dtype(torch::kInt32)).to(this->device_);

            auto system = torch::make_intrusive<metatomic_torch::SystemHolder>(
                    torch_types, torch_positions, torch_cell, torch_pbc);

            bool periodic = torch::all(torch_pbc).item<bool>();
            // Compute and add neighbor lists
            for (const auto& request : nl_requests_)
            {
                auto neighbors = computeNeighbors(
                        request, n_atoms, coerced_positions.data()->as_vec(), box_, periodic);
                metatomic_torch::register_autograd_neighbors(system, neighbors, false);
                system->add_neighbor_list(request, neighbors);
            }

            // Run the model
            metatensor_torch::TensorMap output_map;
            try
            {
                auto ivalue_output = this->model_.forward({
                        std::vector<metatomic_torch::System>{ system },
                        evaluations_options_,
                        this->check_consistency_,
                });
                auto dict_output   = ivalue_output.toGenericDict();
                output_map = dict_output.at("energy").toCustomClass<metatensor_torch::TensorMapHolder>();
            }
            catch (const std::exception& e)
            {
                throw std::runtime_error("[MetatomicPotential] Model evaluation failed");
            }

            // Extract energy and compute forces via autograd
            auto energy_block  = metatensor_torch::TensorMapHolder::block_by_id(output_map, 0);
            auto energy_tensor = energy_block->values(); // This should be a [1, 1] tensor

            if (energy_tensor.sizes().vec() != std::vector<int64_t>{ 1, 1 })
            {
                throw std::runtime_error("Model did not return a single scalar energy value.");
            }
            double energy = energy_tensor.item<double>();

            // Set energy output
            outputs->enerd_.term[F_EMETATOMICPOT] = energy;

            // Compute gradients
            energy_tensor.backward();
            auto grad = system->positions().grad();

            auto forceTensor = -grad.to(torch::kCPU).to(dtype_);

            // Copy into host float buffer
            auto accessor = forceTensor.accessor<float, 2>();
            for (int i = 0; i < n_atoms; ++i)
            {
                global_force[3 * i + 0] = accessor[i][0];
                global_force[3 * i + 1] = accessor[i][1];
                global_force[3 * i + 2] = accessor[i][2];
            }
        } // main rank float block

        // All ranks participate in this collective. After this call, global_force contains the final forces.
        if (mpiComm_.isParallel())
        {
            nblock_abc(mpiComm_.isMainRank(), comm, total, &global_force);
        }

        // 4. Each rank accumulates forces for its local atoms
        for (int i = 0; i < n_atoms; ++i)
        {
            const int localIndex = idxLookup_[i];
            if (localIndex != -1) // This atom is local to the current rank
            {
                for (int m = 0; m < DIM; ++m)
                {
                    outputs->forceWithVirial_.force_[localIndex][m] +=
                            static_cast<float>(global_force[3 * i + m]);
                }
            }
        }
    } // dtype branching

    // TODO(rg): Virial calculation. For now, GROMACS will (incorrectly) calculate it from forces if
    // needed. This is the same behavior as nnpot
}


metatensor_torch::TensorBlock MetatomicForceProvider::computeNeighbors(metatomic_torch::NeighborListOptions request,
                                                                       long         n_atoms,
                                                                       const float* positions,
                                                                       const matrix box,
                                                                       bool         periodic)
{
    auto cutoff = request->engine_cutoff("nm");

    VesinOptions options;
    options.cutoff           = cutoff;
    options.full             = request->full_list();
    options.return_shifts    = true;
    options.return_distances = false;
    options.return_vectors   = true;

    VesinNeighborList* vesin_neighbor_list = new VesinNeighborList();
    // .............................. gromacs likes floats, vesin does not
    double double_box[3][3];

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            double_box[i][j] = static_cast<double>(box[i][j]);
        }
    }

    const size_t        total_elements = static_cast<size_t>(n_atoms) * 3;
    std::vector<double> double_positions(total_elements);

    for (size_t i = 0; i < total_elements; i++)
    {
        double_positions[i] = static_cast<double>(positions[i]);
    }
    const double* positions_ptr = double_positions.data();

    VesinDevice cpu{ VesinCPU, 0 };
    const char* error_message = nullptr;
    int         status = vesin_neighbors(reinterpret_cast<const double (*)[3]>(positions_ptr),
                                 static_cast<size_t>(n_atoms),
                                 double_box,
                                 &periodic,
                                 cpu,
                                 options,
                                 vesin_neighbor_list,
                                 &error_message);


    if (status != EXIT_SUCCESS)
    {
        std::string err_str = "vesin_neighbors failed: ";
        if (error_message)
        {
            err_str += error_message;
        }
        delete vesin_neighbor_list;
        GMX_THROW(APIError(err_str));
    }

    auto n_pairs = static_cast<int64_t>(vesin_neighbor_list->length);

    // Build samples tensor (first_atom, second_atom, cell_shifts)
    auto pair_samples_values = torch::empty({ n_pairs, 5 }, torch::TensorOptions().dtype(torch::kInt32));
    auto pair_samples_ptr = pair_samples_values.accessor<int32_t, 2>();
    for (int64_t i = 0; i < n_pairs; i++)
    {
        pair_samples_ptr[i][0] = static_cast<int32_t>(vesin_neighbor_list->pairs[i][0]);
        pair_samples_ptr[i][1] = static_cast<int32_t>(vesin_neighbor_list->pairs[i][1]);
        pair_samples_ptr[i][2] = vesin_neighbor_list->shifts[i][0];
        pair_samples_ptr[i][3] = vesin_neighbor_list->shifts[i][1];
        pair_samples_ptr[i][4] = vesin_neighbor_list->shifts[i][2];
    }

    // Custom deleter to free vesin's memory when the torch tensor is destroyed
    auto deleter = [=](void*)
    {
        vesin_free(vesin_neighbor_list);
        delete vesin_neighbor_list;
    };

    auto pair_vectors = torch::from_blob(vesin_neighbor_list->vectors,
                                         { n_pairs, 3, 1 },
                                         deleter,
                                         torch::TensorOptions().dtype(this->dtype_));

    auto neighbor_samples = torch::make_intrusive<metatensor_torch::LabelsHolder>(
            std::vector<std::string>{
                    "first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c" },
            pair_samples_values.to(device_));

    auto neighbor_component = torch::make_intrusive<metatensor_torch::LabelsHolder>(
            std::vector<std::string>{ "xyz" },
            torch::tensor({ 0, 1, 2 }, torch::TensorOptions().dtype(torch::kInt32).device(device_))
                    .reshape({ 3, 1 }));

    auto neighbor_properties = torch::make_intrusive<metatensor_torch::LabelsHolder>(
            std::vector<std::string>{ "distance" },
            torch::zeros({ 1, 1 }, torch::TensorOptions().dtype(torch::kInt32).device(device_)));

    return torch::make_intrusive<metatensor_torch::TensorBlockHolder>(
            pair_vectors.to(this->dtype_).to(this->device_),
            neighbor_samples,
            std::vector<metatensor_torch::Labels>{ neighbor_component },
            neighbor_properties);
}


} // namespace gmx
