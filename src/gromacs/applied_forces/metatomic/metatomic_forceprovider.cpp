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
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/mpicomm.h"

namespace gmx
{

MetatomicForceProvider::MetatomicForceProvider(const MetatomicOptions& options,
                                               const MDLogger&         logger,
                                               const MpiComm&          mpiComm) :
    options_(options), logger_(logger), mpiComm_(mpiComm), device_(torch::Device(torch::kCPU))
{
    // All setup that involves file I/O or GPU initialization should only be done on the main rank.
    if (!mpiComm_.isMainRank())
    {
        return;
    }
    GMX_LOG(logger_.info).asParagraph().appendText("Initializing MetatomicForceProvider...");

    if (!std::filesystem::exists(options_.params_.modelPath_))
    {
        GMX_THROW(FileIOError("Metatomic model file does not exist: " + options_.params_.modelPath_));
    }

    // 1. Load the model
    try
    {
        model_ = metatomic_torch::load_atomistic_model(options_.params_.modelPath_, nullptr);
    }
    catch (const std::exception& e)
    {
        GMX_THROW(APIError("Failed to load metatomic model: " + std::string(e.what())));
    }

    // 2. Query model capabilities
    capabilities_ =
            model_.run_method("capabilities").toCustomClass<metatomic_torch::ModelCapabilitiesHolder>();
    auto requests_ivalue = model_.run_method("requested_neighbor_lists");
    for (const auto& request_ivalue : requests_ivalue.toList())
    {
        nl_requests_.push_back(
                request_ivalue.get().toCustomClass<metatomic_torch::NeighborListOptionsHolder>());
    }

    // 3. TODO(rg): determine device
    model_.to(device_);

    // 4. Set data type
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

    // 5. Set up evaluation options
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

    // Initialize data vectors
    const int n_atoms = options_.params_.mtaIndices_.size();
    positions_.resize(n_atoms);
    atomNumbers_.resize(n_atoms);
    idxLookup_.resize(n_atoms);

    GMX_LOG(logger_.info)
            .asParagraph()
            .appendText("MetatomicForceProvider initialization complete.");
}

MetatomicForceProvider::~MetatomicForceProvider() = default;

void MetatomicForceProvider::gatherAtomPositions(ArrayRef<const RVec> globalPositions)
{
    const int n_atoms = options_.params_.mtaIndices_.size();
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
        mpiComm_.sumReduce(3 * n_atoms, positions_.data()->as_vec());
    }
}


void MetatomicForceProvider::gatherAtomNumbersIndices()
{
    // this might not be the most efficient solution, since we are throwing away most of the
    // vectors here in case of NNP/MM

    // create lookup table for local atom indices needed for hybrid ML/MM
    // -1 is used as a flag for atoms that are not local / not in the input
    // used to distribute forces to correct local indices as the NN input tensor does not contain all atoms
    idxLookup_.assign(options_.params_.numAtoms_, -1);
    atomNumbers_.assign(options_.params_.numAtoms_, 0);

    int lIdx, gIdx;
    for (size_t i = 0; i < options_.params_.mtaAtoms_->numAtomsLocal(); i++)
    {
        lIdx = options_.params_.mtaAtoms_->localIndex()[i];
        gIdx = options_.params_.mtaAtoms_->globalIndex()[options_.params_.mtaAtoms_->collectiveIndex()[i]];
        // TODO: make sure that atom number indexing is correct
        atomNumbers_[gIdx] = options_.params_.atoms_.atom[gIdx].atomnumber;
        idxLookup_[gIdx]   = lIdx;
    }

    // distribute atom numbers to all ranks
    if (mpiComm_.isParallel())
    {
        mpiComm_.sumReduce(atomNumbers_);
    }

    // remove unused elements in atomNumbers_, and idxLookup
    auto atIt  = atomNumbers_.begin();
    auto idxIt = idxLookup_.begin();
    while (atIt != atomNumbers_.end() && idxIt != idxLookup_.end())
    {
        if (*atIt == 0)
        {
            atIt  = atomNumbers_.erase(atIt);
            idxIt = idxLookup_.erase(idxIt);
        }
        else
        {
            ++atIt;
            ++idxIt;
        }
    }
}

void MetatomicForceProvider::calculateForces(const ForceProviderInput& inputs, ForceProviderOutput* outputs)
{
    const int n_atoms = options_.params_.mtaIndices_.size();

    // 1. Gather all required data so it is available on every rank
    this->gatherAtomPositions(inputs.x_);
    copy_mat(inputs.box_, box_);

    torch::Tensor forceTensor = torch::zeros({ n_atoms, 3 }, torch::TensorOptions().dtype(dtype_));

    // 2. Perform model evaluation on the main rank only
    if (mpiComm_.isMainRank())
    {

        auto f64_options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);

        auto coerced_positions = makeArrayRef(positions_);
        auto torch_positions =
                torch::from_blob(coerced_positions.data()->as_vec(), { n_atoms, 3 }, f64_options)
                        .to(this->dtype_)
                        .to(this->device_)
                        .set_requires_grad(true);

        auto torch_cell = torch::from_blob(&box_, { 3, 3 }, f64_options);

        auto          cell_norms = torch::norm(torch_cell, 2, /*dim=*/1);
        auto          torch_pbc  = cell_norms.abs() > 1e-9;
        torch::Tensor pbcTensor =
                torch::tensor({ true, true, true }, torch::TensorOptions().dtype(torch::kBool));
        auto torch_types =
                torch::tensor(atomNumbers_, torch::TensorOptions().dtype(torch::kInt32)).to(this->device_);

        auto system = torch::make_intrusive<metatomic_torch::SystemHolder>(
                torch_types, torch_positions, torch_cell, torch_pbc);

        // Compute and add neighbor lists
        for (const auto& request : nl_requests_)
        {
            auto neighbors = computeNeighbors(
                    request, n_atoms, coerced_positions.data()->as_vec(), box_, /*periodic?*/ true);
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
        auto grad   = system->positions().grad();
        forceTensor = -grad.to(torch::kCPU).to(dtype_);
    }

    // 3. Distribute forces from main rank to all other ranks
    if (mpiComm_.isParallel())
    {
        mpiComm_.sumReduce(3 * n_atoms, static_cast<double*>(forceTensor.data_ptr()));
    }

    // 4. Each rank accumulates forces for its local atoms
    auto forceAccessor = forceTensor.accessor<double, 2>();
    for (int i = 0; i < n_atoms; ++i)
    {
        const int localIndex = idxLookup_[i];
        if (localIndex != -1) // This atom is local to the current rank
        {
            for (int m = 0; m < DIM; ++m)
            {
                outputs->forceWithVirial_.force_[localIndex][m] += forceAccessor[i][m];
            }
        }
    }
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
    bool periodic_array[3]   = { periodic, periodic, periodic };

    VesinNeighborList* vesin_neighbor_list = new VesinNeighborList();

    VesinDevice cpu{ VesinCPU, 0 };
    const char* error_message = nullptr;
    int         status        = vesin_neighbors(reinterpret_cast<const double (*)[3]>(positions),
                                 static_cast<size_t>(n_atoms),
                                 reinterpret_cast<const double (*)[3]>(box),
                                 periodic_array,
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
                                         torch::TensorOptions().dtype(torch::kFloat64));

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
            pair_vectors.to(torch::kFloat32).to(this->device_),
            neighbor_samples,
            std::vector<metatensor_torch::Labels>{ neighbor_component },
            neighbor_properties);
}


} // namespace gmx
