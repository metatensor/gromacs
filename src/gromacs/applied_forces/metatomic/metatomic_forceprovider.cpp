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

#include <cstdint>
#include <unordered_map>

#include "gromacs/domdec/localatomset.h"
#include "gromacs/mdlib/broadcaststructs.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/mpicomm.h"

#ifdef DIM
#    undef DIM
#endif
#include <vesin.h>

#include <metatensor/torch.hpp>
#include <metatomic/torch.hpp>


static torch::Tensor preparePbcType(PbcType* pbcType)
{
    torch::Tensor pbcTensor =
            torch::tensor({ true, true, true }, torch::TensorOptions().dtype(torch::kBool));
    if (*pbcType == PbcType::XY)
    {
        pbcTensor[2] = false;
    }
    else if (*pbcType != PbcType::Xyz)
    {
        GMX_THROW(gmx::InconsistentInputError(
                "Option use_pbc was set to true, but PBC type is not supported."));
    }
    return pbcTensor;
}

static metatensor_torch::TensorBlock computeNeighbors(metatomic_torch::NeighborListOptions request,
                                                      long                                 n_atoms,
                                                      const float*      positions,
                                                      const matrix      box,
                                                      bool              periodic,
                                                      torch::Device     device,
                                                      torch::ScalarType dtype)
{
    auto cutoff = request->engine_cutoff("nm");

    VesinOptions options;
    options.cutoff           = cutoff;
    options.full             = request->full_list();
    options.return_shifts    = true;
    options.return_distances = false;
    options.return_vectors   = true;

    VesinNeighborList* vesin_neighbor_list = new VesinNeighborList();

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
        GMX_THROW(gmx::APIError(err_str));
    }

    auto n_pairs = static_cast<int64_t>(vesin_neighbor_list->length);

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

    auto deleter = [=](void*)
    {
        vesin_free(vesin_neighbor_list);
        delete vesin_neighbor_list;
    };

    auto pair_vectors = torch::from_blob(vesin_neighbor_list->vectors,
                                         { n_pairs, 3, 1 },
                                         deleter,
                                         torch::TensorOptions().dtype(torch::kFloat64));
    pair_vectors.to(dtype);

    auto neighbor_samples = torch::make_intrusive<metatensor_torch::LabelsHolder>(
            std::vector<std::string>{
                    "first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c" },
            pair_samples_values.to(device));

    auto neighbor_component = torch::make_intrusive<metatensor_torch::LabelsHolder>(
            std::vector<std::string>{ "xyz" },
            torch::tensor({ 0, 1, 2 }, torch::TensorOptions().dtype(torch::kInt32).device(device))
                    .reshape({ 3, 1 }));

    auto neighbor_properties = torch::make_intrusive<metatensor_torch::LabelsHolder>(
            std::vector<std::string>{ "distance" },
            torch::zeros({ 1, 1 }, torch::TensorOptions().dtype(torch::kInt32).device(device)));

    return torch::make_intrusive<metatensor_torch::TensorBlockHolder>(
            pair_vectors.to(dtype).to(device),
            neighbor_samples,
            std::vector<metatensor_torch::Labels>{ neighbor_component },
            neighbor_properties);
}

namespace gmx
{

struct MetatomicData
{
    metatensor_torch::Module           model = metatensor_torch::Module(torch::jit::Module());
    metatomic_torch::ModelCapabilities capabilities;
    std::vector<metatomic_torch::NeighborListOptions> nl_requests;
    metatomic_torch::ModelEvaluationOptions           evaluations_options;
    torch::ScalarType                                 dtype;
    bool                                              check_consistency;
    torch::Device                                     device = torch::kCPU;
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
    // ALL ranks load the model, not just the main rank
    // This enables each rank to compute forces for its local atoms independently
    
    GMX_LOG(logger_.info).asParagraph().appendText("Initializing MetatomicForceProvider on all ranks...");

    // Load the model on EVERY rank
    try
    {
        torch::optional<std::string> extensions_directory = torch::nullopt;
        if (!options_.params_.extensionsDirectory.empty())
        {
            extensions_directory = options_.params_.extensionsDirectory;
        }

        this->data_->model = metatomic_torch::load_atomistic_model(options_.params_.modelPath_,
                                                                   extensions_directory);
    }
    catch (const std::exception& e)
    {
        GMX_THROW(APIError("Failed to load metatomic model: " + std::string(e.what())));
    }

    // Query model capabilities on all ranks
    data_->capabilities =
            data_->model.run_method("capabilities").toCustomClass<metatomic_torch::ModelCapabilitiesHolder>();
    auto requests_ivalue = data_->model.run_method("requested_neighbor_lists");
    for (const auto& request_ivalue : requests_ivalue.toList())
    {
        data_->nl_requests.push_back(
                request_ivalue.get().toCustomClass<metatomic_torch::NeighborListOptionsHolder>());
    }

    // Determine device - each rank picks its own device
    torch::optional<std::string> desired;
    if (const char* env = std::getenv("GMX_METATOMIC_DEVICE")) {
        GMX_LOG(logger_.info)
            .asParagraph()
            .appendText("Using device from GMX_METATOMIC_DEVICE environment variable: ")
            .appendText(env);
        desired = std::string(env);
    } else {
        desired = options_.params_.device;
    }

    c10::DeviceType device_type_ =
            metatomic_torch::pick_device(data_->capabilities->supported_devices, desired);
    data_->device = torch::Device(device_type_);

    data_->model.to(data_->device);

    // Set data type on all ranks
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
        GMX_THROW(APIError("Unsupported dtype from model: " + data_->capabilities->dtype()));
    }

    // Set up evaluation options on all ranks
    data_->evaluations_options =
            torch::make_intrusive<metatomic_torch::ModelEvaluationOptionsHolder>();
    data_->evaluations_options->set_length_unit("nm");

    auto outputs = data_->capabilities->outputs();
    if (!outputs.contains("energy"))
    {
        GMX_THROW(APIError("Metatomic model must provide an 'energy' output."));
    }

    auto requested_output = torch::make_intrusive<metatomic_torch::ModelOutputHolder>();
    requested_output->per_atom = true;  // KEY: Request per-atom energies for proper DD
    requested_output->explicit_gradients = {};

    data_->evaluations_options->outputs.insert("energy", requested_output);

    data_->check_consistency = options_.params_.checkConsistency;

    // Initialize lookup tables - will be populated on first DD
    const auto& mtaIndices = options_.params_.mtaIndices_;
    const int   n_atoms    = static_cast<int>(mtaIndices.size());
    idxLookup_.resize(n_atoms, -1);
    atomNumbers_.resize(n_atoms, 0);

    GMX_LOG(logger_.info)
            .asParagraph()
            .appendText("MetatomicForceProvider initialization complete on all ranks.");
}

void MetatomicForceProvider::gatherAtomNumbersIndices()
{
    // This function updates the lookup tables after domain decomposition
    // Each rank knows which ML atoms are local to it
    
    const auto& mtaIndices = options_.params_.mtaIndices_;
    const int   n_atoms    = static_cast<int>(mtaIndices.size());

    // Reset lookup table
    std::fill(idxLookup_.begin(), idxLookup_.end(), -1);

    // Build reverse lookup: global index -> ML group index
    std::unordered_map<int, int> globalToMlIndex;
    globalToMlIndex.reserve(n_atoms);
    for (int i = 0; i < n_atoms; ++i)
    {
        globalToMlIndex[mtaIndices[i]] = i;
    }

    // Populate lookup for this rank's local atoms
    const auto* mtaAtoms = options_.params_.mtaAtoms_.get();
    for (size_t i = 0; i < mtaAtoms->numAtomsLocal(); ++i)
    {
        const int lIdx = mtaAtoms->localIndex()[i];
        const int gIdx = mtaAtoms->globalIndex()[mtaAtoms->collectiveIndex()[i]];

        if (auto it = globalToMlIndex.find(gIdx); it != globalToMlIndex.end())
        {
            const int mlIdx   = it->second;
            idxLookup_[mlIdx] = lIdx;
            atomNumbers_[mlIdx] = options_.params_.atoms_.atom[gIdx].atomnumber;
        }
    }

    // For parallel runs, we need all ranks to know ALL atom numbers for the system tensor
    // But we only compute forces for local atoms
    // XXX: seems buggy in parallel, need more checks
    if (mpiComm_.isParallel())
    {
        // Each rank has partial atomNumbers_, sum to get complete list
        // This is needed because the System object needs all atom types
        mpiComm_.sumReduce(gmx::ArrayRef<int>(atomNumbers_.data(), atomNumbers_.data() + n_atoms));
    }
}

MetatomicForceProvider::~MetatomicForceProvider() = default;

void MetatomicForceProvider::gatherAtomPositions(ArrayRef<const RVec> globalPositions)
{
    const int n_atoms = static_cast<int>(options_.params_.mtaIndices_.size());
    positions_.assign(n_atoms, RVec{ 0.0, 0.0, 0.0 });

    // Each rank fills its local atoms' positions
    for (int i = 0; i < n_atoms; ++i)
    {
        if (idxLookup_[i] != -1)
        {
            positions_[i] = globalPositions[idxLookup_[i]];
        }
    }

    // Sum-reduce positions so all ranks have complete position array
    // This is needed because neighbor list computation needs all positions
    if (mpiComm_.isParallel())
    {
        real*        data_ptr = reinterpret_cast<real*>(positions_.data());
        const size_t n_reals  = static_cast<size_t>(n_atoms) * 3;
        mpiComm_.sumReduce(gmx::ArrayRef<real>(data_ptr, data_ptr + n_reals));
    }
}

void MetatomicForceProvider::calculateForces(const ForceProviderInput& inputs, ForceProviderOutput* outputs)
{
    const int n_atoms = static_cast<int>(options_.params_.mtaIndices_.size());

    // Gather positions (all ranks need this for neighbor list computation)
    this->gatherAtomPositions(inputs.x_);
    copy_mat(inputs.box_, box_);

    auto gromacs_scalar_type = torch::kFloat32;
    if (std::is_same_v<real, double>)
    {
        gromacs_scalar_type = torch::kFloat64;
    }
    auto blob_options = torch::TensorOptions().dtype(gromacs_scalar_type).device(torch::kCPU);

    // ALL ranks run the model, each computing forces for its local atoms
    auto coerced_positions = makeArrayRef(positions_);

    auto torch_positions =
            torch::from_blob(coerced_positions.data()->as_vec(), { n_atoms, 3 }, blob_options)
                    .to(data_->dtype)
                    .to(data_->device)
                    .set_requires_grad(true);

    auto torch_cell =
            torch::from_blob(&box_, { 3, 3 }, blob_options).to(data_->dtype).to(data_->device);

    auto torch_pbc = preparePbcType(options_.params_.pbcType_.get()).to(data_->device);
    auto torch_types =
            torch::tensor(atomNumbers_, torch::TensorOptions().dtype(torch::kInt32)).to(data_->device);

    auto system = torch::make_intrusive<metatomic_torch::SystemHolder>(
            torch_types, torch_positions, torch_cell, torch_pbc);

    bool periodic = torch::all(torch_pbc).item<bool>();

    // Compute neighbor lists on each rank
    for (const auto& request : data_->nl_requests)
    {
        auto neighbors = computeNeighbors(
                request, n_atoms, coerced_positions.data()->as_vec(), box_, periodic, data_->device, data_->dtype);
        metatomic_torch::register_autograd_neighbors(system, neighbors, false);
        system->add_neighbor_list(request, neighbors);
    }

    // Run the model on EVERY rank
    metatensor_torch::TensorMap output_map;
    try
    {
        std::vector<metatomic_torch::System> systems;
        systems.push_back(system);

        auto ivalue_output = data_->model.forward(
                { c10::IValue(systems), data_->evaluations_options, data_->check_consistency });
        auto dict_output = ivalue_output.toGenericDict();
        output_map = dict_output.at("energy").toCustomClass<metatensor_torch::TensorMapHolder>();
    }
    catch (const std::exception& e)
    {
        GMX_THROW(APIError("[MetatomicPotential] Model evaluation failed: " + std::string(e.what())));
    }

    // Extract per-atom energies
    auto energy_block  = metatensor_torch::TensorMapHolder::block_by_id(output_map, 0);
    auto energy_tensor = energy_block->values();

    // Handle energy for domain decomposition
    // If per_atom=true, we get per-atom energies [n_atoms, 1]
    // So sum over local atoms' energies
    double local_energy = 0.0;
    
    if (energy_tensor.dim() == 2 && energy_tensor.size(0) == n_atoms)
    {
        // Per-atom energies - sum only local atoms
        auto energy_cpu = energy_tensor.to(torch::kCPU).to(torch::kFloat64);
        auto energy_accessor = energy_cpu.accessor<double, 2>();
        
        for (int i = 0; i < n_atoms; ++i)
        {
            if (idxLookup_[i] != -1)  // Only count local atoms
            {
                local_energy += energy_accessor[i][0];
            }
        }
    }
    else
    {
        // Scalar energy - only main rank contributes (or divide among ranks)
        if (mpiComm_.isMainRank())
        {
            local_energy = energy_tensor.item<double>();
        }
    }

    // Reduce energy across all ranks
    double total_energy = local_energy;
    if (mpiComm_.isParallel())
    {
        mpiComm_.sumReduce(gmx::ArrayRef<double>(&total_energy, &total_energy + 1));
    }
    
    // Only main rank sets the energy (GROMACS handles the rest)
    if (mpiComm_.isMainRank())
    {
        outputs->enerd_.term[InteractionFunction::MetatomicPotentialEnergy] =
                static_cast<float>(total_energy);
    }

    // Compute gradients - all ranks do this
    energy_tensor.sum().backward();
    auto grad = system->positions().grad();
    auto forceTensor = -grad.to(torch::kCPU).to(data_->dtype);

    // Scatter forces to local atoms ONLY - no broadcast needed!
    if (data_->dtype == torch::kFloat64)
    {
        auto accessor = forceTensor.accessor<double, 2>();
        for (int i = 0; i < n_atoms; ++i)
        {
            const int localIndex = idxLookup_[i];
            if (localIndex != -1)  // Only update local atoms
            {
                outputs->forceWithVirial_.force_[localIndex][0] += accessor[i][0];
                outputs->forceWithVirial_.force_[localIndex][1] += accessor[i][1];
                outputs->forceWithVirial_.force_[localIndex][2] += accessor[i][2];
            }
        }
    }
    else if (data_->dtype == torch::kFloat32)
    {
        auto accessor = forceTensor.accessor<float, 2>();
        for (int i = 0; i < n_atoms; ++i)
        {
            const int localIndex = idxLookup_[i];
            if (localIndex != -1)  // Only update local atoms
            {
                outputs->forceWithVirial_.force_[localIndex][0] += static_cast<double>(accessor[i][0]);
                outputs->forceWithVirial_.force_[localIndex][1] += static_cast<double>(accessor[i][1]);
                outputs->forceWithVirial_.force_[localIndex][2] += static_cast<double>(accessor[i][2]);
            }
        }
    }
    // Note: Virial still needs proper implementation
}

} // namespace gmx
