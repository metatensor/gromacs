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

#include "gromacs/domdec/localatomset.h"
#include "gromacs/mdlib/broadcaststructs.h"
#include "gromacs/mdrunutility/mdmodulesnotifiers.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/mpicomm.h"
#include "gromacs/utility/stringutil.h"

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

static std::optional<ptrdiff_t> indexOf(ArrayRef<const int32_t> vec, const int32_t val)
{
    auto it = std::find(vec.begin(), vec.end(), val);
    if (it == vec.end())
    {
        return std::nullopt;
    }
    return std::distance(vec.begin(), it);
}

static torch::Tensor preparePbcType(PbcType* pbcType, torch::Device device)
{
    torch::Tensor pbcTensor =
            torch::tensor({ true, true, true }, torch::TensorOptions().dtype(torch::kBool));
    if (*pbcType == PbcType::XY)
    {
        pbcTensor[2] = false;
    }
    else if (*pbcType != PbcType::Xyz)
    {
        GMX_THROW(InconsistentInputError("PBC type not supported."));
    }
    return pbcTensor.to(device);
}

static metatensor_torch::TensorBlock buildNeighborListFromPairlist(ArrayRef<const int32_t> pairlist,
                                                                   ArrayRef<const RVec> shiftVectors,
                                                                   ArrayRef<const RVec> cellShifts,
                                                                   ArrayRef<const RVec> positions,
                                                                   torch::Device        device,
                                                                   torch::ScalarType    dtype)
{
    const int64_t n_pairs = static_cast<int64_t>(pairlist.size() / 2);

    auto pair_samples_values = torch::zeros({ n_pairs, 5 }, torch::TensorOptions().dtype(torch::kInt32));
    auto pair_samples_ptr = pair_samples_values.accessor<int32_t, 2>();

    // Full interatomic vectors (rj - ri + shift), not just shifts.
    auto pair_vectors = torch::zeros({ n_pairs, 3, 1 }, torch::TensorOptions().dtype(torch::kFloat64));

    auto vectors_cpu = torch::zeros({ n_pairs, 3, 1 }, torch::TensorOptions().dtype(torch::kFloat64));
    auto vectors_accessor = vectors_cpu.accessor<double, 3>();

    for (int64_t i = 0; i < n_pairs; i++)
    {
        int32_t atom_i = pairlist[2 * i];
        int32_t atom_j = pairlist[2 * i + 1];

        pair_samples_ptr[i][0] = static_cast<int32_t>(atom_i);
        pair_samples_ptr[i][1] = static_cast<int32_t>(atom_j);
        pair_samples_ptr[i][2] = cellShifts[i][0];
        pair_samples_ptr[i][3] = cellShifts[i][1];
        pair_samples_ptr[i][4] = cellShifts[i][2];

        // Calculate r_ij = r_j - r_i + shift
        double r_ij_x = positions[atom_j][0] - positions[atom_i][0] + shiftVectors[i][0];
        double r_ij_y = positions[atom_j][1] - positions[atom_i][1] + shiftVectors[i][1];
        double r_ij_z = positions[atom_j][2] - positions[atom_i][2] + shiftVectors[i][2];

        vectors_accessor[i][0][0] = r_ij_x;
        vectors_accessor[i][1][0] = r_ij_y;
        vectors_accessor[i][2][0] = r_ij_z;
    }

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
            vectors_cpu.to(dtype).to(device),
            neighbor_samples,
            std::vector<metatensor_torch::Labels>{ neighbor_component },
            neighbor_properties);
}


struct MetatomicData
{
    metatensor_torch::Module           model = metatensor_torch::Module(torch::jit::Module());
    metatomic_torch::ModelCapabilities capabilities;
    std::vector<metatomic_torch::NeighborListOptions> nl_requests;
    metatomic_torch::ModelEvaluationOptions           evaluations_options;
    torch::ScalarType                                 dtype             = torch::kFloat32;
    bool                                              check_consistency = true;
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

    // Pairlist-based neighbor lists don't work with domain decomposition yet (indices are local)
    // Matches NNPot's limitation
    if (mpiComm_.isParallel())
    {
        GMX_THROW(NotImplementedError(
                "Metatomic does not yet support domain decomposition. "
                "Use thread-MPI (gmx mdrun) instead of MPI (mpirun gmx_mpi mdrun)."));
    }

    // Only main rank loads model
    if (mpiComm_.isMainRank())
    {
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

        // Determine device using capabilities and optional environment variable
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

        auto requests_ivalue = data_->model.run_method("requested_neighbor_lists");
        for (const auto& request_ivalue : requests_ivalue.toList())
        {
            data_->nl_requests.push_back(
                    request_ivalue.get().toCustomClass<metatomic_torch::NeighborListOptionsHolder>());
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
            GMX_THROW(APIError("Unsupported dtype: " + data_->capabilities->dtype()));
        }

        data_->evaluations_options =
                torch::make_intrusive<metatomic_torch::ModelEvaluationOptionsHolder>();
        data_->evaluations_options->set_length_unit("nm");

        auto outputs = data_->capabilities->outputs();
        if (!outputs.contains("energy"))
        {
            GMX_THROW(APIError("Metatomic model must provide 'energy' output."));
        }

        auto requested_output      = torch::make_intrusive<metatomic_torch::ModelOutputHolder>();
        requested_output->per_atom = false;
        requested_output->explicit_gradients = {};

        data_->evaluations_options->outputs.insert("energy", requested_output);
        data_->check_consistency = options_.params_.checkConsistency;
    }

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

void MetatomicForceProvider::gatherAtomNumbersIndices(const MDModulesAtomsRedistributedSignal& signal)
{
    const auto&   mtaIndices = options_.params_.mtaIndices_;
    const int32_t numInput   = static_cast<int32_t>(mtaIndices.size());

    inputToLocalIndex_.assign(numInput, -1);
    inputToGlobalIndex_.assign(numInput, -1);
    atomNumbers_.assign(numInput, 0);

    if (mpiComm_.isParallel())
    {
        GMX_RELEASE_ASSERT(signal.globalAtomIndices_.has_value(),
                           "Global atom indices required for domain decomposition.");
        auto          globalAtomIndices = signal.globalAtomIndices_.value();
        const int32_t numLocal          = signal.x_.size();

        for (int32_t i = 0; i < static_cast<int32_t>(globalAtomIndices.size()); i++)
        {
            int32_t globalIdx = globalAtomIndices[i];
            for (int32_t j = 0; j < numInput; j++)
            {
                if (options_.params_.mtaAtoms_->globalIndex()[j] == globalIdx)
                {
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
        mpiComm_.sumReduce(numInput, atomNumbers_.data());
    }
    else
    {
        const auto* mtaAtoms = options_.params_.mtaAtoms_.get();
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
    fullPairlist_.assign(signal.excludedPairlist_.begin(), signal.excludedPairlist_.end());
    doPairlist_ = true;
}

void MetatomicForceProvider::preparePairlistInput()
{
    if (!doPairlist_)
    {
        return;
    }

    GMX_ASSERT(!fullPairlist_.empty(), "Pairlist empty!");

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

        auto inputIdxA = indexOf(inputToGlobalIndex_, atomPair.first);
        auto inputIdxB = indexOf(inputToGlobalIndex_, atomPair.second);

        if (inputIdxA.has_value() && inputIdxB.has_value())
        {
            RVec       shift;
            const IVec unitShift = shiftIndexToXYZ(shiftIndex);
            mvmul_ur0(box_, unitShift.toRVec(), shift);

            pairlistForModel_.push_back(static_cast<int32_t>(inputIdxA.value()));
            pairlistForModel_.push_back(static_cast<int32_t>(inputIdxB.value()));
            shiftVectors_.push_back(shift);
            cellShifts_.push_back(unitShift);
        }
    }

    GMX_RELEASE_ASSERT(pairlistForModel_.size() == shiftVectors_.size() * 2,
                       "Pairlist/shift size mismatch.");
    doPairlist_ = false;
}

void MetatomicForceProvider::calculateForces(const ForceProviderInput& inputs, ForceProviderOutput* outputs)
{
    const int32_t n_atoms = static_cast<int32_t>(options_.params_.mtaIndices_.size());

    gatherAtomPositions(inputs.x_);
    copy_mat(inputs.box_, box_);
    preparePairlistInput();

    // Force tensor - main rank fills, others have zeros
    torch::Tensor forceTensor =
            torch::zeros({ n_atoms, 3 }, torch::TensorOptions().dtype(torch::kFloat64));

    // Virial tensor for pressure/stress calculations
    torch::Tensor virialTensor = torch::zeros({ 3, 3 }, torch::TensorOptions().dtype(torch::kFloat64));

    if (mpiComm_.isMainRank())
    {
        auto gromacs_scalar_type = torch::kFloat32;
        if (std::is_same_v<real, double>)
        {
            gromacs_scalar_type = torch::kFloat64;
        }
        auto blob_options = torch::TensorOptions().dtype(gromacs_scalar_type).device(torch::kCPU);

        auto torch_positions = torch::from_blob(positions_.data()->as_vec(), { n_atoms, 3 }, blob_options)
                                       .to(data_->dtype)
                                       .to(data_->device)
                                       .set_requires_grad(true);

        auto torch_cell =
                torch::from_blob(&box_, { 3, 3 }, blob_options).to(data_->dtype).to(data_->device);

        // Create strain tensor for virial computation (like LAMMPS does)
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
            // TODO: take from the user / model
            metatomic_torch::register_autograd_neighbors(system, neighbors, /*check_consistency*/ true);
            system->add_neighbor_list(request, neighbors);
        }

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
            GMX_THROW(APIError("[Metatomic] Model evaluation failed: " + std::string(e.what())));
        }

        auto energy_block  = metatensor_torch::TensorMapHolder::block_by_id(output_map, 0);
        auto energy_tensor = energy_block->values();

        outputs->enerd_.term[InteractionFunction::MetatomicPotentialEnergy] =
                static_cast<real>(energy_tensor.item<double>());

        // Reset gradients before backward
        torch_positions.mutable_grad() = torch::Tensor();
        strain.mutable_grad()          = torch::Tensor();

        // Compute forces and virial via backward propagation
        energy_tensor.backward(-torch::ones_like(energy_tensor));

        auto grad   = torch_positions.grad();
        forceTensor = grad.to(torch::kCPU).to(torch::kFloat64);

        // Get virial from strain gradient
        virialTensor = strain.grad().to(torch::kCPU).to(torch::kFloat64);
    }

    // Distribute forces (sumReduce acts as broadcast since non-main ranks have zeros)
    if (mpiComm_.isParallel())
    {
        mpiComm_.sumReduce(n_atoms * 3, static_cast<double*>(forceTensor.data_ptr()));
        mpiComm_.sumReduce(9, static_cast<double*>(virialTensor.data_ptr()));
    }

    // Apply forces to local atoms only
    auto forceAccessor = forceTensor.accessor<double, 2>();
    for (int32_t i = 0; i < n_atoms; ++i)
    {
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
