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
 * Implements the GPU-resident Metatomic Force Provider.
 *
 * Zero-copy coordinate wrapping, GPU neighbor list construction from nbnxm
 * GpuPairlist, and GPU-resident model evaluation with force output.
 *
 * Single-rank: forces via GpuForceReduction (pure GPU path, no D2H copies).
 * Domain decomposition: GPU model evaluation + CPU-side sparse force exchange.
 * GPU-resident (zero-copy) path is single-rank only; DD mode uses CPU force
 * distribution via distributeNonHomeForces.
 *
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "metatomic_gpu_forceprovider.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <string>
#include <unordered_map>
#include <vector>

#include "gromacs/gpu_utils/device_context.h"
#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/gpu_utils/gpueventsynchronizer.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/nbnxm/gpu_types_common.h"
#include "gromacs/nbnxm/nbnxm_enums.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/mpicomm.h"

#include "metatomic_gpu_pairlist_kernel.cuh"
#include "metatomic_options.h"

#ifdef DIM
#    undef DIM
#endif

#include <metatensor/torch.hpp>
#include <metatomic/torch.hpp>

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace gmx
{

/*! \brief Normalizes the variant string for GPU path (same logic as CPU path). */
static torch::optional<std::string> normalizeVariantGpu(std::string variant_string)
{
    if (variant_string == "no" || variant_string.empty())
    {
        return torch::nullopt;
    }
    return variant_string;
}

/*! \brief Pick the energy output key from model capabilities. */
static std::string pickOutputGpu(const std::string&                         baseName,
                                  const metatomic_torch::ModelCapabilities& caps,
                                  torch::optional<std::string>              variant)
{
    auto outputs = caps->outputs();
    if (variant.has_value())
    {
        std::string key = baseName + "::" + variant.value();
        if (outputs.contains(key))
        {
            return key;
        }
    }
    return baseName;
}

/*! \brief Implementation class hiding torch/CUDA/MPI dependencies. */
class MetatomicGpuForceProvider::Impl
{
public:
    Impl(const MetatomicParameters& params,
         const MDLogger&            logger,
         const DeviceContext&       deviceContext,
         const DeviceStream&        deviceStream,
         const MpiComm&             mpiComm,
         std::vector<int>           globalAtomicNumbers);
    ~Impl();

    void calculateForces(DeviceBuffer<RVec>    d_x,
                         int                   numAtoms,
                         const matrix          box,
                         int64_t               step,
                         const GpuPairlist*    gpuPairlist,
                         DeviceBuffer<int>     d_atomIndex,
                         const NBAtomDataGpu*  nbAtomData,
                         bool                  isNsStep,
                         GpuEventSynchronizer* xReadyOnDevice);

    void updateAtomMapping(int numHomeAtoms, int numTotalAtoms, const int* globalAtomIndices);

    void applyOutputs(gmx_enerdata_t* enerd, ForceWithVirial* forceWithVirial);

    DeviceBuffer<RVec>    getForceDeviceBuffer() { return forceBuffer_; }
    GpuEventSynchronizer* getCompletionEvent() { return completionEvent_.get(); }
    bool                  hasDomainDecomposition() const { return isDD_; }

private:
    void buildNeighborListGpu(const GpuPairlist*   gpuPairlist,
                              DeviceBuffer<int>    d_atomIndex,
                              const NBAtomDataGpu* nbAtomData,
                              int                  numAtoms,
                              const matrix         box);

    void distributeNonHomeForces(ForceWithVirial* forceWithVirial);

    const DeviceContext& deviceContext_;
    const DeviceStream&  deviceStream_;
    const MDLogger&      logger_;
    const MpiComm&       mpiComm_;

    //! GPU force output buffer [numAtoms] in natural order
    DeviceBuffer<RVec> forceBuffer_     = nullptr;
    int                forceBufferSize_ = 0;
    std::unique_ptr<GpuEventSynchronizer> completionEvent_;

    //! Metatomic model data
    metatensor_torch::Module                          model_ = metatensor_torch::Module(torch::jit::Module());
    metatomic_torch::ModelCapabilities                capabilities_;
    std::vector<metatomic_torch::NeighborListOptions> nlRequests_;
    metatomic_torch::ModelEvaluationOptions           evaluationOptions_;
    torch::ScalarType dtype_  = torch::kFloat32;
    torch::Device     device_ = torch::Device(torch::kCUDA, 0);

    //! Cached NL Labels
    metatensor_torch::Labels cachedNLComponent_;
    metatensor_torch::Labels cachedNLProperties_;
    std::vector<std::string> nlSampleNames_ = {
        "first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"
    };

    //! Cached atom types tensor on GPU (refreshed on atom redistribution)
    torch::Tensor atomTypes_;
    torch::Tensor pbc_;
    int32_t       numAtoms_ = 0;

    //! NL tensors on GPU (refreshed on NS steps from GpuPairlist)
    torch::Tensor nlSamples_;   //!< [nPairs, 5] int32 on CUDA
    torch::Tensor nlDistances_; //!< [nPairs, 3, 1] float on CUDA

    //! GPU buffers for NL kernel output (pre-allocated, resized as needed)
    DeviceBuffer<int32_t> d_nlSamples_   = nullptr;
    DeviceBuffer<float>   d_nlDistances_ = nullptr;
    DeviceBuffer<int64_t> d_nlCount_     = nullptr;
    int64_t               nlCapacity_    = 0;

    //! Model path for error messages
    std::string modelPath_;

    //! Debug mode flag
    bool debugEnabled_ = false;

    //! Global atomic numbers indexed by global atom index (set in constructor)
    std::vector<int> globalAtomicNumbers_;

    // ---- Domain decomposition state ----

    //! Whether DD is active
    bool isDD_ = false;

    //! Number of home atoms on this rank (DD: < numLocalMta_, serial: == numLocalMta_)
    int32_t numHomeMta_ = 0;

    //! Number of total local atoms (home + halo) for model input
    int32_t numLocalMta_ = 0;

    //! Model index -> GROMACS local buffer index (identity in serial)
    std::vector<int32_t> mtaToGmxLocal_;

    //! Model index -> global atom index (for sparse force exchange)
    std::vector<int32_t> mtaToGlobalIdx_;

    //! Global atom index -> local home model index (for receiving forces)
    std::unordered_map<int32_t, int32_t> globalIdxToLocalHome_;

    //! Whether atom mapping has been initialized
    bool atomMappingReady_ = false;

    // ---- Cached outputs (set by calculateForces, read by applyOutputs) ----

    double cachedEnergy_ = 0.0;
    matrix cachedVirial_ = { { 0 } };

    //! Host force buffer for DD mode [numLocalMta_ * 3] in double
    std::vector<double> cachedForcesHost_;
};


MetatomicGpuForceProvider::Impl::Impl(const MetatomicParameters& params,
                                       const MDLogger&            logger,
                                       const DeviceContext&       deviceContext,
                                       const DeviceStream&        deviceStream,
                                       const MpiComm&             mpiComm,
                                       std::vector<int>           globalAtomicNumbers) :
    deviceContext_(deviceContext),
    deviceStream_(deviceStream),
    logger_(logger),
    mpiComm_(mpiComm),
    completionEvent_(std::make_unique<GpuEventSynchronizer>()),
    modelPath_(params.modelPath_),
    globalAtomicNumbers_(std::move(globalAtomicNumbers))
{
    GMX_LOG(logger_.info).asParagraph().appendText("Initializing MetatomicGpuForceProvider...");

    debugEnabled_ = (std::getenv("GMX_METATOMIC_DEBUG") != nullptr);

    // Load model
    try
    {
        torch::optional<std::string> extensionsDir = torch::nullopt;
        if (!params.extensionsDirectory.empty())
        {
            extensionsDir = params.extensionsDirectory;
        }
        model_ = metatomic_torch::load_atomistic_model(params.modelPath_, extensionsDir);
    }
    catch (const std::exception& e)
    {
        GMX_THROW(APIError("Failed to load metatomic model for GPU: " + std::string(e.what())));
    }

    capabilities_ = model_.run_method("capabilities")
                             .toCustomClass<metatomic_torch::ModelCapabilitiesHolder>();

    // Determine CUDA device and verify it matches the GROMACS DeviceContext
    int cudaDevice = 0;
    cudaGetDevice(&cudaDevice);
    GMX_RELEASE_ASSERT(deviceContext_.deviceInfo().id == cudaDevice,
                       "GROMACS DeviceContext and CUDA current device disagree");
    device_ = torch::Device(torch::kCUDA, cudaDevice);

    // Set GROMACS CUDA stream as LibTorch's current stream to avoid inter-stream sync
    auto torchStream = c10::cuda::getStreamFromExternal(
            deviceStream_.stream(), static_cast<c10::DeviceIndex>(cudaDevice));
    c10::cuda::setCurrentCUDAStream(torchStream);

    GMX_LOG(logger_.info)
            .asParagraph()
            .appendTextFormatted("MetatomicGpu using CUDA device %d, MPI size %d",
                                 cudaDevice,
                                 mpiComm_.size());

    // Cache NL labels
    auto devIntOpts = torch::TensorOptions().dtype(torch::kInt32).device(device_);
    cachedNLComponent_ = torch::make_intrusive<metatensor_torch::LabelsHolder>(
            std::vector<std::string>{ "xyz" },
            torch::tensor({ 0, 1, 2 }, devIntOpts).reshape({ 3, 1 }));
    cachedNLProperties_ = torch::make_intrusive<metatensor_torch::LabelsHolder>(
            std::vector<std::string>{ "distance" },
            torch::zeros({ 1, 1 }, devIntOpts));

    // Get NL requests from model
    auto requestsIvalue = model_.run_method("requested_neighbor_lists");
    for (const auto& reqIvalue : requestsIvalue.toList())
    {
        nlRequests_.push_back(
                reqIvalue.get().toCustomClass<metatomic_torch::NeighborListOptionsHolder>());
    }

    model_.to(device_);

    // Determine model dtype
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
        GMX_THROW(APIError("Unsupported dtype: " + capabilities_->dtype()));
    }

    // Set up evaluation options
    evaluationOptions_ = torch::make_intrusive<metatomic_torch::ModelEvaluationOptionsHolder>();
    evaluationOptions_->set_length_unit("nm");

    auto outputs   = capabilities_->outputs();
    auto variant   = normalizeVariantGpu(params.variant);
    auto energyKey = pickOutputGpu("energy", capabilities_, variant);

    if (!outputs.contains(energyKey))
    {
        GMX_THROW(
                APIError("The model does not provide '" + energyKey + "' output for GPU path."));
    }

    auto modelOutput     = outputs.at(energyKey);
    auto requestedOutput = torch::make_intrusive<metatomic_torch::ModelOutputHolder>();
    requestedOutput->per_atom           = modelOutput->per_atom;
    requestedOutput->explicit_gradients = {};
    requestedOutput->set_quantity("energy");
    requestedOutput->set_unit("kJ/mol");
    evaluationOptions_->outputs.insert(energyKey, requestedOutput);

    // PBC tensor (assume 3D periodic for GPU path)
    pbc_ = torch::tensor({ true, true, true },
                          torch::TensorOptions().dtype(torch::kBool).device(device_));

    // Allocate NL counter on GPU
    allocateDeviceBuffer(&d_nlCount_, 1, deviceContext_);

    GMX_LOG(logger_.info)
            .asParagraph()
            .appendText("MetatomicGpuForceProvider initialization complete.");
}

MetatomicGpuForceProvider::Impl::~Impl()
{
    if (forceBuffer_)
    {
        freeDeviceBuffer(&forceBuffer_);
    }
    if (d_nlSamples_)
    {
        freeDeviceBuffer(&d_nlSamples_);
    }
    if (d_nlDistances_)
    {
        freeDeviceBuffer(&d_nlDistances_);
    }
    if (d_nlCount_)
    {
        freeDeviceBuffer(&d_nlCount_);
    }
}


void MetatomicGpuForceProvider::Impl::updateAtomMapping(int        numHomeAtoms,
                                                         int        numTotalAtoms,
                                                         const int* globalAtomIndices)
{
    mtaToGmxLocal_.clear();
    mtaToGlobalIdx_.clear();
    globalIdxToLocalHome_.clear();

    if (globalAtomIndices != nullptr && mpiComm_.isParallel())
    {
        // DD mode: map local atoms (home + halo) to global indices
        isDD_ = true;

        // Home atoms: local indices [0, numHomeAtoms)
        numHomeMta_ = 0;
        for (int i = 0; i < numHomeAtoms; i++)
        {
            int globalIdx = globalAtomIndices[i];
            if (globalIdx >= 0) // skip filler particles
            {
                mtaToGmxLocal_.push_back(i);
                mtaToGlobalIdx_.push_back(globalIdx);
                numHomeMta_++;
            }
        }

        // Halo atoms: local indices [numHomeAtoms, numTotalAtoms)
        for (int i = numHomeAtoms; i < numTotalAtoms; i++)
        {
            int globalIdx = globalAtomIndices[i];
            if (globalIdx >= 0)
            {
                mtaToGmxLocal_.push_back(i);
                mtaToGlobalIdx_.push_back(globalIdx);
            }
        }

        numLocalMta_ = static_cast<int32_t>(mtaToGmxLocal_.size());

        // Reverse map: global atom index -> local home model index
        for (int32_t i = 0; i < numHomeMta_; i++)
        {
            globalIdxToLocalHome_[mtaToGlobalIdx_[i]] = i;
        }

        // Build atom types for all local atoms
        std::vector<int32_t> atomNumbers(numLocalMta_);
        for (int32_t i = 0; i < numLocalMta_; i++)
        {
            int globalIdx = mtaToGlobalIdx_[i];
            GMX_RELEASE_ASSERT(globalIdx >= 0
                                       && globalIdx < static_cast<int>(globalAtomicNumbers_.size()),
                               "Global atom index out of range in updateAtomMapping");
            atomNumbers[i] = globalAtomicNumbers_[globalIdx];
        }
        atomTypes_ =
                torch::tensor(atomNumbers, torch::TensorOptions().dtype(torch::kInt32)).to(device_);
        numAtoms_ = numLocalMta_;

        if (debugEnabled_)
        {
            GMX_LOG(logger_.info)
                    .asParagraph()
                    .appendTextFormatted(
                            "MetatomicGpu DD mapping: rank=%d, home=%d, halo=%d, total=%d",
                            mpiComm_.rank(),
                            numHomeMta_,
                            numLocalMta_ - numHomeMta_,
                            numLocalMta_);
        }
    }
    else
    {
        // Serial / single-rank: identity mapping, all atoms are home
        isDD_        = false;
        numHomeMta_  = numTotalAtoms;
        numLocalMta_ = numTotalAtoms;
        numAtoms_    = numTotalAtoms;

        std::vector<int32_t> atomNumbers(numTotalAtoms);
        for (int i = 0; i < numTotalAtoms; i++)
        {
            atomNumbers[i] = globalAtomicNumbers_[i];
        }
        atomTypes_ =
                torch::tensor(atomNumbers, torch::TensorOptions().dtype(torch::kInt32)).to(device_);
    }

    atomMappingReady_ = true;
}


void MetatomicGpuForceProvider::Impl::buildNeighborListGpu(const GpuPairlist*   gpuPairlist,
                                                            DeviceBuffer<int>    d_atomIndex,
                                                            const NBAtomDataGpu* nbAtomData,
                                                            int                  numAtoms,
                                                            const matrix         box)
{
    if (!gpuPairlist || gpuPairlist->numSci <= 0)
    {
        return;
    }

    GMX_ASSERT(nbAtomData != nullptr, "NBAtomDataGpu must be valid for GPU NL construction");

    // Get cutoff from the first NL request
    float cutoff = 0.0f;
    if (!nlRequests_.empty())
    {
        cutoff = static_cast<float>(nlRequests_[0]->engine_cutoff("nm"));
    }
    const float cutoffSq = cutoff * cutoff;

    // Initial capacity estimate: ~20 pairs per atom (typical for ML potentials)
    if (nlCapacity_ == 0)
    {
        nlCapacity_ = static_cast<int64_t>(numAtoms) * 20;
        allocateDeviceBuffer(&d_nlSamples_, nlCapacity_ * 5, deviceContext_);
        allocateDeviceBuffer(&d_nlDistances_, nlCapacity_ * 3, deviceContext_);
    }

    // Reset counter to 0
    int64_t zero = 0;
    copyToDeviceBuffer(
            &d_nlCount_, &zero, 0, 1, deviceStream_, GpuApiCallBehavior::Async, nullptr);

    // Extract nbnxm atom data for the kernel
    const auto* d_xq       = reinterpret_cast<const float4*>(nbAtomData->xq);
    const auto* d_shiftVec = reinterpret_cast<const float3*>(nbAtomData->shiftVec);

    launchConvertPairlistToMetatomicKernel(gpuPairlist->sci,
                                           gpuPairlist->cjPacked,
                                           d_atomIndex,
                                           d_xq,
                                           d_shiftVec,
                                           d_nlSamples_,
                                           d_nlDistances_,
                                           d_nlCount_,
                                           nlCapacity_,
                                           cutoffSq,
                                           gpuPairlist->numSci,
                                           deviceStream_);

    // Check for overflow: copy counter back
    int64_t numPairs = 0;
    copyFromDeviceBuffer(
            &numPairs, &d_nlCount_, 0, 1, deviceStream_, GpuApiCallBehavior::Sync, nullptr);

    if (numPairs > nlCapacity_)
    {
        // Overflow: reallocate 2x and rerun
        freeDeviceBuffer(&d_nlSamples_);
        freeDeviceBuffer(&d_nlDistances_);
        nlCapacity_ = numPairs * 2;
        allocateDeviceBuffer(&d_nlSamples_, nlCapacity_ * 5, deviceContext_);
        allocateDeviceBuffer(&d_nlDistances_, nlCapacity_ * 3, deviceContext_);

        zero = 0;
        copyToDeviceBuffer(
                &d_nlCount_, &zero, 0, 1, deviceStream_, GpuApiCallBehavior::Async, nullptr);

        launchConvertPairlistToMetatomicKernel(gpuPairlist->sci,
                                               gpuPairlist->cjPacked,
                                               d_atomIndex,
                                               d_xq,
                                               d_shiftVec,
                                               d_nlSamples_,
                                               d_nlDistances_,
                                               d_nlCount_,
                                               nlCapacity_,
                                               cutoffSq,
                                               gpuPairlist->numSci,
                                               deviceStream_);

        copyFromDeviceBuffer(
                &numPairs, &d_nlCount_, 0, 1, deviceStream_, GpuApiCallBehavior::Sync, nullptr);
    }

    // Wrap GPU buffers as torch tensors (zero-copy)
    auto intOptions   = torch::TensorOptions().dtype(torch::kInt32).device(device_);
    auto floatOptions = torch::TensorOptions().dtype(torch::kFloat32).device(device_);

    auto halfSamples   = torch::from_blob(d_nlSamples_, { numPairs, 5 }, intOptions);
    auto halfDistances = torch::from_blob(d_nlDistances_, { numPairs, 3, 1 }, floatOptions);

    // The nbnxm GPU pairlist is approximately a half-list, but some pairs
    // may appear in both directions due to overlapping super-clusters.
    // ML models require a full list with both (i,j) and (j,i).
    // Strategy: create all reverses, concatenate, then deduplicate.
    auto revSamples = torch::empty_like(halfSamples);
    revSamples.index({ torch::indexing::Slice(), 0 }) =
            halfSamples.index({ torch::indexing::Slice(), 1 }); // j -> i
    revSamples.index({ torch::indexing::Slice(), 1 }) =
            halfSamples.index({ torch::indexing::Slice(), 0 }); // i -> j
    revSamples.index({ torch::indexing::Slice(), torch::indexing::Slice(2, 5) }) =
            -halfSamples.index({ torch::indexing::Slice(), torch::indexing::Slice(2, 5) });

    auto revDistances = -halfDistances;

    auto allSamples   = torch::cat({ halfSamples, revSamples }, /*dim=*/0);
    auto allDistances = torch::cat({ halfDistances, revDistances }, /*dim=*/0);

    // Deduplicate: encode each 5-column row as a single int64 key,
    // sort, then keep only the first occurrence of each key.
    {
        auto s64       = allSamples.to(torch::kInt64);
        constexpr int64_t D = 50; // shift offset (shifts are small integers)
        constexpr int64_t M = 100000; // multiplier (> max atom index + 2*D)
        auto col0 = s64.index({ torch::indexing::Slice(), 0 });
        auto col1 = s64.index({ torch::indexing::Slice(), 1 });
        auto col2 = s64.index({ torch::indexing::Slice(), 2 }) + D;
        auto col3 = s64.index({ torch::indexing::Slice(), 3 }) + D;
        auto col4 = s64.index({ torch::indexing::Slice(), 4 }) + D;
        auto keys = ((((col0 * M + col1) * M + col2) * M + col3) * M + col4);

        // Sort by key, then keep rows where key differs from predecessor
        auto [sortedKeys, sortIdx] = keys.sort();
        auto uniqueMask = torch::ones({ sortedKeys.size(0) },
                                       torch::TensorOptions().dtype(torch::kBool).device(device_));
        if (sortedKeys.size(0) > 1)
        {
            uniqueMask.index({ torch::indexing::Slice(1, torch::indexing::None) }) =
                    sortedKeys.index({ torch::indexing::Slice(1, torch::indexing::None) })
                    != sortedKeys.index({ torch::indexing::Slice(torch::indexing::None, -1) });
        }
        auto keepIdx = sortIdx.index({ uniqueMask });

        nlSamples_   = allSamples.index_select(0, keepIdx);
        nlDistances_ = allDistances.index_select(0, keepIdx);
    }

    if (dtype_ == torch::kFloat64)
    {
        nlDistances_ = nlDistances_.to(torch::kFloat64);
    }
}


void MetatomicGpuForceProvider::Impl::calculateForces(DeviceBuffer<RVec>    d_x,
                                                        int                   numAtoms,
                                                        const matrix          box,
                                                        int64_t               step,
                                                        const GpuPairlist*    gpuPairlist,
                                                        DeviceBuffer<int>     d_atomIndex,
                                                        const NBAtomDataGpu*  nbAtomData,
                                                        bool                  isNsStep,
                                                        GpuEventSynchronizer* xReadyOnDevice)
{
    GMX_RELEASE_ASSERT(atomMappingReady_, "updateAtomMapping must be called before calculateForces");

    // Wait for coordinates to be ready on the GPU.
    // The H2D copy runs on the StatePropagatorDataGpu stream; we insert
    // a barrier into the metatomic (NonBondedLocal) stream so that all
    // subsequent torch/CUDA work sees the updated coordinates.
    if (xReadyOnDevice != nullptr)
    {
        xReadyOnDevice->enqueueWaitEvent(deviceStream_);
    }

    // In DD mode, use the total local atom count (home + halo) for model input.
    // The caller passes mdatoms->homenr, but we need all local atoms.
    const int modelNumAtoms = isDD_ ? numLocalMta_ : numAtoms;

    // Ensure force buffer is large enough (for single-rank GPU path)
    if (!isDD_ && modelNumAtoms > forceBufferSize_)
    {
        if (forceBuffer_)
        {
            freeDeviceBuffer(&forceBuffer_);
        }
        allocateDeviceBuffer(&forceBuffer_, modelNumAtoms, deviceContext_);
        forceBufferSize_ = modelNumAtoms;
    }

    if (!isDD_)
    {
        clearDeviceBufferAsync(&forceBuffer_, 0, modelNumAtoms, deviceStream_);
    }

    // Build NL on NS steps
    if (isNsStep)
    {
        buildNeighborListGpu(gpuPairlist, d_atomIndex, nbAtomData, modelNumAtoms, box);
    }

    // Zero-copy wrap GPU coordinates as torch tensor
    auto coordOptions =
            torch::TensorOptions()
                    .dtype(std::is_same_v<real, double> ? torch::kFloat64 : torch::kFloat32)
                    .device(device_);
    auto positions = torch::from_blob(static_cast<void*>(d_x),
                                       { static_cast<int64_t>(modelNumAtoms), 3 },
                                       coordOptions)
                             .to(dtype_)
                             .set_requires_grad(true);

    // Build cell tensor from box (host -> device)
    auto cellHost = torch::from_blob(
            const_cast<real*>(&box[0][0]),
            { 3, 3 },
            torch::TensorOptions().dtype(
                    std::is_same_v<real, double> ? torch::kFloat64 : torch::kFloat32));
    auto cell = cellHost.to(device_, dtype_);

    // Strain tensor for virial computation
    auto strain = torch::eye(
            3, torch::TensorOptions().dtype(dtype_).device(device_).requires_grad(true));
    auto strainedCell      = torch::matmul(cell, strain);
    auto strainedPositions = torch::matmul(positions, strain);

    // Construct metatensor System (all tensors on GPU)
    auto system = torch::make_intrusive<metatomic_torch::SystemHolder>(
            atomTypes_, strainedPositions, strainedCell, pbc_);

    // Add neighbor lists
    for (const auto& request : nlRequests_)
    {
        if (nlSamples_.numel() == 0)
        {
            continue;
        }

        // Detach from any previous step's computational graph
        auto samples   = nlSamples_.detach();
        auto distances = nlDistances_.detach();

        auto neighborSamples = torch::make_intrusive<metatensor_torch::LabelsHolder>(
                nlSampleNames_, samples);
        auto neighbors = torch::make_intrusive<metatensor_torch::TensorBlockHolder>(
                distances,
                neighborSamples,
                std::vector<metatensor_torch::Labels>{ cachedNLComponent_ },
                cachedNLProperties_);

        metatomic_torch::register_autograd_neighbors(system, neighbors, false);
        system->add_neighbor_list(request, neighbors);
    }

    // Pairlist DD mode: all atoms contribute to energy (each pair on one rank).
    // Serial mode: same.
    evaluationOptions_->set_selected_atoms(torch::nullopt);

    // Debug: print diagnostics
    if (debugEnabled_)
    {
        auto posHost = positions.detach().to(torch::kCPU).to(torch::kFloat64);
        auto cellCpu = cell.to(torch::kCPU).to(torch::kFloat64);
        auto pA = posHost.accessor<double, 2>();
        auto cA = cellCpu.accessor<double, 2>();
        fprintf(stderr,
                "[MetatomicGpu] step=%ld atoms=%d NL_pairs=%ld\n"
                "  pos[0] = (%.6f, %.6f, %.6f)\n"
                "  pos[1] = (%.6f, %.6f, %.6f)\n"
                "  cell = diag(%.6f, %.6f, %.6f)\n",
                step,
                modelNumAtoms,
                nlSamples_.numel() > 0 ? nlSamples_.size(0) : 0L,
                pA[0][0], pA[0][1], pA[0][2],
                pA[1][0], pA[1][1], pA[1][2],
                cA[0][0], cA[1][1], cA[2][2]);
    }

    // In debug mode, checksum a sample of the raw coordinate buffer before
    // model evaluation to verify the assumption that d_x is stable (not mutated
    // by another stream) during the metatomic call.
    torch::Tensor coordChecksumPre;
    if (debugEnabled_)
    {
        auto rawCoords = torch::from_blob(static_cast<void*>(d_x),
                                           { static_cast<int64_t>(modelNumAtoms), 3 },
                                           coordOptions);
        coordChecksumPre = rawCoords.sum().detach().clone();
    }

    // Forward pass
    metatensor_torch::TensorMap outputMap;
    try
    {
        std::vector<metatomic_torch::System> systems;
        systems.push_back(system);

        auto ivalueOutput = model_.forward({ systems, evaluationOptions_, false });
        auto dictOutput   = ivalueOutput.toGenericDict();
        outputMap = dictOutput.at("energy").toCustomClass<metatensor_torch::TensorMapHolder>();
    }
    catch (const std::exception& e)
    {
        GMX_THROW(APIError("[MetatomicGpu] Model evaluation failed: " + std::string(e.what())));
    }

    auto energyBlock  = metatensor_torch::TensorMapHolder::block_by_id(outputMap, 0);
    auto energyTensor = energyBlock->values();

    // Extract energy (always needed for logging/output)
    cachedEnergy_ = energyTensor.sum().item<double>();

    if (debugEnabled_)
    {
        fprintf(stderr, "[MetatomicGpu] energy = %.6f (per_atom tensor shape: [%ld, %ld])\n",
                cachedEnergy_,
                energyTensor.size(0),
                energyTensor.size(1));
    }

    // Backward pass: compute forces via autograd
    positions.mutable_grad() = torch::Tensor();
    strain.mutable_grad()    = torch::Tensor();
    energyTensor.backward(-torch::ones_like(energyTensor));

    // Verify coordinate buffer was not mutated during model evaluation
    if (debugEnabled_ && coordChecksumPre.defined())
    {
        auto rawCoords = torch::from_blob(static_cast<void*>(d_x),
                                           { static_cast<int64_t>(modelNumAtoms), 3 },
                                           coordOptions);
        auto coordChecksumPost = rawCoords.sum();
        auto diff = (coordChecksumPost - coordChecksumPre).abs().item<float>();
        GMX_RELEASE_ASSERT(diff < 1e-6f,
                           "Coordinate buffer d_x was mutated during metatomic model evaluation. "
                           "This indicates a stream synchronization or aliasing bug.");
    }

    // Forces = position gradients (already negated by backward with -1)
    auto forces = positions.grad(); // [modelNumAtoms, 3] on CUDA

    // Extract virial from strain gradient (always needed)
    auto virialTensor = strain.grad().to(torch::kCPU).to(torch::kFloat64);
    auto virialAcc    = virialTensor.accessor<double, 2>();
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            cachedVirial_[i][j] = static_cast<real>(virialAcc[i][j]);
        }
    }

    if (isDD_)
    {
        // DD mode: copy forces to CPU for distribution via MPI.
        // Force tensor is on GPU; we need double precision on host.
        auto forcesHost = forces.to(torch::kCPU).to(torch::kFloat64).contiguous();
        cachedForcesHost_.resize(modelNumAtoms * 3);
        std::memcpy(cachedForcesHost_.data(),
                     forcesHost.data_ptr<double>(),
                     modelNumAtoms * 3 * sizeof(double));
    }
    else
    {
        // Single-rank: copy forces to GPU DeviceBuffer for GpuForceReduction
        auto forcesContig =
                forces.to(std::is_same_v<real, double> ? torch::kFloat64 : torch::kFloat32)
                        .contiguous();
        cudaMemcpyAsync(forceBuffer_,
                         forcesContig.data_ptr(),
                         modelNumAtoms * sizeof(RVec),
                         cudaMemcpyDeviceToDevice,
                         deviceStream_.stream());

        // Record completion event for GpuForceReduction
        completionEvent_->markEvent(deviceStream_);
    }
}


void MetatomicGpuForceProvider::Impl::distributeNonHomeForces(ForceWithVirial* forceWithVirial)
{
    GMX_RELEASE_ASSERT(isDD_, "distributeNonHomeForces called without DD");
    GMX_RELEASE_ASSERT(!cachedForcesHost_.empty(), "No cached forces for DD distribution");

    const int32_t numTotalGlobal = static_cast<int32_t>(globalAtomicNumbers_.size());
    const double* forceData      = cachedForcesHost_.data();

    // For small systems, dense allreduce has lower latency than sparse exchange
    constexpr int32_t sparseThreshold = 1000;

    if (numTotalGlobal < sparseThreshold)
    {
        // Dense fallback: allocate N_total buffer, scatter, allreduce, readback.
        std::vector<double> denseForces(3 * numTotalGlobal, 0.0);
        for (int32_t i = 0; i < numLocalMta_; i++)
        {
            int32_t g              = mtaToGlobalIdx_[i];
            denseForces[3 * g]     = forceData[3 * i];
            denseForces[3 * g + 1] = forceData[3 * i + 1];
            denseForces[3 * g + 2] = forceData[3 * i + 2];
        }
        mpiComm_.sumReduce(static_cast<std::size_t>(3 * numTotalGlobal), denseForces.data());

        for (int32_t i = 0; i < numHomeMta_; i++)
        {
            int32_t gmxIdx = mtaToGmxLocal_[i];
            int32_t g      = mtaToGlobalIdx_[i];
            forceWithVirial->force_[gmxIdx][0] += static_cast<real>(denseForces[3 * g]);
            forceWithVirial->force_[gmxIdx][1] += static_cast<real>(denseForces[3 * g + 1]);
            forceWithVirial->force_[gmxIdx][2] += static_cast<real>(denseForces[3 * g + 2]);
        }
        return;
    }

    // Sparse path: apply home forces directly, exchange only non-home forces.

    // Step 1: Apply home atom forces directly
    for (int32_t i = 0; i < numHomeMta_; i++)
    {
        int32_t gmxIdx = mtaToGmxLocal_[i];
        forceWithVirial->force_[gmxIdx][0] += static_cast<real>(forceData[3 * i]);
        forceWithVirial->force_[gmxIdx][1] += static_cast<real>(forceData[3 * i + 1]);
        forceWithVirial->force_[gmxIdx][2] += static_cast<real>(forceData[3 * i + 2]);
    }

    // Step 2: Pack non-home forces as sparse tuples (globalIdx, fx, fy, fz)
    const int32_t       numNonHome = numLocalMta_ - numHomeMta_;
    std::vector<double> sendBuf(4 * numNonHome);
    for (int32_t i = numHomeMta_; i < numLocalMta_; i++)
    {
        int32_t k      = i - numHomeMta_;
        sendBuf[4 * k] = static_cast<double>(mtaToGlobalIdx_[i]);
        sendBuf[4 * k + 1] = forceData[3 * i];
        sendBuf[4 * k + 2] = forceData[3 * i + 1];
        sendBuf[4 * k + 3] = forceData[3 * i + 2];
    }

    // Step 3: Exchange counts via allreduce
    const int          numRanks = mpiComm_.size();
    std::vector<int>   counts(numRanks, 0);
    counts[mpiComm_.rank()] = numNonHome;
    mpiComm_.sumReduce(ArrayRef<int>(counts));

    // Step 4: Allgatherv via Gatherv + Bcast (thread-MPI compatible)
    int              totalNonHome = 0;
    std::vector<int> displs(numRanks);
    for (int r = 0; r < numRanks; r++)
    {
        displs[r] = totalNonHome;
        totalNonHome += counts[r];
    }

    std::vector<int> dcounts(numRanks), ddispls(numRanks);
    for (int r = 0; r < numRanks; r++)
    {
        dcounts[r] = 4 * counts[r];
        ddispls[r] = 4 * displs[r];
    }

    std::vector<double> recvBuf(4 * totalNonHome);
    MPI_Gatherv(sendBuf.data(),
                4 * numNonHome,
                MPI_DOUBLE,
                recvBuf.data(),
                dcounts.data(),
                ddispls.data(),
                MPI_DOUBLE,
                mpiComm_.mainRank(),
                mpiComm_.comm());
    MPI_Bcast(recvBuf.data(), 4 * totalNonHome, MPI_DOUBLE, mpiComm_.mainRank(), mpiComm_.comm());

    // Step 5: Scan received tuples for forces destined for our home atoms
    for (int t = 0; t < totalNonHome; t++)
    {
        int32_t globalIdx = static_cast<int32_t>(recvBuf[4 * t]);
        auto    it        = globalIdxToLocalHome_.find(globalIdx);
        if (it != globalIdxToLocalHome_.end())
        {
            int32_t localHomeIdx = it->second;
            int32_t gmxIdx       = mtaToGmxLocal_[localHomeIdx];
            forceWithVirial->force_[gmxIdx][0] += static_cast<real>(recvBuf[4 * t + 1]);
            forceWithVirial->force_[gmxIdx][1] += static_cast<real>(recvBuf[4 * t + 2]);
            forceWithVirial->force_[gmxIdx][2] += static_cast<real>(recvBuf[4 * t + 3]);
        }
    }
}


void MetatomicGpuForceProvider::Impl::applyOutputs(gmx_enerdata_t*  enerd,
                                                     ForceWithVirial* forceWithVirial)
{
    // Energy: per-rank contribution (GROMACS global_stat sums across ranks)
    enerd->term[InteractionFunction::MetatomicPotentialEnergy] = static_cast<real>(cachedEnergy_);

    // Virial: per-rank contribution
    forceWithVirial->addVirialContribution(cachedVirial_);

    // DD: distribute forces via CPU
    if (isDD_)
    {
        distributeNonHomeForces(forceWithVirial);
    }
}


// ---- Public API forwarding to Impl ----

MetatomicGpuForceProvider::MetatomicGpuForceProvider(const MetatomicParameters& params,
                                                       const MDLogger&            logger,
                                                       const DeviceContext&       deviceContext,
                                                       const DeviceStream&        deviceStream,
                                                       const MpiComm&             mpiComm,
                                                       std::vector<int>           globalAtomicNumbers) :
    impl_(std::make_unique<Impl>(params,
                                  logger,
                                  deviceContext,
                                  deviceStream,
                                  mpiComm,
                                  std::move(globalAtomicNumbers)))
{
}

MetatomicGpuForceProvider::~MetatomicGpuForceProvider() = default;

void MetatomicGpuForceProvider::calculateForces(DeviceBuffer<RVec>    d_x,
                                                 int                   numAtoms,
                                                 const matrix          box,
                                                 int64_t               step,
                                                 const GpuPairlist*    gpuPairlist,
                                                 DeviceBuffer<int>     d_atomIndex,
                                                 const NBAtomDataGpu*  nbAtomData,
                                                 bool                  isNsStep,
                                                 GpuEventSynchronizer* xReadyOnDevice)
{
    impl_->calculateForces(d_x, numAtoms, box, step, gpuPairlist, d_atomIndex, nbAtomData, isNsStep, xReadyOnDevice);
}

void MetatomicGpuForceProvider::updateAtomMapping(int        numHomeAtoms,
                                                    int        numTotalAtoms,
                                                    const int* globalAtomIndices)
{
    impl_->updateAtomMapping(numHomeAtoms, numTotalAtoms, globalAtomIndices);
}

void MetatomicGpuForceProvider::applyOutputs(gmx_enerdata_t* enerd, ForceWithVirial* forceWithVirial)
{
    impl_->applyOutputs(enerd, forceWithVirial);
}

DeviceBuffer<RVec> MetatomicGpuForceProvider::getForceDeviceBuffer()
{
    return impl_->getForceDeviceBuffer();
}

GpuEventSynchronizer* MetatomicGpuForceProvider::getCompletionEvent()
{
    return impl_->getCompletionEvent();
}

bool MetatomicGpuForceProvider::hasDomainDecomposition() const
{
    return impl_->hasDomainDecomposition();
}

} // namespace gmx
