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
 * \brief CUDA kernel converting nbnxm GpuPairlist to metatensor neighbor list format.
 *
 * Mirrors appendPlainPairlistGpu() (pairlistset.cpp) but runs entirely on device.
 * One thread block per i-super-cluster; each thread handles one i-atom within
 * the super-cluster and iterates over all j-clusters.
 *
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "metatomic_gpu_pairlist_kernel.cuh"

#include "gromacs/nbnxm/nbnxm_enums.h"
#include "gromacs/nbnxm/pairlist.h"

namespace gmx
{

//! Number of i-atoms per super-cluster (= c_numClusterPerCell * c_clSize)
static constexpr int c_superClusterSize =
        sc_gpuNumClusterPerBin(sc_layoutType) * detail::c_nbnxnGpuClusterSize;

//! Number of i-clusters per super-cluster
static constexpr int c_numClusterPerCell = sc_gpuNumClusterPerBin(sc_layoutType);

//! Cluster size
static constexpr int c_clSize = detail::c_nbnxnGpuClusterSize;

//! Number of j-clusters per packed j-group
static constexpr int c_jGroupSize = sc_gpuJgroupSize(sc_layoutType);

//! Split cluster size (for exclusion indexing)
static constexpr int c_splitClusterSize =
        detail::c_nbnxnGpuClusterSize / sc_gpuClusterPairSplit(sc_layoutType);

//! Interaction mask for all i-clusters within a super-cluster
static constexpr unsigned int c_superClInteractionMask = ((1U << c_numClusterPerCell) - 1U);

//! Shift grid dimensions (from ishift.h: c_dBoxX=2, c_dBoxY=1, c_dBoxZ=1)
static constexpr int c_nBoxX = 2 * c_dBoxX + 1;
static constexpr int c_nBoxY = 2 * c_dBoxY + 1;

/*! \brief Convert GROMACS shift index to metatensor cell shifts (negated).
 *
 * GROMACS convention: d_ij = x[i] + shift_vec - x[j]
 * Metatensor convention: r_ij = x[j] + cell_shift*box - x[i]
 * Therefore: metatensor cell_shift = -(GROMACS cell_shift)
 */
__device__ __forceinline__ void shiftIndexToNegCellShift(int shiftIndex,
                                                         int& sa,
                                                         int& sb,
                                                         int& sc)
{
    const int divZ = shiftIndex / (c_nBoxY * c_nBoxX);
    const int rem  = shiftIndex - divZ * (c_nBoxY * c_nBoxX);
    const int divY = rem / c_nBoxX;
    const int xPart = rem - divY * c_nBoxX;

    // GROMACS shift = (xPart - c_dBoxX, divY - c_dBoxY, divZ - c_dBoxZ)
    // Metatensor = negation
    sa = -(xPart - c_dBoxX);
    sb = -(divY - c_dBoxY);
    sc = -(divZ - c_dBoxZ);
}

/*! \brief CUDA kernel: convert nbnxm GpuPairlist → metatensor NL format.
 *
 * Each thread block handles one i-super-cluster. Thread index maps to i-atom
 * within the super-cluster (64 threads = 8 clusters × 8 atoms).
 * For each i-atom, iterates over all j-packed groups and j-atoms, checks
 * interaction mask + distance filter, and emits valid pairs to output buffers.
 *
 * Outputs ALL pairs within cutoff (including classically excluded pairs),
 * since ML models need the full neighbor list.
 *
 * \param[in]  d_sci          i-super-cluster entries [numSci]
 * \param[in]  d_cjPacked     Packed j-cluster groups
 * \param[in]  d_excl         Exclusion masks (unused — we emit all pairs)
 * \param[in]  d_atomIndex    nbnxm→natural atom index mapping
 * \param[in]  d_xq           Atom positions+charge in nbnxm order [float4: x,y,z,q]
 * \param[in]  d_shiftVec     Shift vectors [float3, c_numShiftVectors entries]
 * \param[out] d_samples      Output: pair samples [capacity, 5] as (i, j, sa, sb, sc)
 * \param[out] d_distances    Output: distance vectors [capacity, 3] as (dx, dy, dz)
 * \param[out] d_count        Atomic counter for number of pairs written
 * \param[in]  capacity       Max pairs before overflow
 * \param[in]  cutoffSq       Squared cutoff distance
 * \param[in]  numSci         Number of super-clusters
 */
__global__ void convertPairlistToMetatomicKernel(const nbnxn_sci_t*       d_sci,
                                                  const nbnxn_cj_packed_t* d_cjPacked,
                                                  const int*               d_atomIndex,
                                                  const float4*            d_xq,
                                                  const float3*            d_shiftVec,
                                                  int32_t*                 d_samples,
                                                  float*                   d_distances,
                                                  int64_t*                 d_count,
                                                  int64_t                  capacity,
                                                  float                    cutoffSq,
                                                  int                      numSci)
{
    const int sciIdx = blockIdx.x;
    if (sciIdx >= numSci)
    {
        return;
    }

    const nbnxn_sci_t nbSci = d_sci[sciIdx];
    const int         shift  = nbSci.shift;
    const float3      sv     = d_shiftVec[shift];

    // Thread index → i-atom within super-cluster
    // tid ∈ [0, c_superClusterSize)  where c_superClusterSize = 64
    const int tid          = threadIdx.x;
    const int iClusterIdx  = tid / c_clSize;     // which i-cluster (0..7)
    const int iInCluster   = tid % c_clSize;     // which atom within i-cluster (0..7)

    // Global nbnxm index for this i-atom
    const int iAtomNbnxm = nbSci.sci * c_superClusterSize + tid;
    const int atomI      = d_atomIndex[iAtomNbnxm];

    // Skip filler atoms
    if (atomI < 0)
    {
        return;
    }

    // Load i-atom position (shifted)
    const float4 xqI = d_xq[iAtomNbnxm];
    const float  xiS = xqI.x + sv.x;
    const float  yiS = xqI.y + sv.y;
    const float  ziS = xqI.z + sv.z;

    // Precompute cell shift for this super-cluster
    int cellShiftA, cellShiftB, cellShiftC;
    shiftIndexToNegCellShift(shift, cellShiftA, cellShiftB, cellShiftC);

    // Central shift index for self-pair check
    constexpr int centralShift = c_dBoxX + c_dBoxY * c_nBoxX + c_dBoxZ * c_nBoxY * c_nBoxX;

    // Iterate over packed j-cluster groups
    for (int jPackIdx = nbSci.cjPackedBegin; jPackIdx < nbSci.cjPackedEnd; jPackIdx++)
    {
        const nbnxn_cj_packed_t jPack = d_cjPacked[jPackIdx];

        for (int jInPack = 0; jInPack < c_jGroupSize; jInPack++)
        {
            // Check if this j-cluster interacts with any i-cluster at all
            if ((jPack.imei[0].imask & (c_superClInteractionMask << (jInPack * c_numClusterPerCell))) == 0)
            {
                continue;
            }

            // Check if this specific (i-cluster, j-cluster) pair interacts
            const unsigned int clusterPairMask = 1U << (jInPack * c_numClusterPerCell + iClusterIdx);
            if ((jPack.imei[0].imask & clusterPairMask) == 0)
            {
                continue;
            }

            const int jCluster = jPack.cj[jInPack];

            // Iterate over j-atoms in this cluster
            for (int j = 0; j < c_clSize; j++)
            {
                const int jAtomNbnxm = jCluster * c_clSize + j;
                const int atomJ      = d_atomIndex[jAtomNbnxm];

                if (atomJ < 0)
                {
                    continue; // filler
                }

                // Self-pair exclusion: same atom in central image
                if (shift == centralShift && atomI == atomJ)
                {
                    continue;
                }

                // Compute distance (metatensor convention: r_ij = x[j] - x[i] - shift_vec)
                const float4 xqJ = d_xq[jAtomNbnxm];
                const float  dx  = xqJ.x - xiS;
                const float  dy  = xqJ.y - yiS;
                const float  dz  = xqJ.z - ziS;
                const float  r2  = dx * dx + dy * dy + dz * dz;

                if (r2 < cutoffSq)
                {
                    const int64_t idx = atomicAdd(reinterpret_cast<unsigned long long*>(d_count),
                                                  static_cast<unsigned long long>(1));
                    if (idx < capacity)
                    {
                        d_samples[idx * 5 + 0] = atomI;
                        d_samples[idx * 5 + 1] = atomJ;
                        d_samples[idx * 5 + 2] = cellShiftA;
                        d_samples[idx * 5 + 3] = cellShiftB;
                        d_samples[idx * 5 + 4] = cellShiftC;
                        d_distances[idx * 3 + 0] = dx;
                        d_distances[idx * 3 + 1] = dy;
                        d_distances[idx * 3 + 2] = dz;
                    }
                }
            }
        }
    }
}


void launchConvertPairlistToMetatomicKernel(const nbnxn_sci_t*       d_sci,
                                             const nbnxn_cj_packed_t* d_cjPacked,
                                             const int*               d_atomIndex,
                                             const float4*            d_xq,
                                             const float3*            d_shiftVec,
                                             int32_t*                 d_samples,
                                             float*                   d_distances,
                                             int64_t*                 d_count,
                                             int64_t                  capacity,
                                             float                    cutoffSq,
                                             int                      numSci,
                                             const DeviceStream&      deviceStream)
{
    if (numSci == 0)
    {
        return;
    }

    const int blockSize = c_superClusterSize; // 64 threads
    const int numBlocks = numSci;

    convertPairlistToMetatomicKernel<<<numBlocks, blockSize, 0, deviceStream.stream()>>>(
            d_sci, d_cjPacked, d_atomIndex, d_xq, d_shiftVec,
            d_samples, d_distances, d_count, capacity, cutoffSq, numSci);
}

} // namespace gmx
