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
 * \brief Declaration for the CUDA kernel converting nbnxm GpuPairlist
 *        to metatensor neighbor list format.
 *
 * \ingroup module_applied_forces
 */
#ifndef GMX_APPLIED_FORCES_METATOMIC_GPU_PAIRLIST_KERNEL_CUH
#define GMX_APPLIED_FORCES_METATOMIC_GPU_PAIRLIST_KERNEL_CUH

#include <cstdint>

#include "gromacs/gpu_utils/device_stream.h"

struct nbnxn_sci_t;
struct nbnxn_cj_packed_t;

namespace gmx
{

/*! \brief Launch the CUDA kernel to convert nbnxm GPU pairlist to metatensor NL format.
 *
 * Iterates over the GpuPairlist's super-cluster entries on GPU, producing
 * metatensor-format neighbor list tensors (samples + distances) directly in
 * device memory.  All atom pairs within the cutoff are emitted, including
 * classically excluded pairs (ML models need the full NL).
 *
 * If the output counter exceeds \p capacity, the caller should reallocate
 * and relaunch.
 *
 * \param[in]  d_sci          i-super-cluster entries [numSci]
 * \param[in]  d_cjPacked     Packed j-cluster groups
 * \param[in]  d_atomIndex    nbnxm→natural atom index mapping
 * \param[in]  d_xq           Atom positions+charge in nbnxm order [float4]
 * \param[in]  d_shiftVec     Shift vectors [float3]
 * \param[out] d_samples      Output: pair samples [capacity, 5] (i, j, sa, sb, sc) int32
 * \param[out] d_distances    Output: distance vectors [capacity, 3] float
 * \param[out] d_count        Atomic counter for written pairs (int64)
 * \param[in]  capacity       Maximum number of pairs before overflow
 * \param[in]  cutoffSq       Squared cutoff distance
 * \param[in]  numSci         Number of i-super-clusters
 * \param[in]  deviceStream   CUDA stream to launch on
 */
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
                                             const DeviceStream&      deviceStream);

} // namespace gmx

#endif
