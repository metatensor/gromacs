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
 * Declares the GPU-resident Metatomic Force Provider.
 *
 * This is a pure-ML GPU path that bypasses the IForceProvider interface.
 * Coordinates are read directly from StatePropagatorDataGpu (zero-copy via
 * torch::from_blob), neighbor lists are built on-device from the nbnxm
 * GpuPairlist, and forces are written to a GPU DeviceBuffer registered with
 * GpuForceReduction.  No GPU-CPU copies in the hot path (single-rank).
 *
 * Domain decomposition is supported via the "pairlist" mode: each pair is
 * assigned to exactly one rank by the nbnxm DD decomposition, and halo
 * forces are redistributed via sparse MPI exchange (same pattern as the
 * CPU MetatomicForceProvider).
 *
 * \ingroup module_applied_forces
 */
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "gromacs/gpu_utils/devicebuffer_datatype.h"
#include "gromacs/utility/vectypes.h"

class DeviceContext;
class DeviceStream;
class GpuEventSynchronizer;
struct gmx_enerdata_t;

namespace gmx
{
class GpuPairlist;
class MpiComm;
struct NBAtomDataGpu;
struct MetatomicData;
struct MetatomicParameters;
class ForceWithVirial;
class MDLogger;

/*! \brief GPU-resident metatomic force provider (pure-ML, no ML/MM).
 *
 * Computes ALL forces from the metatomic model on GPU.
 * Follows the PME GPU integration pattern: provides a DeviceBuffer<RVec>
 * and a GpuEventSynchronizer for registration with GpuForceReduction.
 *
 * ## Domain decomposition
 *
 * When DD is active, forces are copied back to the CPU for sparse MPI
 * exchange (home forces applied directly to ForceWithVirial, halo forces
 * redistributed to their home ranks). Energy and virial are always
 * extracted to the CPU for proper global statistics.
 */
class MetatomicGpuForceProvider
{
public:
    /*! \brief Construct GPU metatomic force provider.
     *
     * \param[in] params               Model parameters (path, device, variant)
     * \param[in] logger               Logger for status messages
     * \param[in] deviceContext         GPU device context
     * \param[in] deviceStream          GPU stream for kernels
     * \param[in] mpiComm              MPI communicator (serial or parallel)
     * \param[in] globalAtomicNumbers  Atomic number for each global atom index
     */
    MetatomicGpuForceProvider(const MetatomicParameters& params,
                              const MDLogger&            logger,
                              const DeviceContext&       deviceContext,
                              const DeviceStream&        deviceStream,
                              const MpiComm&             mpiComm,
                              std::vector<int>           globalAtomicNumbers);
    ~MetatomicGpuForceProvider();

    /*! \brief Evaluate model on GPU. Called each step from do_force().
     *
     * 1. Wraps GPU coordinates as torch tensor (zero-copy)
     * 2. Builds NL on GPU from nbnxm GpuPairlist (NS steps only)
     * 3. Runs model forward + backward pass
     * 4. Single-rank: copies forces to forceBuffer_ (GpuForceReduction)
     *    DD: copies forces to host buffer for later distribution
     * 5. Extracts energy and virial to host
     * 6. Records completion event (single-rank only)
     *
     * \param[in] d_x           Device coordinates in natural order
     * \param[in] numAtoms      Number of atoms (home only in DD, total in serial)
     * \param[in] box           Simulation box
     * \param[in] step          Current MD step
     * \param[in] gpuPairlist   nbnxm GPU pairlist (local)
     * \param[in] d_atomIndex   nbnxm->natural atom index mapping
     * \param[in] nbAtomData    nbnxm GPU atom data (xq, shiftVec for NL kernel)
     * \param[in] isNsStep      Whether this is a neighbor-search step
     * \param[in] xReadyOnDevice Event signaling coordinates are ready on GPU (consumed here)
     */
    void calculateForces(DeviceBuffer<RVec>    d_x,
                         int                   numAtoms,
                         const matrix          box,
                         int64_t               step,
                         const GpuPairlist*    gpuPairlist,
                         DeviceBuffer<int>     d_atomIndex,
                         const NBAtomDataGpu*  nbAtomData,
                         bool                  isNsStep,
                         GpuEventSynchronizer* xReadyOnDevice);

    /*! \brief Update atom types and home/halo mapping after DD redistribution.
     *
     * Must be called before calculateForces() on each NS step.
     * For serial runs, called once with identity mapping.
     *
     * \param[in] numHomeAtoms        Number of home atoms on this rank
     * \param[in] numTotalAtoms       Total local atoms (home + halo)
     * \param[in] globalAtomIndices   Global atom index for each local atom [numTotalAtoms]
     *                                Use nullptr for serial (identity mapping, numHome == numTotal)
     */
    void updateAtomMapping(int numHomeAtoms, int numTotalAtoms, const int* globalAtomIndices);

    /*! \brief Apply cached energy, virial, and (DD) forces to output structures.
     *
     * Must be called after calculateForces(), once ForceWithVirial is available.
     * For single-rank: writes energy and virial only (forces via GpuForceReduction).
     * For DD: also distributes forces (home -> ForceWithVirial, halo -> sparse MPI exchange).
     *
     * \param[in,out] enerd            Energy data (metatomic term written)
     * \param[in,out] forceWithVirial  Force buffer for virial contribution + DD force application
     */
    void applyOutputs(gmx_enerdata_t* enerd, ForceWithVirial* forceWithVirial);

    //! GPU force buffer in natural order — register with GpuForceReduction.
    DeviceBuffer<RVec> getForceDeviceBuffer();

    //! Sync event marking force computation complete.
    GpuEventSynchronizer* getCompletionEvent();

    //! Whether domain decomposition is active (affects force output path).
    bool hasDomainDecomposition() const;

private:
    //! Build NL on GPU from nbnxm GpuPairlist (called on NS steps).
    void buildNeighborListGpu(const GpuPairlist*   gpuPairlist,
                              DeviceBuffer<int>    d_atomIndex,
                              const NBAtomDataGpu* nbAtomData,
                              int                  numAtoms,
                              const matrix         box);

    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace gmx
