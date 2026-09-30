/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2026- The GROMACS Authors
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
 * derived work must be properly licensed. Other licenses are available.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */
/*! \internal \file
 * \brief
 * Implements the correction for the classical Coulomb interaction between
 * embedded atoms, see embedded_coulomb_correction.h.
 *
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "embedded_coulomb_correction.h"

#include <cmath>

#include <algorithm>
#include <vector>

#include "gromacs/math/units.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/mdtypes/group.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/utility/gmxomp.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/mpicomm.h"
#include "gromacs/utility/stringutil.h"

namespace gmx
{

EmbeddedCoulombCorrection computeEmbeddedCoulombCorrection(ArrayRef<const RVec>       x,
                                                           ArrayRef<const real>       q,
                                                           const t_pbc&               pbc,
                                                           const interaction_const_t& ic,
                                                           ArrayRef<const int>        energyGroups,
                                                           int                 numEnergyGroups,
                                                           ArrayRef<const int> sites,
                                                           int                 numThreads)
{
    const auto& coulomb = ic.coulomb;
    const bool  ewald   = usingPmeOrEwald(coulomb.type);
    if (!ewald && !usingRF(coulomb.type) && coulomb.type != CoulombInteractionType::Cut)
    {
        GMX_THROW(NotImplementedError(
                formatString("metatomic-oniom supports Ewald, PME, reaction-field and cut-off "
                             "electrostatics, not %s.",
                             enumValueToString(coulomb.type))));
    }

    const int n = x.ssize();
    GMX_RELEASE_ASSERT(q.ssize() == n, "Need one charge per embedded atom");
    GMX_RELEASE_ASSERT(energyGroups.empty() || energyGroups.ssize() == n,
                       "Need one energy group per embedded atom");
    GMX_RELEASE_ASSERT(sites.empty() || sites.ssize() == n, "Need one site per embedded atom");

    EmbeddedCoulombCorrection result;
    result.forces.assign(n, { 0, 0, 0 });
    result.groupPairEnergies.assign(numEnergyGroups * numEnergyGroups, 0.0);

    const double rc2               = coulomb.cutoff * coulomb.cutoff;
    const double beta              = coulomb.ewaldCoeff;
    const double krf               = coulomb.reactionFieldCoefficient;
    const double crf               = coulomb.reactionFieldShift;
    const double twoBetaOverSqrtPi = 2 * beta / std::sqrt(M_PI);
    const auto   group = [&](int i) { return energyGroups.empty() ? 0 : energyGroups[i]; };

    // For an ML region of thousands of atoms this is millions of pairs per step: the pairs
    // are split over the OpenMP threads, each with its own forces, virial and energies
    const int                              numThreadsUsed = std::max(1, std::min(numThreads, n));
    std::vector<EmbeddedCoulombCorrection> perThread(numThreadsUsed);
#pragma omp parallel num_threads(numThreadsUsed)
    {
      try
      {
        auto& local = perThread[gmx_omp_get_thread_num()];
        local.forces.assign(n, { 0, 0, 0 });
        local.groupPairEnergies.assign(numEnergyGroups * numEnergyGroups, 0.0);
        // Rows get shorter with i; interleave them over the threads
#pragma omp for schedule(dynamic, 16)
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                if (!sites.empty() && sites[i] != sites[j])
                {
                    continue;
                }
                const double qq = coulomb.epsfac * q[i] * q[j];
                if (qq == 0)
                {
                    continue;
                }
                RVec dxr;
                pbc_dx_aiuc(&pbc, x[i], x[j], dxr);
                const DVec   dx(dxr[XX], dxr[YY], dxr[ZZ]);
                const double r2 = dx.norm2();
                // Ewald: the kernels remove the reciprocal-space pair within the
                // cut-off, we remove it beyond. Reaction-field: the kernels add
                // k_rf r^2 - c_rf within the cut-off, we take it out again.
                if (ewald == (r2 < rc2))
                {
                    continue;
                }
                // v is the pair energy, fscal = -(dv/dr) / r
                double v, fscal;
                if (ewald)
                {
                    const double r   = std::sqrt(r2);
                    const double erf = std::erf(beta * r);
                    v                = -qq * erf / r;
                    fscal = -qq * (erf / r - twoBetaOverSqrtPi * std::exp(-beta * beta * r2)) / r2;
                }
                else
                {
                    v     = -qq * (krf * r2 - crf);
                    fscal = 2 * qq * krf;
                }
                const DVec f = fscal * dx;
                local.energy += v;
                local.groupPairEnergies[GID(group(i), group(j), numEnergyGroups)] += v;
                for (int d = 0; d < DIM; d++)
                {
                    local.forces[i][d] += f[d];
                    local.forces[j][d] -= f[d];
                    for (int e = 0; e < DIM; e++)
                    {
                        local.virial[d][e] -= 0.5 * dx[d] * f[e];
                    }
                }
                local.numPairs++;
            }
        }
      }
      GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR
    }
    for (const auto& local : perThread)
    {
        result.energy += local.energy;
        result.numPairs += local.numPairs;
        for (size_t g = 0; g < local.groupPairEnergies.size(); g++)
        {
            result.groupPairEnergies[g] += local.groupPairEnergies[g];
        }
        for (int i = 0; i < n; i++)
        {
            for (int d = 0; d < DIM; d++)
            {
                result.forces[i][d] += local.forces[i][d];
            }
        }
        for (int d = 0; d < DIM; d++)
        {
            for (int e = 0; e < DIM; e++)
            {
                result.virial[d][e] += local.virial[d][e];
            }
        }
    }
    return result;
}

EmbeddedCoulombCorrectionProvider::EmbeddedCoulombCorrectionProvider(const LocalAtomSet& atoms,
                                                                     std::vector<real>   charges,
                                                                     std::vector<int> energyGroups,
                                                                     int            numEnergyGroups,
                                                                     PbcType        pbcType,
                                                                     const MpiComm& mpiComm,
                                                                     std::vector<int> sites) :
    atoms_(atoms),
    charges_(std::move(charges)),
    energyGroups_(std::move(energyGroups)),
    numEnergyGroups_(numEnergyGroups),
    sites_(std::move(sites)),
    pbcType_(pbcType),
    mpiComm_(mpiComm),
    positions_(3 * charges_.size())
{
}

void EmbeddedCoulombCorrectionProvider::calculateForces(const ForceProviderInput& input,
                                                        ForceProviderOutput*      output)
{
    GMX_RELEASE_ASSERT(input.interactionConst_ != nullptr,
                       "The embedded Coulomb correction needs the interaction constants");

    std::fill(positions_.begin(), positions_.end(), 0.0);
    const auto local      = atoms_.localIndex();
    const auto collective = atoms_.collectiveIndex();
    for (size_t k = 0; k < atoms_.numAtomsLocal(); k++)
    {
        for (int d = 0; d < DIM; d++)
        {
            positions_[3 * collective[k] + d] = input.x_[local[k]][d];
        }
    }
    if (mpiComm_.isParallel())
    {
        mpiComm_.sumReduce(positions_);
    }
    std::vector<RVec> x(charges_.size());
    for (size_t i = 0; i < x.size(); i++)
    {
        x[i] = { real(positions_[3 * i]), real(positions_[3 * i + 1]), real(positions_[3 * i + 2]) };
    }

    t_pbc pbc;
    set_pbc(&pbc, pbcType_, input.box_);
    const auto correction = computeEmbeddedCoulombCorrection(
            x, charges_, pbc, *input.interactionConst_, energyGroups_, numEnergyGroups_, sites_,
            gmx_omp_nthreads_get(ModuleMultiThread::Default));

    auto force = output->forceWithVirial_.force_;
    for (size_t k = 0; k < atoms_.numAtomsLocal(); k++)
    {
        force[local[k]] += correction.forces[collective[k]];
    }
    if (mpiComm_.isMainRank())
    {
        auto& coulombSR = output->enerd_.grpp.energyGroupPairTerms[NonBondedEnergyTerms::CoulombSR];
        for (size_t g = 0; g < correction.groupPairEnergies.size(); g++)
        {
            coulombSR[g] += correction.groupPairEnergies[g];
        }
        output->forceWithVirial_.addVirialContribution(correction.virial);
    }
}

} // namespace gmx
