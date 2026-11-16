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
 * Removes the classical Coulomb interaction between embedded (Metatomic) atoms
 * at every distance.
 *
 * With metatomic-oniom, all embedded atoms are mutually excluded, which is how
 * E_MM(embedded) is subtracted. The non-bonded kernels only treat excluded
 * pairs within rcoulomb:
 *
 * - Ewald/PME: the reciprocal sum contains every pair, and the kernels remove
 *   it with -q_i q_j erf(beta r)/r for r < rcoulomb only. Pairs beyond
 *   rcoulomb keep their full MM interaction, and the energy jumps when a pair
 *   crosses the cut-off. The correction is -q_i q_j erf(beta r)/r for
 *   r >= rcoulomb.
 * - Reaction-field and plain cut-off: the kernels leave
 *   q_i q_j (k_rf r^2 - c_rf) on excluded pairs within rcoulomb, which jumps to
 *   zero at the cut-off. The correction cancels it for r < rcoulomb.
 *
 * In both cases the MM Coulomb energy between embedded atoms is zero at all
 * distances, apart from the constant reaction-field self term.
 *
 * \ingroup module_applied_forces
 */
#ifndef GMX_APPLIED_FORCES_METATOMIC_EMBEDDED_COULOMB_CORRECTION_H
#define GMX_APPLIED_FORCES_METATOMIC_EMBEDDED_COULOMB_CORRECTION_H

#include <vector>

#include "gromacs/domdec/localatomset.h"
#include "gromacs/mdtypes/iforceprovider.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/vectypes.h"

struct interaction_const_t;
struct t_pbc;
enum class PbcType : int;

namespace gmx
{

class MpiComm;

//! Energy, forces and virial of the embedded Coulomb correction.
struct EmbeddedCoulombCorrection
{
    //! Correction energy (kJ/mol).
    double energy = 0;
    //! Energy per energy-group pair, indexed with GID().
    std::vector<double> groupPairEnergies;
    //! Force on each embedded atom (kJ/mol/nm).
    std::vector<RVec> forces;
    //! Virial, -0.5 sum_pairs dx_ij (x) f_i.
    matrix virial = { { 0 } };
    //! Number of pairs that contributed.
    int numPairs = 0;
};

/*! \brief Correction for the classical Coulomb interaction between embedded atoms.
 *
 * Every pair of \p x is treated as excluded, with the minimum-image distance.
 * Only Ewald-family and reaction-field (including plain cut-off) electrostatics
 * are supported.
 *
 * \param[in] x       Positions of the embedded atoms.
 * \param[in] q       Their charges.
 * \param[in] pbc     Periodic boundary conditions.
 * \param[in] ic      Non-bonded interaction constants.
 * \param[in] energyGroups     Energy group of each atom, empty for a single group.
 * \param[in] numEnergyGroups  Number of energy groups.
 * \param[in] sites   Site of each atom, empty for a single site; only pairs within a site
 *                    are corrected, as the MM interaction between sites is kept.
 * \param[in] numThreads  OpenMP threads for the pair loop.
 */
EmbeddedCoulombCorrection computeEmbeddedCoulombCorrection(ArrayRef<const RVec>       x,
                                                           ArrayRef<const real>       q,
                                                           const t_pbc&               pbc,
                                                           const interaction_const_t& ic,
                                                           ArrayRef<const int> energyGroups    = {},
                                                           int                 numEnergyGroups = 1,
                                                           ArrayRef<const int> sites           = {},
                                                           int                 numThreads      = 1);

/*! \brief Force provider applying computeEmbeddedCoulombCorrection() to the embedded atoms.
 *
 * With domain decomposition, the embedded positions are summed over ranks,
 * each rank evaluates all pairs and applies forces to its home atoms, and the
 * main rank adds the energy and virial. The energy goes to the Coulomb (SR)
 * term, next to the in-cut-off exclusion corrections of the kernels.
 */
class EmbeddedCoulombCorrectionProvider final : public IForceProvider
{
public:
    /*! \param[in] atoms         Embedded atoms, collective index = embedded index.
     *  \param[in] charges       Charge of each embedded atom.
     *  \param[in] energyGroups  Energy group of each embedded atom.
     *  \param[in] numEnergyGroups  Number of energy groups.
     *  \param[in] pbcType       Periodic boundary conditions.
     *  \param[in] mpiComm       Communicator of the simulation.
     *  \param[in] sites         Site of each embedded atom, empty for a single site.
     */
    EmbeddedCoulombCorrectionProvider(const LocalAtomSet& atoms,
                                      std::vector<real>   charges,
                                      std::vector<int>    energyGroups,
                                      int                 numEnergyGroups,
                                      PbcType             pbcType,
                                      const MpiComm&      mpiComm,
                                      std::vector<int>    sites = {});

    void calculateForces(const ForceProviderInput& input, ForceProviderOutput* output) override;

private:
    LocalAtomSet      atoms_;
    std::vector<real> charges_;
    std::vector<int>  energyGroups_;
    int               numEnergyGroups_;
    std::vector<int>  sites_;
    PbcType           pbcType_;
    const MpiComm&    mpiComm_;
    //! Position of every embedded atom, summed over ranks.
    std::vector<double> positions_;
};

} // namespace gmx

#endif
