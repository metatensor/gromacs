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
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at https://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */
/*! \internal \file
 * \brief
 * Environment and pairs of the home atoms of a rank, for domain decomposition.
 *
 * With domain decomposition, the model runs on each rank for its home atoms,
 * and needs every atom within its interaction range of them, in every
 * direction, and the pairs among these atoms. The halo of GROMACS is one-sided
 * and about rlist wide, so these functions find both from the positions.
 *
 * \author Eric Boittier <eric.boittier@epfl.ch>
 * \ingroup module_applied_forces
 */
#ifndef GMX_APPLIED_FORCES_METATOMIC_ENVIRONMENT_H
#define GMX_APPLIED_FORCES_METATOMIC_ENVIRONMENT_H

#include <cstdint>

#include <array>
#include <vector>

#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/vectypes.h"

struct t_pbc;

namespace gmx
{

/*! \brief Indices of the \p candidates within \p range of an atom of \p home.
 *
 * Distances are minimum-image distances under \p pbc (nullptr for none), so an
 * atom close to a home atom through any periodic image is found. A range of 0
 * finds nothing.
 */
std::vector<int32_t> atomsWithinRange(const t_pbc*         pbc,
                                      ArrayRef<const RVec> home,
                                      ArrayRef<const RVec> candidates,
                                      real                 range);

/*! \brief Each pair of \p x closer than \p cutoff, once, in metatensor convention.
 *
 * Pair k is (pairs[2k], pairs[2k+1]) with cell shift shifts[k], so that
 * x[pairs[2k+1]] + cellShiftVector(box, shifts[k]) - x[pairs[2k]] is the
 * minimum-image vector between them. Shifts are along the \p periodic box
 * vectors only. A cutoff of 0 finds nothing.
 */
void findPairs(const t_pbc*                  pbc,
               const matrix                  box,
               const std::array<uint8_t, 3>& periodic,
               ArrayRef<const RVec>          x,
               real                          cutoff,
               std::vector<int32_t>*         pairs,
               std::vector<IVec>*            shifts);

} // namespace gmx

#endif
