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
 * and about rlist wide, so the force provider exchanges its own: each rank
 * describes where its home atoms are with a LatticeRegion, and sends the
 * home atoms that may be near the region of another rank to that rank. The
 * receiver keeps the atoms within range (atomsWithinRange) and finds the pairs
 * (findPairs).
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

//! Coordinates of a point: fractional along the periodic box vectors, Cartesian along the others.
using LatticePoint = std::array<double, 3>;

/*! \brief Converts positions to lattice coordinates.
 *
 * A displacement of length r changes lattice coordinate d by at most r * perLength(d):
 * the inverse of the distance between the lattice planes for a periodic direction,
 * 1 for a Cartesian one.
 */
class LatticeCoordinates
{
public:
    LatticeCoordinates(const matrix box, const std::array<uint8_t, 3>& periodic);

    //! Lattice coordinates of \p x.
    LatticePoint operator()(const RVec& x) const;
    //! Largest change of coordinate \p d per unit of length.
    double perLength(int d) const { return perLength_[d]; }

private:
    //! Inverse of the box with Cartesian unit vectors along the non-periodic directions.
    std::array<std::array<double, 3>, 3> inverse_;
    std::array<double, 3>                perLength_;
};

/*! \brief The interval spanned by a set of points along each lattice coordinate.
 *
 * Positions are not put in the box, so the interval of an atom near a periodic
 * boundary stays contiguous. A region without points has lo > hi.
 */
struct LatticeRegion
{
    LatticePoint lo = { 1, 1, 1 };
    LatticePoint hi = { 0, 0, 0 };

    //! Whether the region holds no point.
    bool empty() const { return lo[0] > hi[0]; }
};

//! The region spanned by \p points.
LatticeRegion latticeRegion(ArrayRef<const LatticePoint> points);

/*! \brief Whether \p point may be within \p range of a point of \p region.
 *
 * Conservative: true for every point within range, and maybe for some others.
 * Periodic directions wrap, so a point near the other side of the box is near.
 */
bool isNearRegion(const LatticeRegion&          region,
                  const LatticePoint&           point,
                  const LatticeCoordinates&     lattice,
                  const std::array<uint8_t, 3>& periodic,
                  double                        range);

/*! \brief Whether a point of \p b may be within \p range of a point of \p a.
 *
 * Conservative, like isNearRegion(); false when either region is empty.
 */
bool regionsAreNear(const LatticeRegion&          a,
                    const LatticeRegion&          b,
                    const LatticeCoordinates&     lattice,
                    const std::array<uint8_t, 3>& periodic,
                    double                        range);

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
