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
 * Implements the environment and pair search of the metatomic force provider.
 *
 * \author Eric Boittier <eric.boittier@epfl.ch>
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "metatomic_environment.h"

#include <cmath>

#include <algorithm>

#include "gromacs/math/boxmatrix.h"
#include "gromacs/selection/nbsearch.h"
#include "gromacs/utility/vec.h"

#include "metatomic_cellshift.h"

namespace gmx
{

LatticeCoordinates::LatticeCoordinates(const matrix box, const std::array<uint8_t, 3>& periodic)
{
    // Box matrices are lower triangular, so replacing a non-periodic vector by its
    // Cartesian unit vector keeps an invertible box matrix (pbc = xy has a and b in the
    // xy plane).
    matrix lattice;
    clear_mat(lattice);
    for (int d = 0; d < DIM; ++d)
    {
        if (periodic[d])
        {
            copy_rvec(box[d], lattice[d]);
        }
        else
        {
            lattice[d][d] = 1;
        }
    }
    matrix inverse;
    invertBoxMatrix(lattice, inverse);
    for (int d = 0; d < DIM; ++d)
    {
        double sum = 0;
        for (int k = 0; k < DIM; ++k)
        {
            inverse_[k][d] = inverse[k][d];
            sum += inverse_[k][d] * inverse_[k][d];
        }
        // the length of a reciprocal vector is the inverse of the distance between planes
        perLength_[d] = std::sqrt(sum);
    }
}

LatticePoint LatticeCoordinates::operator()(const RVec& x) const
{
    LatticePoint u = { 0, 0, 0 };
    for (int d = 0; d < DIM; ++d)
    {
        for (int k = 0; k < DIM; ++k)
        {
            u[d] += x[k] * inverse_[k][d];
        }
    }
    return u;
}

LatticeRegion latticeRegion(ArrayRef<const LatticePoint> points)
{
    LatticeRegion region;
    if (points.empty())
    {
        return region;
    }
    region.lo = points[0];
    region.hi = points[0];
    for (const auto& u : points)
    {
        for (int d = 0; d < DIM; ++d)
        {
            region.lo[d] = std::min(region.lo[d], u[d]);
            region.hi[d] = std::max(region.hi[d], u[d]);
        }
    }
    return region;
}

namespace
{

//! Whether \p u is in [lo, hi], modulo 1 along a periodic direction.
bool inInterval(double u, double lo, double hi, bool periodic)
{
    if (!periodic)
    {
        return lo <= u && u <= hi;
    }
    if (hi - lo >= 1)
    {
        return true;
    }
    return u - lo - std::floor(u - lo) <= hi - lo;
}

} // namespace

bool isNearRegion(const LatticeRegion&          region,
                  const LatticePoint&           point,
                  const LatticeCoordinates&     lattice,
                  const std::array<uint8_t, 3>& periodic,
                  double                        range)
{
    if (region.empty())
    {
        return false;
    }
    for (int d = 0; d < DIM; ++d)
    {
        const double margin = range * lattice.perLength(d);
        if (!inInterval(point[d], region.lo[d] - margin, region.hi[d] + margin, periodic[d] != 0))
        {
            return false;
        }
    }
    return true;
}

bool regionsAreNear(const LatticeRegion&          a,
                    const LatticeRegion&          b,
                    const LatticeCoordinates&     lattice,
                    const std::array<uint8_t, 3>& periodic,
                    double                        range)
{
    if (a.empty() || b.empty())
    {
        return false;
    }
    for (int d = 0; d < DIM; ++d)
    {
        // Two intervals overlap when the start of one is in the other, also on a circle.
        const double margin = range * lattice.perLength(d);
        const double lo     = a.lo[d] - margin;
        const double hi     = a.hi[d] + margin;
        const bool   wraps  = periodic[d] != 0;
        if (!inInterval(b.lo[d], lo, hi, wraps) && !inInterval(lo, b.lo[d], b.hi[d], wraps))
        {
            return false;
        }
    }
    return true;
}

std::vector<int32_t> atomsWithinRange(const t_pbc*         pbc,
                                      ArrayRef<const RVec> home,
                                      ArrayRef<const RVec> candidates,
                                      real                 range)
{
    std::vector<int32_t> found;
    // a cutoff of 0 means none to AnalysisNeighborhood
    if (range <= 0 || home.empty())
    {
        return found;
    }
    AnalysisNeighborhood neighborhood;
    neighborhood.setCutoff(range);
    const auto search = neighborhood.initSearch(
            pbc, AnalysisNeighborhoodPositions(as_rvec_array(home.data()), home.ssize()));
    for (Index i = 0; i < candidates.ssize(); ++i)
    {
        if (search.isWithin(AnalysisNeighborhoodPositions(candidates[i].as_vec())))
        {
            found.push_back(static_cast<int32_t>(i));
        }
    }
    return found;
}

void findPairs(const t_pbc*                  pbc,
               const matrix                  box,
               const std::array<uint8_t, 3>& periodic,
               ArrayRef<const RVec>          x,
               real                          cutoff,
               std::vector<int32_t>*         pairs,
               std::vector<IVec>*            shifts)
{
    pairs->clear();
    shifts->clear();
    if (cutoff <= 0 || x.ssize() < 2)
    {
        return;
    }
    AnalysisNeighborhood neighborhood;
    neighborhood.setCutoff(cutoff);
    const auto search = neighborhood.initSearch(
            pbc, AnalysisNeighborhoodPositions(as_rvec_array(x.data()), x.ssize()));
    auto                     pairSearch = search.startSelfPairSearch();
    AnalysisNeighborhoodPair pair;
    while (pairSearch.findNextPair(&pair))
    {
        // dx() is the minimum image of x[ref] - x[test]: the pair goes from test to ref,
        // and its lattice shift is what dx() adds to the plain difference.
        const int  first  = pair.testIndex();
        const int  second = pair.refIndex();
        const RVec dx(pair.dx());
        const RVec lattice = dx - (x[second] - x[first]);
        // minimumImageCellShift(lattice) brings a lattice vector to 0, so it is minus its shift
        const IVec n = minimumImageCellShift(box, periodic, lattice);
        pairs->push_back(first);
        pairs->push_back(second);
        shifts->push_back(IVec(-n[XX], -n[YY], -n[ZZ]));
    }
}

} // namespace gmx
