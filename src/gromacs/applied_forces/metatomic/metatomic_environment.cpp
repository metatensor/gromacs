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

#include "gromacs/selection/nbsearch.h"
#include "gromacs/utility/vec.h"

#include "metatomic_cellshift.h"

namespace gmx
{

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
