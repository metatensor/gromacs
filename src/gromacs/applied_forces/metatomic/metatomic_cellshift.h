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
 * Integer cell shifts of metatomic pair lists in GROMACS boxes.
 *
 * GROMACS boxes are lower triangular: the rows are the box vectors a, b, c,
 * with a along x and b in the xy plane. A metatomic pair sample stores the
 * integer cell shift n of its second atom, r_ij = x_j + n_a a + n_b b + n_c c - x_i.
 *
 * \author Metatensor developers <https://github.com/metatensor>
 * \ingroup module_applied_forces
 */
#ifndef GMX_APPLIED_FORCES_METATOMIC_CELLSHIFT_H
#define GMX_APPLIED_FORCES_METATOMIC_CELLSHIFT_H

#include <cmath>
#include <cstdint>

#include <array>

#include "gromacs/utility/vectypes.h"
#include "gromacs/utility/real.h"

namespace gmx
{

//! Cartesian shift n_a a + n_b b + n_c c of an integer cell shift.
inline RVec cellShiftVector(const matrix box, const IVec& cellShift)
{
    return { cellShift[XX] * box[XX][XX] + cellShift[YY] * box[YY][XX] + cellShift[ZZ] * box[ZZ][XX],
             cellShift[YY] * box[YY][YY] + cellShift[ZZ] * box[ZZ][YY],
             cellShift[ZZ] * box[ZZ][ZZ] };
}

/*! \brief Cell shift n, along the periodic box vectors only, bringing \p d to its minimum image.
 *
 * \p d + cellShiftVector(\p box, n) is the minimum image of \p d, found from c to a
 * as GROMACS does for triclinic boxes. Directions that are not \p periodic keep a
 * zero shift, so pbc = xy shifts in the plane only and pbc = no not at all, and
 * their (possibly zero) box vectors are never divided by.
 */
inline IVec minimumImageCellShift(const matrix box, const std::array<uint8_t, 3>& periodic, const RVec& d)
{
    IVec   n(0, 0, 0);
    double r[3] = { d[XX], d[YY], d[ZZ] };
    for (int k = ZZ; k >= XX; --k)
    {
        if (periodic[k] && box[k][k] != 0)
        {
            n[k] = static_cast<int>(std::round(-r[k] / box[k][k]));
            for (int dim = XX; dim <= k; ++dim)
            {
                r[dim] += n[k] * box[k][dim];
            }
        }
    }
    return n;
}

} // namespace gmx

#endif
