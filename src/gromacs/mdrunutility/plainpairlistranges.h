/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2025- The GROMACS Authors
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
/*! \libinternal \file
 * \brief
 * Declares and implements PlainPairlistRanges
 *
 * \author Berk Hess <hess@kth.se>
 * \inlibraryapi
 * \ingroup module_mdrunutility
 */

#ifndef GMX_MDRUNUTILITY_PLAINPAIRLISTRANGES_H
#define GMX_MDRUNUTILITY_PLAINPAIRLISTRANGES_H

#include <optional>
#include <vector>

#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/real.h"

struct gmx_mtop_t;
struct t_inputrec;

namespace gmx
{

/*! \libinternal \brief Gathers pairlist range requests from different MDModules
 */
class PlainPairlistRanges
{
public:
    //! Constructor
    PlainPairlistRanges(const gmx_mtop_t& mtop, const t_inputrec& inputrec);

    /*! Returns the, optional, RMS change of pair distances over the pairlist lifetime
     *
     * With dynamics and a constant ensemble temperature, returns the RMS change of pair distances
     * over the pairlist lifetime. This is computed as: lifetime * sqrt(2 * kT * avInvMass),
     * where avInvMass is the average inverse mass over all particles with mass.
     * With free-energy calculations the lowest mass from the A and B states is used.
     * This rmsdDistance can be used to set the pairlist buffer. A value of 2*rmsdDistance
     * is recommended.
     *
     * With energy minimization 0 is returned to indicate negligible displacement.
     *
     * Otherwise no value is returned.
     */
    std::optional<real> rmsdDistance() const;

    /*! Adds a range request for the plain pairlist
     *
     * \param[in] range  The range of the pairlist in nm, should be > 0 (throws otherwise)
     */
    void addRange(real range);

    /*! Adds a range request for pairs between atoms in \p atoms only
     *
     * The plain pairlist is restricted to pairs of atoms that are both in the union of
     * the requested atom sets, but only when all range requests supply an atom set.
     *
     * \param[in] range  The range of the pairlist in nm, should be > 0 (throws otherwise)
     * \param[in] atoms  Global atom indices, should not be empty (throws otherwise)
     */
    void addRange(real range, ArrayRef<const Index> atoms);

    //! Returns the list of requested ranges
    ArrayRef<const real> ranges() const { return ranges_; }

    //! Returns the range of the normal pairlist, the upper limit for any requested range
    real pairlistCutoff() const;

    /*! Returns the global atom indices the plain pairlist can be restricted to
     *
     * Returns the sorted union of the requested atom sets when every range request
     * supplied one, otherwise no value.
     */
    std::optional<std::vector<Index>> atoms() const;

private:
    //! Reference to the global topology
    const gmx_mtop_t& mtop_;
    //! Reference to the input record
    const t_inputrec& inputrec_;
    //! List of ranges
    std::vector<real> ranges_;
    //! Union of the atom sets of the restricted range requests
    std::vector<Index> atoms_;
    //! Whether a range was requested without an atom set
    bool haveUnrestrictedRange_ = false;
};

} // namespace gmx

#endif
