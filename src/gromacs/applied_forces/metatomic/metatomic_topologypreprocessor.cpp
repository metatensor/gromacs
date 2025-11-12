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
 * Implements MetatomicTopologyPrepocessor.
 *
 * \author Metatensor developers <https://github.com/metatensor>
 * \ingroup module_applied_forces
 */

#include "gromacs/fileio/warninp.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/logger.h"

#include "metatomic_topologypreprocessor.h"

namespace gmx
{

MetatomicTopologyPreprocessor::MetatomicTopologyPreprocessor(ArrayRef<const Index> inputIndices) :
    QMMMTopologyPreprocessor(inputIndices), mtaIndices_(inputIndices.begin(), inputIndices.end())
{
}

void MetatomicTopologyPreprocessor::preprocess(gmx_mtop_t* mtop, const MDLogger& logger, WarningHandler* wi)
{
    // We're re-using the topology-modifying functions from QMMM module for now,
    // since they contain the same modifications as needed for ML/MM with metatomic. This should be
    // refactored in the future.
    GMX_LOG(logger.info)
            .appendText("Metatomic potential Interface is active, topology was modified!");

    GMX_LOG(logger.info)
            .appendTextFormatted(
                    "Number of embedded metatomic atoms: %td\nNumber of regular atoms: %td\n",
                    gmx::ssize(mtaIndices_),
                    mtop->natoms - gmx::ssize(mtaIndices_));

    // 1) Split molecules containing Metatomic input atoms from other molecules in blocks
    std::vector<bool> isMTABlock = splitQMBlocks(mtop, mtaIndices_);

    // 2) Exclude LJ interactions between MTA atoms
    // this also excludes coulomb interactions
    addQMLJExclusions(mtop, mtaIndices_, logger);

    // 3) Build atomNumbers vector with atomic numbers of all atoms
    buildQMMMAtomNumbers(*mtop);

    // 4) Make F_CONNBOND between atoms within MTA region
    modifyQMMMTwoCenterInteractions(mtop, mtaIndices_, isMTABlock, logger);

    // 5) Remove angles and settles containing 2 or more MTA atoms
    modifyQMMMThreeCenterInteractions(mtop, mtaIndices_, isMTABlock, logger);

    // 6) Remove dihedrals containing 3 or more MTA atoms
    modifyQMMMFourCenterInteractions(mtop, mtaIndices_, isMTABlock, logger);

    // 7) Check for constrained bonds in QM subsystem
    checkConstrainedBonds(mtop, mtaIndices_, isMTABlock, wi);

    // finalize topology
    mtop->finalize();
}

} // namespace gmx
