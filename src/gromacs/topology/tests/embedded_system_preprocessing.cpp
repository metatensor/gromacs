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
 * Tests embedded-system topology preprocessing helpers.
 *
 * \ingroup module_topology
 */

#include "gmxpre.h"

#include "gromacs/topology/embedded_system_preprocessing.h"

#include <tuple>

#include <gtest/gtest.h>

namespace gmx
{
namespace test
{
namespace
{

TEST(EmbeddedSystemPreprocessingTest, LinkAtomForceSpreadUsesExplicitCellShift)
{
    const RVec embeddedPosition = { 0.95_real, 0.0_real, 0.0_real };
    const RVec mmPosition       = { 0.05_real, 0.0_real, 0.0_real };
    const RVec mmShift          = { 1.0_real, 0.0_real, 0.0_real };
    const RVec linkForce        = { 0.0_real, 2.0_real, 0.0_real };

    const auto [embeddedForce, mmForce] =
            spreadLinkAtomForce(linkForce, embeddedPosition, mmPosition, mmShift, 0.05_real);

    EXPECT_NEAR(embeddedForce[XX], 0.0_real, 1e-6_real);
    EXPECT_NEAR(embeddedForce[YY], 1.0_real, 1e-6_real);
    EXPECT_NEAR(embeddedForce[ZZ], 0.0_real, 1e-6_real);
    EXPECT_NEAR(mmForce[XX], 0.0_real, 1e-6_real);
    EXPECT_NEAR(mmForce[YY], 1.0_real, 1e-6_real);
    EXPECT_NEAR(mmForce[ZZ], 0.0_real, 1e-6_real);
}

TEST(EmbeddedSystemPreprocessingTest, LinkAtomForceSpreadConservesTotalForce)
{
    const RVec embeddedPosition = { 0.0_real, 0.0_real, 0.0_real };
    const RVec mmPosition       = { 1.0_real, 0.0_real, 0.0_real };
    const RVec mmShift          = { 0.0_real, 0.0_real, 0.0_real };
    const RVec linkForce        = { 1.0_real, 2.0_real, 3.0_real };

    const auto [embeddedForce, mmForce] =
            spreadLinkAtomForce(linkForce, embeddedPosition, mmPosition, mmShift, 0.25_real);

    EXPECT_NEAR(embeddedForce[XX] + mmForce[XX], linkForce[XX], 1e-6_real);
    EXPECT_NEAR(embeddedForce[YY] + mmForce[YY], linkForce[YY], 1e-6_real);
    EXPECT_NEAR(embeddedForce[ZZ] + mmForce[ZZ], linkForce[ZZ], 1e-6_real);
}

} // namespace
} // namespace test
} // namespace gmx
