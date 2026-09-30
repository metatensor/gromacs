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
 * Tests for the correction of the Coulomb interaction between embedded atoms.
 *
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "gromacs/applied_forces/metatomic/embedded_coulomb_correction.h"

#include <cmath>

#include <vector>

#include <gtest/gtest.h>

#include "gromacs/ewald/ewald_utils.h"
#include "gromacs/math/units.h"
#include "gromacs/mdtypes/group.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/pbcutil/pbc.h"

namespace gmx
{
namespace test
{
namespace
{

constexpr real c_cutoff = 0.8;
constexpr real c_box    = 3.0;

interaction_const_t pmeConstants()
{
    interaction_const_t ic;
    ic.coulomb.type       = CoulombInteractionType::Pme;
    ic.coulomb.cutoff     = c_cutoff;
    ic.coulomb.ewaldCoeff = calc_ewaldcoeff_q(c_cutoff, 1e-5);
    ic.coulomb.epsfac     = c_one4PiEps0;
    return ic;
}

interaction_const_t reactionFieldConstants()
{
    interaction_const_t ic;
    ic.coulomb.type                     = CoulombInteractionType::RF;
    ic.coulomb.cutoff                   = c_cutoff;
    ic.coulomb.epsfac                   = c_one4PiEps0;
    ic.coulomb.reactionFieldCoefficient = 0.4;
    ic.coulomb.reactionFieldShift       = 1 / c_cutoff + 0.4 * c_cutoff * c_cutoff;
    return ic;
}

t_pbc cubicPbc()
{
    matrix box = { { c_box, 0, 0 }, { 0, c_box, 0 }, { 0, 0, c_box } };
    t_pbc  pbc;
    set_pbc(&pbc, PbcType::Xyz, box);
    return pbc;
}

//! Energy of a pair at distance \p r along x.
double pairEnergy(real r, const interaction_const_t& ic)
{
    const std::vector<RVec> x = { { 1, 1, 1 }, { 1 + r, 1, 1 } };
    const std::vector<real> q = { 0.4, -0.3 };
    return computeEmbeddedCoulombCorrection(x, q, cubicPbc(), ic).energy;
}

TEST(EmbeddedCoulombCorrection, EwaldRemovesReciprocalPairBeyondCutoffOnly)
{
    const auto   ic   = pmeConstants();
    const double qq   = c_one4PiEps0 * 0.4 * -0.3;
    const double beta = ic.coulomb.ewaldCoeff;
    EXPECT_EQ(pairEnergy(0.5, ic), 0);
    for (const double r : { 0.8001, 1.2 })
    {
        EXPECT_NEAR(pairEnergy(r, ic), -qq * std::erf(beta * r) / r, 1e-4) << "r = " << r;
    }
}

TEST(EmbeddedCoulombCorrection, ReactionFieldRemovesExclusionTermWithinCutoffOnly)
{
    const auto   ic  = reactionFieldConstants();
    const double qq  = c_one4PiEps0 * 0.4 * -0.3;
    const double krf = ic.coulomb.reactionFieldCoefficient;
    const double crf = ic.coulomb.reactionFieldShift;
    EXPECT_NEAR(pairEnergy(0.5, ic), -qq * (krf * 0.25 - crf), 1e-4);
    EXPECT_EQ(pairEnergy(1.2, ic), 0);
}

TEST(EmbeddedCoulombCorrection, CompletesKernelExclusionContinuously)
{
    // The kernels remove -qq erf(beta r)/r (Ewald) or add qq (k_rf r^2 - c_rf)
    // (reaction-field) for excluded pairs within the cut-off; with the
    // correction, the total over the cut-off is continuous.
    const double qq        = c_one4PiEps0 * 0.4 * -0.3;
    const auto   pme       = pmeConstants();
    const double beta      = pme.coulomb.ewaldCoeff;
    const auto   kernelPme = [&](double r)
    { return r < c_cutoff ? -qq * std::erf(beta * r) / r : 0.0; };
    const auto rf       = reactionFieldConstants();
    const auto kernelRf = [&](double r)
    {
        return r < c_cutoff ? qq * (rf.coulomb.reactionFieldCoefficient * r * r - rf.coulomb.reactionFieldShift)
                            : 0.0;
    };
    for (const real r : { c_cutoff - real(1e-4), c_cutoff + real(1e-4) })
    {
        EXPECT_NEAR(kernelPme(r) + pairEnergy(r, pme), -qq * std::erf(beta * c_cutoff) / c_cutoff, 0.02)
                << "r = " << r;
        EXPECT_NEAR(kernelRf(r) + pairEnergy(r, rf), 0, 1e-4) << "r = " << r;
    }
}

TEST(EmbeddedCoulombCorrection, ForcesAndVirialMatchFiniteDifferences)
{
    // Three atoms, one pair across the periodic boundary and beyond the cut-off.
    const std::vector<RVec> x0  = { { 0.2, 1.0, 1.1 }, { 2.7, 1.3, 0.9 }, { 1.4, 1.6, 1.4 } };
    const std::vector<real> q   = { 0.4, -0.3, 0.25 };
    const t_pbc             pbc = cubicPbc();
    for (const auto& ic : { pmeConstants(), reactionFieldConstants() })
    {
        const auto result = computeEmbeddedCoulombCorrection(x0, q, pbc, ic);
        EXPECT_GT(result.numPairs, 0);
        const double h                = 1e-3;
        double       virial[DIM][DIM] = { { 0 } };
        for (int i = 0; i < 3; i++)
        {
            for (int d = 0; d < DIM; d++)
            {
                auto plus = x0, minus = x0;
                plus[i][d] += h;
                minus[i][d] -= h;
                const double fd = -(computeEmbeddedCoulombCorrection(plus, q, pbc, ic).energy
                                    - computeEmbeddedCoulombCorrection(minus, q, pbc, ic).energy)
                                  / (2 * h);
                EXPECT_NEAR(result.forces[i][d], fd, 0.05 + 1e-3 * std::abs(fd))
                        << "atom " << i << " dim " << d;
            }
        }
        // -0.5 sum_pairs dx (x) f, with the minimum-image dx
        for (int i = 0; i < 3; i++)
        {
            for (int j = i + 1; j < 3; j++)
            {
                const std::vector<RVec> pair  = { x0[i], x0[j] };
                const std::vector<real> qPair = { q[i], q[j] };
                const auto f = computeEmbeddedCoulombCorrection(pair, qPair, pbc, ic).forces[0];
                RVec       dx;
                pbc_dx_aiuc(&pbc, x0[i], x0[j], dx);
                for (int d = 0; d < DIM; d++)
                {
                    for (int e = 0; e < DIM; e++)
                    {
                        virial[d][e] -= 0.5 * dx[d] * f[e];
                    }
                }
            }
        }
        for (int d = 0; d < DIM; d++)
        {
            for (int e = 0; e < DIM; e++)
            {
                EXPECT_NEAR(result.virial[d][e], virial[d][e], 1e-3);
            }
        }
    }
}

TEST(EmbeddedCoulombCorrection, SplitsEnergyOverEnergyGroups)
{
    const std::vector<RVec> x      = { { 0.2, 1, 1 }, { 1.2, 1, 1 }, { 2.2, 1, 1 } };
    const std::vector<real> q      = { 0.4, -0.3, 0.25 };
    const std::vector<int>  groups = { 0, 1, 1 };
    const auto result = computeEmbeddedCoulombCorrection(x, q, cubicPbc(), pmeConstants(), groups, 2);
    ASSERT_EQ(result.groupPairEnergies.size(), 4U);
    double total = 0;
    for (const double e : result.groupPairEnergies)
    {
        total += e;
    }
    EXPECT_NEAR(total, result.energy, 1e-6);
    EXPECT_NE(result.groupPairEnergies[GID(0, 1, 2)], 0);
    EXPECT_NE(result.groupPairEnergies[GID(1, 1, 2)], 0);
    EXPECT_EQ(result.groupPairEnergies[GID(0, 0, 2)], 0);
}

TEST(EmbeddedCoulombCorrection, SameResultOnSeveralThreadsAndOnlyWithinSites)
{
    // 60 charged atoms spread over the box, in two sites
    std::vector<RVec> x;
    std::vector<real> q;
    std::vector<int>  sites;
    for (int i = 0; i < 60; i++)
    {
        x.push_back({ real(0.1 + 0.037 * i), real(0.2 + std::fmod(0.61 * i, c_box - 0.4)), real(0.3 + std::fmod(0.29 * i, c_box - 0.6)) });
        q.push_back(real(0.1 * ((i % 5) - 2)));
        sites.push_back(i % 2);
    }
    for (const auto& ic : { pmeConstants(), reactionFieldConstants() })
    {
        const auto serial   = computeEmbeddedCoulombCorrection(x, q, cubicPbc(), ic, {}, 1, sites, 1);
        const auto threaded = computeEmbeddedCoulombCorrection(x, q, cubicPbc(), ic, {}, 1, sites, 4);
        EXPECT_GT(serial.numPairs, 0);
        EXPECT_EQ(serial.numPairs, threaded.numPairs);
        EXPECT_NEAR(serial.energy, threaded.energy, 1e-9 * std::abs(serial.energy) + 1e-12);
        for (size_t i = 0; i < x.size(); i++)
        {
            for (int d = 0; d < DIM; d++)
            {
                EXPECT_NEAR(serial.forces[i][d], threaded.forces[i][d], 1e-4);
            }
        }
        // Pairs between the two sites are left alone: an all-pairs run has more of them
        const auto allPairs = computeEmbeddedCoulombCorrection(x, q, cubicPbc(), ic, {}, 1, {}, 4);
        EXPECT_GT(allPairs.numPairs, serial.numPairs);
    }
}

} // namespace
} // namespace test
} // namespace gmx
