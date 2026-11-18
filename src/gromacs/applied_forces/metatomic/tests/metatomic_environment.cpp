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
 * Tests the environment and pair search of the metatomic force provider
 * against a brute-force search over periodic images.
 *
 * \author Eric Boittier <eric.boittier@epfl.ch>
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "config.h"

#include <cmath>

#include <algorithm>
#include <array>
#include <set>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/pbcutil/pbc.h"
#include "gromacs/utility/vec.h"

#if GMX_METATOMIC_ACTIVE
#    include "gromacs/applied_forces/metatomic/metatomic_cellshift.h"
#    include "gromacs/applied_forces/metatomic/metatomic_environment.h"
#endif

namespace gmx
{
namespace test
{
namespace
{

#if GMX_METATOMIC_ACTIVE

const matrix c_box = { { 3, 0, 0 }, { 0.8, 2.8, 0 }, { -0.6, 0.7, 2.6 } };

//! Atoms spread over the box with a Weyl sequence, some of them moved by a box vector.
std::vector<RVec> spreadAtoms(int count)
{
    std::vector<RVec> x;
    for (int i = 0; i < count; ++i)
    {
        const double f[3] = { std::fmod(i * 0.6180339887, 1.0),
                              std::fmod(i * 0.4142135624, 1.0),
                              std::fmod(i * 0.7320508076, 1.0) };
        RVec         r(0, 0, 0);
        for (int a = 0; a < 3; ++a)
        {
            for (int d = 0; d < 3; ++d)
            {
                r[d] += f[a] * c_box[a][d];
            }
        }
        // GROMACS does not keep atoms inside the unit cell
        if (i % 7 == 0)
        {
            r += RVec(c_box[XX]) - RVec(c_box[ZZ]);
        }
        x.push_back(r);
    }
    return x;
}

//! Shortest distance between x[i] and the images of x[j] along the periodic directions.
double minimumDistance(const std::array<uint8_t, 3>& periodic, const RVec& xi, const RVec& xj)
{
    double shortest = 1e30;
    for (int na = -2; na <= 2; ++na)
    {
        for (int nb = -2; nb <= 2; ++nb)
        {
            for (int nc = -2; nc <= 2; ++nc)
            {
                const IVec n(na, nb, nc);
                if ((!periodic[XX] && na != 0) || (!periodic[YY] && nb != 0) || (!periodic[ZZ] && nc != 0))
                {
                    continue;
                }
                shortest = std::min(shortest,
                                    static_cast<double>(norm(xj + cellShiftVector(c_box, n) - xi)));
            }
        }
    }
    return shortest;
}

struct PbcCase
{
    PbcType                pbcType;
    std::array<uint8_t, 3> periodic;
};

class MetatomicEnvironment : public ::testing::TestWithParam<PbcCase>
{
};

TEST_P(MetatomicEnvironment, FindsEachPairOnceWithItsMinimumImage)
{
    const PbcType pbcType  = GetParam().pbcType;
    const auto&   periodic = GetParam().periodic;
    const auto    x        = spreadAtoms(80);
    const real    cutoff   = 0.9;
    t_pbc         pbc;
    set_pbc(&pbc, pbcType, c_box);

    std::vector<int32_t> pairs;
    std::vector<IVec>    shifts;
    findPairs(&pbc, c_box, periodic, x, cutoff, &pairs, &shifts);

    std::set<std::pair<int, int>> found;
    for (size_t k = 0; k < shifts.size(); ++k)
    {
        const int i = pairs[2 * k];
        const int j = pairs[2 * k + 1];
        EXPECT_TRUE(found.insert({ std::min(i, j), std::max(i, j) }).second) << i << ", " << j;
        // the vector of the pair is the minimum image
        const double length = norm(x[j] + cellShiftVector(c_box, shifts[k]) - x[i]);
        EXPECT_NEAR(length, minimumDistance(periodic, x[i], x[j]), 1e-5) << i << ", " << j;
        for (int d = 0; d < 3; ++d)
        {
            if (!periodic[d])
            {
                EXPECT_EQ(shifts[k][d], 0);
            }
        }
    }
    std::set<std::pair<int, int>> expected;
    for (int i = 0; i < static_cast<int>(x.size()); ++i)
    {
        for (int j = i + 1; j < static_cast<int>(x.size()); ++j)
        {
            if (minimumDistance(periodic, x[i], x[j]) < cutoff)
            {
                expected.insert({ i, j });
            }
        }
    }
    EXPECT_FALSE(expected.empty());
    EXPECT_EQ(found, expected);

    findPairs(&pbc, c_box, periodic, x, 0, &pairs, &shifts);
    EXPECT_TRUE(pairs.empty());
}

TEST_P(MetatomicEnvironment, FindsTheAtomsWithinRangeOfTheHomeAtoms)
{
    const PbcType pbcType    = GetParam().pbcType;
    const auto&   periodic   = GetParam().periodic;
    const auto    x          = spreadAtoms(80);
    const auto    home       = ArrayRef<const RVec>(x).subArray(0, 6);
    const auto    candidates = ArrayRef<const RVec>(x).subArray(6, x.size() - 6);
    const real    range      = 1.1;
    t_pbc         pbc;
    set_pbc(&pbc, pbcType, c_box);

    std::vector<int32_t> expected;
    for (int k = 0; k < candidates.ssize(); ++k)
    {
        const bool near = std::any_of(home.begin(),
                                      home.end(),
                                      [&](const RVec& h)
                                      { return minimumDistance(periodic, h, candidates[k]) < range; });
        if (near)
        {
            expected.push_back(k);
        }
    }
    const auto found = atomsWithinRange(&pbc, home, candidates, range);
    EXPECT_FALSE(expected.empty());
    EXPECT_LT(expected.size(), candidates.size());
    EXPECT_EQ(found, expected);
    EXPECT_TRUE(atomsWithinRange(&pbc, home, candidates, 0).empty());
}

INSTANTIATE_TEST_SUITE_P(Periodicities,
                         MetatomicEnvironment,
                         ::testing::Values(PbcCase{ PbcType::Xyz, { 1, 1, 1 } },
                                           PbcCase{ PbcType::XY, { 1, 1, 0 } },
                                           PbcCase{ PbcType::No, { 0, 0, 0 } }));

#endif // GMX_METATOMIC_ACTIVE

} // namespace
} // namespace test
} // namespace gmx
