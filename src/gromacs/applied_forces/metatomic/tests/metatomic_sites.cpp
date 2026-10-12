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
 * derived work in the form of exact copies or altered versions of
 * the source code is not likely to be accepted, so we can not
 * accept modifications of the source code.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */
/*! \internal \file
 * \brief
 * Tests for the assignment of ML sites to ranks under domain decomposition
 *
 * \author Metatensor developers <https://github.com/metatensor>
 * \ingroup module_applied_forces
 */

#include "gmxpre.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/applied_forces/metatomic/metatomic_options.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/stringutil.h"

namespace gmx
{
namespace test
{
namespace
{

using Sites = std::vector<std::vector<int>>;

TEST(AssignSitesToRanks, OneRankEvaluatesEverySiteInOrder)
{
    EXPECT_EQ(assignSitesToRanks(std::vector<int>{ 7, 30, 5, 12 }, 1), (Sites{ { 0, 1, 2, 3 } }));
}

TEST(AssignSitesToRanks, EqualSitesGoRoundRobin)
{
    const std::vector<int> sizes(4, 85);
    EXPECT_EQ(assignSitesToRanks(sizes, 2), (Sites{ { 0, 2 }, { 1, 3 } }));
    EXPECT_EQ(assignSitesToRanks(sizes, 3), (Sites{ { 0, 3 }, { 1 }, { 2 } }));
    EXPECT_EQ(assignSitesToRanks(sizes, 4), (Sites{ { 0 }, { 1 }, { 2 }, { 3 } }));
}

TEST(AssignSitesToRanks, PlacesTheLargestSitesFirst)
{
    // 20 to rank 0, then 7, 7 and 5 to rank 1, which stays below 20 throughout
    EXPECT_EQ(assignSitesToRanks(std::vector<int>{ 5, 20, 7, 7 }, 2), (Sites{ { 1 }, { 0, 2, 3 } }));
    // a site in the input group's order may still come first by size
    EXPECT_EQ(assignSitesToRanks(std::vector<int>{ 3, 3, 9 }, 2), (Sites{ { 2 }, { 0, 1 } }));
}

TEST(AssignSitesToRanks, BreaksTiesTowardsTheLowestRank)
{
    // after 6 and 6, both ranks carry 6: the next site goes to rank 0
    EXPECT_EQ(assignSitesToRanks(std::vector<int>{ 6, 6, 1 }, 2), (Sites{ { 0, 2 }, { 1 } }));
}

TEST(AssignSitesToRanks, LeavesRanksWithoutSitesWhenThereAreFewer)
{
    EXPECT_EQ(assignSitesToRanks(std::vector<int>{ 7, 5 }, 4), (Sites{ { 0 }, { 1 }, {}, {} }));
    EXPECT_EQ(assignSitesToRanks(std::vector<int>{}, 3), (Sites{ {}, {}, {} }));
}

TEST(AssignSitesToRanks, CountsEmptySitesAsSites)
{
    EXPECT_EQ(assignSitesToRanks(std::vector<int>{ 0, 4, 0 }, 2), (Sites{ { 1 }, { 0, 2 } }));
}

TEST(AssignSitesToRanks, GivesEverySiteToOneRankWithinOneSiteOfBalance)
{
    std::mt19937                       rng(2026);
    std::uniform_int_distribution<int> size(1, 400);
    for (int trial = 0; trial < 200; ++trial)
    {
        std::vector<int> sizes(1 + trial % 17);
        std::generate(sizes.begin(), sizes.end(), [&] { return size(rng); });
        const int  numRanks = 1 + trial % 9;
        const auto bySite   = assignSitesToRanks(sizes, numRanks);
        SCOPED_TRACE(testing::Message() << sizes.size() << " sites, " << numRanks << " ranks");

        ASSERT_EQ(bySite.size(), size_t(numRanks));
        std::vector<int> all;
        std::vector<int> load;
        for (const auto& sites : bySite)
        {
            EXPECT_TRUE(std::is_sorted(sites.begin(), sites.end()));
            all.insert(all.end(), sites.begin(), sites.end());
            load.push_back(std::accumulate(
                    sites.begin(), sites.end(), 0, [&](int sum, int s) { return sum + sizes[s]; }));
        }
        std::sort(all.begin(), all.end());
        std::vector<int> expected(sizes.size());
        std::iota(expected.begin(), expected.end(), 0);
        EXPECT_EQ(all, expected);
        // the last site placed on the busiest rank went to the least-loaded rank of the time
        EXPECT_LE(*std::max_element(load.begin(), load.end()) - *std::min_element(load.begin(), load.end()),
                  *std::max_element(sizes.begin(), sizes.end()));
        // and every rank computes the same answer
        EXPECT_EQ(assignSitesToRanks(sizes, numRanks), bySite);
    }
}

TEST(AssignSitesToRanks, RejectsFewerThanOneRank)
{
    for (const int numRanks : { 0, -2 })
    {
        try
        {
            assignSitesToRanks(std::vector<int>{ 1, 2 }, numRanks);
            ADD_FAILURE() << "no exception for " << numRanks << " ranks";
        }
        catch (const InternalError& e)
        {
            EXPECT_STREQ(e.what(),
                         formatString("Cannot assign metatomic sites to %d ranks.", numRanks).c_str());
        }
    }
}

} // namespace
} // namespace test
} // namespace gmx
