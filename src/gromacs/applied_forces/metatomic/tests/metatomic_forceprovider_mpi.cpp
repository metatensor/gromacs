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
 * Tests of the MetatomicForceProvider with domain decomposition over two ranks
 *
 * \author Eric Boittier <eric.boittier@epfl.ch>
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "config.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "gromacs/applied_forces/metatomic/metatomic_forceprovider.h"
#include "gromacs/applied_forces/metatomic/metatomic_options.h"
#include "gromacs/domdec/localatomset.h"
#include "gromacs/domdec/localatomsetmanager.h"
#include "gromacs/mdrunutility/mdmodulesnotifiers.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/mdtypes/iforceprovider.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/topology/atoms.h"
#include "gromacs/utility/gmxmpi.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/loggerbuilder.h"
#include "gromacs/utility/mpicomm.h"
#include "gromacs/utility/stringstream.h"

#include "testutils/mpitest.h"

#include "linkcaps.h"
#include "testmodel.h"

namespace gmx
{
namespace test
{
namespace
{

using ::testing::HasSubstr;

/*! \brief The warnings logged by the main rank after one evaluation of two argon atoms
 *
 * Each atom has an energy uncertainty of 12 kJ/mol or more, above the threshold of
 * 10 kJ/mol. \p homeAtoms are the global indices of the atoms on this rank.
 */
std::string uncertaintyWarnings(const MpiComm& mpiComm, const std::vector<int>& homeAtoms)
{
    MetatomicOptions    options;
    LocalAtomSetManager atomSets;
    init_t_atoms(&options.params_.atoms_, 2, false);
    options.params_.atoms_.atom[0].atomnumber = 18;
    options.params_.atoms_.atom[1].atomnumber = 18;
    options.params_.active                    = true;
    options.params_.modelPath_                = "gromacs-test-recorder-model";
    options.params_.modelOptions              = R"(uncertainty="12")";
    options.params_.uncertaintyThreshold      = "10";
    options.params_.mtaIndices_               = { 0, 1 };
    options.params_.numAtoms_                 = 2;
    options.params_.mtaAtoms_                 = std::make_unique<LocalAtomSet>(
            atomSets.add(ArrayRef<const Index>(options.params_.mtaIndices_)));
    options.params_.pbcType_ = std::make_unique<PbcType>(PbcType::Xyz);

    // as in mdrun, only the main rank logs
    StringOutputStream warnings;
    LoggerBuilder      builder;
    if (mpiComm.rank() == 0)
    {
        builder.addTargetStream(MDLogger::LogLevel::Warning, &warnings);
    }
    const LoggerOwner logger = builder.build();

    {
        MetatomicForceProvider provider(options, logger.logger(), mpiComm);

        // the two atoms are far apart
        const matrix      box = { { 3, 0, 0 }, { 0, 3, 0 }, { 0, 0, 3 } };
        std::vector<RVec> x;
        for (int atom : homeAtoms)
        {
            x.push_back(atom == 0 ? RVec(0.5, 0.5, 0.5) : RVec(2, 2, 2));
        }
        provider.gatherAtomNumbersIndices(
                MDModulesAtomsRedistributedSignal(box, x, {}, {}, ArrayRef<const int>(homeAtoms)));

        std::vector<RVec>   forces(x.size(), RVec(0, 0, 0));
        ForceWithVirial     forceWithVirial(forces, false);
        gmx_enerdata_t      enerd(1, nullptr);
        ForceProviderOutput output(&forceWithVirial, &enerd);
        provider.calculateForces(
                ForceProviderInput(x, static_cast<int>(x.size()), {}, {}, 0.0, 0, box, mpiComm, nullptr),
                &output);
    }
    done_atom(&options.params_.atoms_);
    return warnings.toString();
}

TEST(MetatomicForceProviderMpi, CountsAtomsAboveTheUncertaintyThresholdOnAllRanks)
{
    GMX_MPI_TEST(RequireRankCount<2>);
    ASSERT_TRUE(registerRecorderPlugin());
    const MpiComm mpiComm(MPI_COMM_WORLD);
    // one atom on each rank
    const std::string warnings = uncertaintyWarnings(mpiComm, { mpiComm.rank() });
    if (mpiComm.rank() == 0)
    {
        EXPECT_THAT(warnings, HasSubstr("uncertainty on atomic energies for 2 atoms is larger"));
    }
}

TEST(MetatomicForceProviderMpi, CountsUncertaintiesWithARankWithoutAtoms)
{
    GMX_MPI_TEST(RequireRankCount<2>);
    ASSERT_TRUE(registerRecorderPlugin());
    const MpiComm mpiComm(MPI_COMM_WORLD);
    // both atoms on rank 1: rank 0 evaluates nothing but logs the count of rank 1
    const std::string warnings = uncertaintyWarnings(
            mpiComm, mpiComm.rank() == 1 ? std::vector<int>{ 0, 1 } : std::vector<int>{});
    if (mpiComm.rank() == 0)
    {
        EXPECT_THAT(warnings, HasSubstr("uncertainty on atomic energies for 2 atoms is larger"));
    }
}

/*! \brief Link caps of linkcaps.h with the atoms in \p homeAtoms on each rank.
 *
 * Checks the energy and virial summed over the ranks, and the forces on the home atoms
 * of each rank, against those of linkcaps.h.
 */
void checkLinkCaps(const MpiComm&                       mpiComm,
                   const LinkCapSystem&                 system,
                   const std::vector<std::vector<int>>& homeAtoms)
{
    const int           numAtoms = static_cast<int>(system.positions.size());
    MetatomicOptions    options;
    LocalAtomSetManager atomSets;
    init_t_atoms(&options.params_.atoms_, numAtoms, false);
    for (int i = 0; i < numAtoms; ++i)
    {
        options.params_.atoms_.atom[i].atomnumber = 18;
        options.params_.mtaIndices_.push_back(i);
    }
    options.params_.active       = true;
    options.params_.modelPath_   = "metatomic-lj-model";
    options.params_.modelOptions = system.modelOptions();
    options.params_.numAtoms_    = numAtoms;
    options.params_.mtaAtoms_    = std::make_unique<LocalAtomSet>(
            atomSets.add(ArrayRef<const Index>(options.params_.mtaIndices_)));
    options.params_.pbcType_      = std::make_unique<PbcType>(PbcType::Xyz);
    options.params_.linkFrontier_ = system.linkFrontier();

    matrix box;
    for (int a = 0; a < DIM; ++a)
    {
        for (int b = 0; b < DIM; ++b)
        {
            box[a][b] = static_cast<real>(system.box[a][b]);
        }
    }
    const auto&       home = homeAtoms[mpiComm.rank()];
    std::vector<RVec> x;
    for (const int atom : home)
    {
        x.push_back(system.positions[atom].toRVec());
    }

    const MDLogger         logger;
    MetatomicForceProvider provider(options, logger, mpiComm);
    provider.gatherAtomNumbersIndices(
            MDModulesAtomsRedistributedSignal(box, x, {}, {}, ArrayRef<const int>(home)));
    std::vector<RVec>   forces(x.size(), RVec(0, 0, 0));
    ForceWithVirial     forceWithVirial(forces, true);
    gmx_enerdata_t      enerd(1, nullptr);
    ForceProviderOutput output(&forceWithVirial, &enerd);
    provider.calculateForces(
            ForceProviderInput(x, static_cast<int>(x.size()), {}, {}, 0.0, 0, box, mpiComm, nullptr),
            &output);

    std::vector<double> sums = { enerd.term[InteractionFunction::MetatomicPotentialEnergy] };
    for (int a = 0; a < DIM; ++a)
    {
        for (int b = 0; b < DIM; ++b)
        {
            sums.push_back(forceWithVirial.getVirial()[a][b]);
        }
    }
    mpiComm.sumReduce(sums.size(), sums.data());

    constexpr double tolerance = GMX_DOUBLE ? 1e-8 : 1e-4;
    const double     energy    = system.energy();
    EXPECT_NEAR(sums[0], energy, tolerance * std::abs(energy));
    const auto expectedForces = system.forces();
    for (size_t k = 0; k < home.size(); ++k)
    {
        for (int d = 0; d < DIM; ++d)
        {
            const double expected = expectedForces[home[k]][d];
            EXPECT_NEAR(forces[k][d], expected, 1e-3 * (1 + std::abs(expected)))
                    << "atom " << home[k] << ", dimension " << d;
        }
    }
    const auto expectedVirial = system.virial();
    for (int a = 0; a < DIM; ++a)
    {
        for (int b = 0; b < DIM; ++b)
        {
            EXPECT_NEAR(sums[1 + 3 * a + b], expectedVirial[a][b], 1e-3 * (1 + std::abs(expectedVirial[a][b])))
                    << "virial " << a << ", " << b;
        }
    }
    done_atom(&options.params_.atoms_);
}

TEST(MetatomicForceProviderMpi, SpreadsTheForcesOnLinkCapsAcrossRanks)
{
    GMX_MPI_TEST(RequireRankCount<2>);
    if (!std::filesystem::exists(std::filesystem::path(GMX_METATOMIC_PLUGINS_DIR) / "metatomic-lj-plugin.so"))
    {
        GTEST_SKIP() << "the Lennard-Jones plugin of metatomic-core is not installed";
    }
    const MpiComm mpiComm(MPI_COMM_WORLD);
    // MM atom 3 and one of its embedded atoms on rank 0, its other embedded atom on rank 1:
    // rank 0 evaluates the cap of the other on a row of its own
    {
        SCOPED_TRACE("caps of atom 3 from both ranks");
        checkLinkCaps(mpiComm, LinkCapSystem(), { { 0, 3 }, { 1, 2, 4 } });
    }
    {
        SCOPED_TRACE("every capped bond across ranks");
        checkLinkCaps(mpiComm, LinkCapSystem(), { { 1, 3 }, { 0, 2, 4 } });
    }
    // rank 0 has the MM atom only, and its cap needs embedded atom 1, beyond the interaction
    // range of the MM atom
    {
        SCOPED_TRACE("a neighbour of a cap beyond the range of its MM atom");
        checkLinkCaps(
                mpiComm, LinkCapSystem::withANeighbourBeyondTheRangeOfTheMMAtom(), { { 2 }, { 0, 1 } });
    }
}

} // namespace
} // namespace test
} // namespace gmx
