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
 * Tests that mdrun evaluates a metatomic model and that the resulting
 * energy and forces agree across domain-decomposition ranks.
 *
 * \ingroup module_mdrun_integration_tests
 */
#include "gmxpre.h"

#include "config.h"

#include <cmath>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/pbcutil/pbc.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/trajectory/energyframe.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/vectypes.h"

#include "testutils/cmdlinetest.h"
#include "testutils/mpitest.h"
#include "testutils/simulationdatabase.h"
#include "testutils/testasserts.h"
#include "testutils/testfilemanager.h"
#include "testutils/trajectoryreader.h"

#include "programs/mdrun/mdrun_main.h"

#include "energycomparison.h"
#include "energyreader.h"
#include "moduletest.h"
#include "simulatorcomparison.h"
#include "trajectorycomparison.h"

namespace gmx
{
namespace test
{
namespace
{

//! Lennard-Jones parameters of the exported argon model, in kJ/mol and nm.
constexpr double c_cutoff  = 0.9;
constexpr double c_epsilon = 0.978638;
constexpr double c_sigma   = 0.3401;

/*! \brief Relative tolerance for the analytic comparison.
 *
 * The reference is computed in double precision and mdrun stores the
 * model output after casting it to \c real. An absolute floor of 1e-4
 * (the tolerance times this magnitude) still rejects a missing shift,
 * which is about 0.01 kJ/mol on every pair inside the cutoff.
 */
constexpr double c_analyticMagnitude = 1.0;
constexpr double c_analyticRelative  = 1.0e-4;

//! Single-precision tolerances used when two mdrun ranks are compared.
constexpr double c_energyMagnitude = 10.0;
constexpr double c_energyRelative  = 5.0e-5;
constexpr double c_forceMagnitude  = 100.0;
constexpr double c_forceRelative   = 5.0e-5;

//! Shift that makes the pair energy zero at the cutoff. Constant, so it does not enter the forces.
double lennardJonesShift()
{
    const double sigmaOverCutoff = c_sigma / c_cutoff;
    const double sr2             = sigmaOverCutoff * sigmaOverCutoff;
    const double sr6             = sr2 * sr2 * sr2;
    return 4.0 * c_epsilon * (sr6 * sr6 - sr6);
}

struct AnalyticLennardJones
{
    double            energy   = 0.0;
    int               numPairs = 0;
    std::vector<RVec> forces;
};

/*! \brief Truncated, shifted Lennard-Jones energy and forces.
 *
 * The exported model (metatomic_lj_test, pure PyTorch, full_list false)
 * evaluates \c 4*epsilon*((sigma/r)^12-(sigma/r)^6)-shift for each pair
 * once and splits that energy equally over the two atoms. Pairs with
 * \c r^2 greater than \c cutoff^2 are dropped, matching the filter in
 * the force provider. The shift is constant, so the force on atom \c i
 * from the vector \c r_j-r_i is \c 24*epsilon*(sr6-2*sr12)/r^2 times
 * that vector.
 */
AnalyticLennardJones analyticLennardJones(ArrayRef<const RVec> positions, const matrix box)
{
    t_pbc pbc;
    set_pbc(&pbc, PbcType::Xyz, box);

    const double         shift    = lennardJonesShift();
    const double         cutoff2  = c_cutoff * c_cutoff;
    const int            numAtoms = gmx::ssize(positions);
    std::vector<DVec>    forces(numAtoms, DVec(0.0, 0.0, 0.0));
    AnalyticLennardJones result;

    for (int i = 0; i < numAtoms; ++i)
    {
        for (int j = i + 1; j < numAtoms; ++j)
        {
            rvec dx;
            pbc_dx(&pbc, positions[j].as_vec(), positions[i].as_vec(), dx);
            const double r2 = dx[XX] * dx[XX] + dx[YY] * dx[YY] + dx[ZZ] * dx[ZZ];
            if (r2 > cutoff2 || r2 <= 0.0)
            {
                continue;
            }
            const double r    = std::sqrt(r2);
            const double sr2  = (c_sigma / r) * (c_sigma / r);
            const double sr6  = sr2 * sr2 * sr2;
            const double sr12 = sr6 * sr6;
            result.energy += 4.0 * c_epsilon * (sr12 - sr6) - shift;
            const double scale = 24.0 * c_epsilon * (sr6 - 2.0 * sr12) / r2;
            const DVec   forceOnI(scale * dx[XX], scale * dx[YY], scale * dx[ZZ]);
            forces[i] += forceOnI;
            forces[j] -= forceOnI;
            result.numPairs++;
        }
    }

    result.forces.reserve(numAtoms);
    for (const DVec& force : forces)
    {
        result.forces.emplace_back(force[XX], force[YY], force[ZZ]);
    }
    return result;
}

std::string tailOfFile(const std::filesystem::path& path)
{
    std::ifstream            input(path);
    std::vector<std::string> lines;
    std::string              line;
    while (std::getline(input, line))
    {
        lines.push_back(std::move(line));
    }
    const int   start = std::max(0, static_cast<int>(lines.size()) - 40);
    std::string tail;
    for (int i = start; i < static_cast<int>(lines.size()); ++i)
    {
        tail += lines[i];
        tail += '\n';
    }
    return tail;
}

//! mdrun integration fixture for one exported Lennard-Jones model.
class MetatomicForceProviderTest : public MdrunTestFixture
{
public:
    //! Build the argon tpr with the metatomic model switched on.
    int prepareSimulation()
    {
        runner_.topFileName_ =
                gmx::test::TestFileManager::getInputFilePath("argon-metatomic.top").string();
        runner_.useGroFromDatabase("argon12");

        auto mdp                       = prepareMdpFieldValues("argon12", "md", "no", "no");
        mdp["nsteps"]                  = "4";
        mdp["dt"]                      = "0.001";
        mdp["nstenergy"]               = "1";
        mdp["nstcalcenergy"]           = "1";
        mdp["nstxout"]                 = "1";
        mdp["nstvout"]                 = "0";
        mdp["nstfout"]                 = "1";
        mdp["nstlist"]                 = "1";
        mdp["comm-mode"]               = "None";
        mdp["constraints"]             = "none";
        mdp["coulombtype"]             = "Cut-off";
        mdp["rcoulomb"]                = "0.9";
        mdp["vdwtype"]                 = "Cut-off";
        mdp["rvdw"]                    = "0.9";
        mdp["rlist"]                   = "1.2";
        mdp["verlet-buffer-tolerance"] = "-1";
        mdp["DispCorr"]                = "no";
        mdp["pbc"]                     = "xyz";
        mdp["metatomic-active"]        = "true";
        mdp["metatomic-model"]  = gmx::test::TestFileManager::getInputFilePath("model.pt").string();
        mdp["metatomic-device"] = "cpu";
        mdp["metatomic-uncertainty-threshold"] = "off";
        mdp["metatomic-non-conservative"]      = "false";
        runner_.useStringAsMdpFile(prepareMdpFileContents(mdp));
        return runner_.callGrompp();
    }

    //! Point the runner at a fresh set of mdrun outputs.
    void setOutputTag(const char* tag)
    {
        runner_.edrFileName_ = fileManager_.getTemporaryFilePath(std::string(tag) + ".edr").string();
        runner_.fullPrecisionTrajectoryFileName_ =
                fileManager_.getTemporaryFilePath(std::string(tag) + ".trr").string();
        runner_.logFileName_ = fileManager_.getTemporaryFilePath(std::string(tag) + ".log").string();
        runner_.groOutputFileName_ =
                fileManager_.getTemporaryFilePath(std::string(tag) + ".gro").string();
        runner_.cptOutputFileName_ =
                fileManager_.getTemporaryFilePath(std::string(tag) + ".cpt").string();
        runner_.reducedPrecisionTrajectoryFileName_ =
                fileManager_.getTemporaryFilePath(std::string(tag) + ".xtc").string();
    }

    /*! \brief Run mdrun on \p numRanks.
     *
     * \c callMdrun always appends the test binary's \c -ntmpi, so a
     * one-rank reference inside a multi-rank test has to call
     * \c gmx_mdrun directly. \p ddGrid, when set, is \c -dd nx ny nz.
     * Four ranks use a 2x2x1 grid: the argon box is large enough for
     * two domains along x and y, and four ranks do not make a 3D grid.
     * \returns the mdrun exit code.
     */
    int runMdrun(int numRanks, const std::optional<std::array<int, 3>>& ddGrid)
    {
        CommandLine caller;
        caller.append("mdrun");
        caller.addOption("-s", runner_.tprFileName_);
        caller.addOption("-g", runner_.logFileName_);
        caller.addOption("-e", runner_.edrFileName_);
        caller.addOption("-o", runner_.fullPrecisionTrajectoryFileName_);
        caller.addOption("-x", runner_.reducedPrecisionTrajectoryFileName_);
        caller.addOption("-c", runner_.groOutputFileName_);
        caller.addOption("-cpo", runner_.cptOutputFileName_);
        caller.addOption("-nb", "cpu");
        caller.addOption("-pme", "cpu");
        caller.addOption("-npme", 0);
        caller.addOption("-dlb", "no");
#if GMX_OPENMP
        caller.addOption("-ntomp", 1);
#endif
#if GMX_THREAD_MPI
        caller.addOption("-ntmpi", numRanks);
#else
        GMX_UNUSED_VALUE(numRanks);
#endif
        if (ddGrid.has_value())
        {
            caller.append("-dd");
            for (const int cells : *ddGrid)
            {
                caller.append(std::to_string(cells).c_str());
            }
        }

        return gmx_mdrun(MdrunTestFixtureBase::s_communicator,
                         *MdrunTestFixtureBase::s_hwinfo,
                         caller.argc(),
                         caller.argv());
    }
};

TEST_F(MetatomicForceProviderTest, MatchesAnalyticLennardJones)
{
    if (getNumberOfTestMpiRanks() != 1)
    {
        GTEST_SKIP() << "The analytic comparison runs on one rank; this test has "
                     << getNumberOfTestMpiRanks() << " ranks.";
    }

    ASSERT_EQ(0, prepareSimulation());
    setOutputTag("analytic");
    ASSERT_EQ(0, runMdrun(1, std::nullopt)) << tailOfFile(runner_.logFileName_);

    const std::string metatomicName =
            interaction_function[InteractionFunction::MetatomicPotentialEnergy].longname;
    const std::string ljName = interaction_function[InteractionFunction::LennardJonesShortRange].longname;
    const auto analyticTolerance =
            relativeToleranceAsFloatingPoint(c_analyticMagnitude, c_analyticRelative);
    const auto zeroTolerance = absoluteTolerance(1.0e-4);

    TrajectoryFrameReader trajectory(runner_.fullPrecisionTrajectoryFileName_);
    auto energies  = openEnergyFileToReadTerms(runner_.edrFileName_, { metatomicName, ljName });
    int  numFrames = 0;
    int  numPairs  = 0;
    bool sawNonZeroForce = false;
    do
    {
        const auto trajectoryFrame = trajectory.frame();
        const auto energyFrame     = energies->frame();
        SCOPED_TRACE(trajectoryFrame.frameName());
        ASSERT_TRUE(trajectoryFrame.hasBox());
        ASSERT_FALSE(trajectoryFrame.x().empty());
        ASSERT_FALSE(trajectoryFrame.f().empty());

        matrix box;
        for (int dim = 0; dim < DIM; ++dim)
        {
            for (int dim2 = 0; dim2 < DIM; ++dim2)
            {
                box[dim][dim2] = trajectoryFrame.box()[dim][dim2];
            }
        }
        const auto expected = analyticLennardJones(trajectoryFrame.x(), box);
        numPairs += expected.numPairs;
        ASSERT_EQ(trajectoryFrame.f().size(), expected.forces.size());
        EXPECT_REAL_EQ_TOL(expected.energy, energyFrame.at(metatomicName), analyticTolerance);
        EXPECT_REAL_EQ_TOL(0.0, energyFrame.at(ljName), zeroTolerance);
        for (std::size_t atom = 0; atom < expected.forces.size(); ++atom)
        {
            for (int dim = 0; dim < DIM; ++dim)
            {
                if (std::fabs(expected.forces[atom][dim]) > 1.0)
                {
                    sawNonZeroForce = true;
                }
                EXPECT_REAL_EQ_TOL(expected.forces[atom][dim], trajectoryFrame.f()[atom][dim], analyticTolerance)
                        << "atom " << atom << " dimension " << dim;
            }
        }
        ++numFrames;
        const bool moreTrajectory = trajectory.readNextFrame();
        const bool moreEnergy     = energies->readNextFrame();
        if (!moreTrajectory || !moreEnergy)
        {
            EXPECT_EQ(moreTrajectory, moreEnergy)
                    << "Energy and trajectory files do not contain the same number of frames";
            break;
        }
    } while (true);
    ASSERT_GT(numFrames, 0);
    ASSERT_GT(numPairs, 0);
    EXPECT_TRUE(sawNonZeroForce);
}

TEST_F(MetatomicForceProviderTest, MatchesOneRankRun)
{
    const int numRanks = getNumberOfTestMpiRanks();
    if (numRanks < 2)
    {
        GTEST_SKIP() << "This comparison needs at least two ranks.";
    }
#if !GMX_THREAD_MPI
    GTEST_SKIP()
            << "The one-rank reference is launched in this process, which requires thread-MPI.";
#endif

    ASSERT_EQ(0, prepareSimulation());
    setOutputTag("reference");
    ASSERT_EQ(0, runMdrun(1, std::nullopt)) << tailOfFile(runner_.logFileName_);
    const std::string referenceEdr = runner_.edrFileName_;
    const std::string referenceTrr = runner_.fullPrecisionTrajectoryFileName_;

    std::optional<std::array<int, 3>> ddGrid;
    if (numRanks == 2)
    {
        ddGrid = std::array<int, 3>{ 2, 1, 1 };
    }
    else if (numRanks == 4)
    {
        ddGrid = std::array<int, 3>{ 2, 2, 1 };
    }
    setOutputTag("parallel");
    ASSERT_EQ(0, runMdrun(numRanks, ddGrid)) << tailOfFile(runner_.logFileName_);

    const std::string metatomicName =
            interaction_function[InteractionFunction::MetatomicPotentialEnergy].longname;
    EnergyTermsToCompare energiesToCompare{ {
            { metatomicName, relativeToleranceAsFloatingPoint(c_energyMagnitude, c_energyRelative) },
    } };
    compareEnergies(
            referenceEdr, runner_.edrFileName_, energiesToCompare, MaxNumFrames::compareAllFrames());

    TrajectoryFrameMatchSettings match;
    match.coordinatesComparison     = ComparisonConditions::NoComparison;
    match.velocitiesComparison      = ComparisonConditions::NoComparison;
    match.forcesComparison          = ComparisonConditions::MustCompare;
    match.handlePbcIfPossible       = false;
    TrajectoryTolerances tolerances = TrajectoryComparison::s_defaultTrajectoryTolerances;
    tolerances.forces = relativeToleranceAsFloatingPoint(c_forceMagnitude, c_forceRelative);
    compareTrajectories(referenceTrr,
                        runner_.fullPrecisionTrajectoryFileName_,
                        TrajectoryComparison(match, tolerances));
}

} // namespace
} // namespace test
} // namespace gmx
