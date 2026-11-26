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
 * Tests the metatomic force provider against an analytic pair model.
 *
 * The model is the Lennard-Jones model of metatomic-core, loaded from the plugin
 * installed with libmetatomic, with its parameters in a length and energy unit
 * chosen per test. Energies, forces and virials are compared with the same
 * potential evaluated here in GROMACS units. The small model of testmodel.h
 * covers the inputs a model requests, the gradients GROMACS asks for, the
 * non-conservative force and stress, and the energy-uncertainty warning.
 *
 * \author Eric Boittier <eric.boittier@epfl.ch>
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "gromacs/applied_forces/metatomic/metatomic_forceprovider.h"

#include "config.h"

#include <cmath>

#include <array>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "gromacs/applied_forces/metatomic/metatomic_cellshift.h"
#include "gromacs/applied_forces/metatomic/metatomic_options.h"
#include "gromacs/domdec/localatomset.h"
#include "gromacs/domdec/localatomsetmanager.h"
#include "gromacs/mdrunutility/mdmodulesnotifiers.h"
#include "gromacs/mdrunutility/plainpairlistranges.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/mdtypes/iforceprovider.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/topology/atoms.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/loggerbuilder.h"
#include "gromacs/utility/mpicomm.h"
#include "gromacs/utility/stringstream.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/vec.h"

#include "testutils/testasserts.h"

#if GMX_METATOMIC_ACTIVE
#    include "linkcaps.h"
#    include "testmodel.h"
#endif

namespace gmx
{
namespace test
{
namespace
{

using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::ThrowsMessage;

const matrix c_triclinicBox = { { 3, 0, 0 }, { 1.2, 3, 0 }, { -0.9, 1.1, 3 } };

TEST(MetatomicCellShift, FindsTheMinimumImageInTriclinicBoxes)
{
    const RVec small(0.11, -0.07, 0.05);
    for (const IVec& n : { IVec(0, 0, 0), IVec(1, 0, 0), IVec(-1, 1, 0), IVec(1, -1, 1), IVec(0, 1, -1) })
    {
        const RVec d = small + cellShiftVector(c_triclinicBox, n);
        EXPECT_EQ(minimumImageCellShift(c_triclinicBox, { 1, 1, 1 }, d), IVec(-n[XX], -n[YY], -n[ZZ]));
    }
}

TEST(MetatomicCellShift, ShiftsAlongPeriodicDirectionsOnly)
{
    const RVec d(0.1, 2.9, 2.9);
    // pbc = xyz would also shift along c
    EXPECT_EQ(minimumImageCellShift(c_triclinicBox, { 1, 1, 1 }, d)[ZZ], -1);
    // pbc = xy: in the plane only, z untouched
    const IVec inPlane = minimumImageCellShift(c_triclinicBox, { 1, 1, 0 }, d);
    EXPECT_EQ(inPlane, IVec(0, -1, 0));
    EXPECT_FLOAT_EQ((d + cellShiftVector(c_triclinicBox, inPlane))[ZZ], d[ZZ]);
    // pbc = no: never, and a zero box is not divided by
    const matrix zeroBox = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };
    EXPECT_EQ(minimumImageCellShift(zeroBox, { 0, 0, 0 }, d), IVec(0, 0, 0));
}

#if GMX_METATOMIC_ACTIVE

//! Relative tolerance: pair vectors are built from `real` positions.
constexpr double c_tolerance = GMX_DOUBLE ? 1e-8 : 1e-4;

//! Lennard-Jones parameters in GROMACS units: nm and kJ/mol.
constexpr double c_sigma   = 0.34;
constexpr double c_epsilon = 0.99;
constexpr double c_cutoff  = 0.9;
//! Argon, the type of the test atoms.
constexpr int c_atomicNumber = 18;

//! Two argon atoms evaluated by the force provider, with the recorder model of testmodel.h.
class MetatomicForceProviderTest : public ::testing::Test
{
public:
    MetatomicForceProviderTest()
    {
        EXPECT_TRUE(registerRecorderPlugin());
        init_t_atoms(&options_.params_.atoms_, 2, false);
        for (int i = 0; i < 2; ++i)
        {
            options_.params_.atoms_.atom[i].atomnumber = c_atomicNumber;
        }
        options_.params_.active      = true;
        options_.params_.modelPath_  = "gromacs-test-recorder-model";
        options_.params_.mtaIndices_ = { 0, 1 };
        options_.params_.numAtoms_   = 2;
        options_.params_.mtaAtoms_   = std::make_unique<LocalAtomSet>(
                atomSets_.add(ArrayRef<const Index>(options_.params_.mtaIndices_)));
        options_.params_.pbcType_ = std::make_unique<PbcType>(PbcType::Xyz);
    }
    ~MetatomicForceProviderTest() override { done_atom(&options_.params_.atoms_); }

    /*! \brief Evaluate the pair (0, 1) with metatensor cell shift \p cellShift
     *
     * \p x1 is chosen such that x1 + cellShift . box - x0 = \p distance. The
     * warnings logged during the evaluation go to evaluationWarnings_.
     */
    void evaluate(const matrix box, const RVec& distance, const IVec& cellShift, bool computeVirial = true)
    {
        const DVec shift(cellShift[XX] * box[XX][XX] + cellShift[YY] * box[YY][XX]
                                 + cellShift[ZZ] * box[ZZ][XX],
                         cellShift[YY] * box[YY][YY] + cellShift[ZZ] * box[ZZ][YY],
                         cellShift[ZZ] * box[ZZ][ZZ]);
        x_ = { RVec(0.3, 0.4, 0.5), RVec(0.3, 0.4, 0.5) + distance - shift.toRVec() };
        // the pair vector the model sees, up to rounding inside the force provider
        distance_ = x_[1].toDVec() + shift - x_[0].toDVec();

        StringOutputStream warnings;
        LoggerBuilder      builder;
        builder.addTargetStream(MDLogger::LogLevel::Warning, &warnings);
        const LoggerOwner      logger = builder.build();
        MetatomicForceProvider provider(options_, logger.logger(), mpiComm_);
        provider.gatherAtomNumbersIndices(MDModulesAtomsRedistributedSignal(box, x_, {}, {}, std::nullopt));
        // GROMACS shifts the first atom of a pair, metatensor the second
        const std::vector<MDModulesPairlistConstructedSignal::PairlistEntry> pairlist = {
            { { 0, 1 }, xyzToShiftIndex(-cellShift[XX], -cellShift[YY], -cellShift[ZZ]) }
        };
        provider.setPairlist(MDModulesPairlistConstructedSignal(pairlist, {}, {}));

        forces_.assign(2, RVec(0, 0, 0));
        ForceWithVirial     forceWithVirial(forces_, computeVirial);
        gmx_enerdata_t      enerd(1, nullptr);
        ForceProviderOutput output(&forceWithVirial, &enerd);
        provider.calculateForces(ForceProviderInput(x_, 2, {}, {}, 0.0, 0, box, mpiComm_, nullptr), &output);
        energy_ = enerd.term[InteractionFunction::MetatomicPotentialEnergy];
        copy_mat(forceWithVirial.getVirial(), virial_);
        evaluationWarnings_ = warnings.toString();
    }

    /*! \brief The warnings logged while the model is loaded.
     *
     * They include one for each installed plugin that fails to load, such as a
     * metatomic-torch plugin without libtorch.
     */
    std::string setupWarnings()
    {
        StringOutputStream warnings;
        LoggerBuilder      builder;
        builder.addTargetStream(MDLogger::LogLevel::Warning, &warnings);
        const LoggerOwner      logger = builder.build();
        MetatomicForceProvider provider(options_, logger.logger(), mpiComm_);
        return warnings.toString();
    }

protected:
    MetatomicOptions    options_;
    LocalAtomSetManager atomSets_;
    MDLogger            logger_;
    const MpiComm       mpiComm_ = MpiComm(MpiComm::SingleRank{});

    std::vector<RVec> x_;
    DVec              distance_;
    std::vector<RVec> forces_;
    real              energy_ = 0;
    matrix            virial_;
    std::string       evaluationWarnings_;
};

/*! \brief The same atoms with the Lennard-Jones model of metatomic-core.
 *
 * metatomic-core installs its plugin next to libmetatomic, where an empty metatomic-plugin
 * finds it. The tests skip when it is not installed; the CI checks that it is.
 */
class MetatomicLennardJonesTest : public MetatomicForceProviderTest
{
public:
    void SetUp() override
    {
        const auto plugin =
                std::filesystem::path(GMX_METATOMIC_PLUGINS_DIR) / "metatomic-lj-plugin.so";
        if (!std::filesystem::exists(plugin))
        {
            GTEST_SKIP() << plugin << " is not installed";
        }
        options_.params_.pluginPath = "";
        options_.params_.modelPath_ = "metatomic-lj-model";
    }

    //! Declare the parameters of the model in \p lengthUnit and \p energyUnit.
    void setModelUnits(const std::string& lengthUnit, const std::string& energyUnit)
    {
        const double length           = metatomic::unit_conversion_factor("nm", lengthUnit);
        const double energy           = metatomic::unit_conversion_factor("kJ/mol", energyUnit);
        options_.params_.modelOptions = formatString(
                "sigma=%.17g epsilon=%.17g cutoff=%.17g atomic_type=%d length_unit=%s "
                "energy_unit=%s",
                c_sigma * length,
                c_epsilon * energy,
                c_cutoff * length,
                c_atomicNumber,
                lengthUnit.c_str(),
                energyUnit.c_str());
    }

    //! Compare with the Lennard-Jones energy, forces and virial (zero without) of the last evaluation.
    void checkAgainstLennardJones(bool withVirial = true) const
    {
        const double r   = norm(distance_);
        const double sr6 = std::pow(c_sigma / r, 6);
        // the energy of metatomic-core's model is shifted to zero at the cutoff
        const double cr6    = std::pow(c_sigma / c_cutoff, 6);
        const double energy = 4 * c_epsilon * (sr6 * sr6 - sr6) - 4 * c_epsilon * (cr6 * cr6 - cr6);
        const double dEdr   = 4 * c_epsilon * (-12 * sr6 * sr6 + 6 * sr6) / r;

        EXPECT_NEAR(energy, energy_, c_tolerance * std::abs(energy));
        for (int a = 0; a < 3; ++a)
        {
            // F_1 = -dE/dr d/r in kJ/mol/nm, and F_0 = -F_1
            const double force = -dEdr * distance_[a] / r;
            EXPECT_NEAR(force, forces_[1][a], c_tolerance * std::abs(dEdr)) << "atom 1, dimension " << a;
            EXPECT_NEAR(-force, forces_[0][a], c_tolerance * std::abs(dEdr)) << "atom 0, dimension " << a;
            for (int b = 0; b < 3; ++b)
            {
                // -0.5 sum_i x_i (x) F_i = -0.5 d (x) F_1
                const double virial = withVirial ? -0.5 * distance_[a] * -dEdr * distance_[b] / r : 0.0;
                EXPECT_NEAR(virial, virial_[a][b], c_tolerance * std::abs(dEdr * r))
                        << "virial " << a << ", " << b;
            }
        }
    }
};

const matrix c_cubicBox = { { 3, 0, 0 }, { 0, 3, 0 }, { 0, 0, 3 } };

TEST_F(MetatomicLennardJonesTest, ModelInNanometre)
{
    setModelUnits("nm", "kJ/mol");
    evaluate(c_cubicBox, RVec(0.21, 0.24, 0.17), IVec(0, 0, 0));
    checkAgainstLennardJones();
}

// metatomic converts model gradients from eV/Angstrom to kJ/mol/nm.
TEST_F(MetatomicLennardJonesTest, ModelInAngstromAndElectronvolt)
{
    setModelUnits("Angstrom", "eV");
    evaluate(c_cubicBox, RVec(0.21, 0.24, 0.17), IVec(0, 0, 0));
    checkAgainstLennardJones();
}

TEST_F(MetatomicLennardJonesTest, ChecksConsistencyOfInputsAndOutputs)
{
    setModelUnits("Angstrom", "eV");
    options_.params_.checkConsistency = true;
    evaluate(c_cubicBox, RVec(0.21, 0.24, 0.17), IVec(0, 0, 0));
    checkAgainstLennardJones();
}

TEST_F(MetatomicLennardJonesTest, PairAcrossTriclinicBoundary)
{
    // The lattice shift of a pair is n_a a + n_b b + n_c c, which differs from box . n
    // as soon as the box is not rectangular.
    const matrix box = { { 3, 0, 0 }, { 1.2, 3, 0 }, { -0.9, 1.1, 3 } };
    setModelUnits("Angstrom", "eV");
    evaluate(box, RVec(0.21, -0.24, 0.17), IVec(1, -1, 1));
    checkAgainstLennardJones();
}

TEST_F(MetatomicLennardJonesTest, NonperiodicSystemWithBox)
{
    setModelUnits("nm", "kJ/mol");
    options_.params_.pbcType_         = std::make_unique<PbcType>(PbcType::No);
    options_.params_.checkConsistency = true;
    evaluate(c_cubicBox, RVec(0.21, 0.24, 0.17), IVec(0, 0, 0));
    checkAgainstLennardJones();
}

TEST_F(MetatomicLennardJonesTest, TwoDimensionalPeriodicity)
{
    setModelUnits("Angstrom", "eV");
    options_.params_.pbcType_         = std::make_unique<PbcType>(PbcType::XY);
    options_.params_.checkConsistency = true;
    evaluate(c_cubicBox, RVec(0.21, 0.24, 0.17), IVec(1, -1, 0));
    checkAgainstLennardJones();
}

TEST_F(MetatomicLennardJonesTest, ForcesWithoutTheVirial)
{
    setModelUnits("nm", "kJ/mol");
    evaluate(c_cubicBox, RVec(0.21, 0.24, 0.17), IVec(0, 0, 0), false);
    checkAgainstLennardJones(false);
}

TEST_F(MetatomicLennardJonesTest, PlainPairlistRange)
{
    gmx_mtop_t mtop;
    t_inputrec inputrec;
    inputrec.eI    = IntegrationAlgorithm::Steep; // no displacement between searches
    inputrec.rlist = 1.2;
    const PlainPairlistRanges ranges(mtop, inputrec);

    // the pair-list cutoff with a buffer of 0.1 nm
    setModelUnits("Angstrom", "eV");
    EXPECT_REAL_EQ_TOL(MetatomicForceProvider(options_, logger_, mpiComm_).plainPairlistRange(ranges),
                       c_cutoff + 0.1,
                       relativeToleranceAsFloatingPoint(1.0, 1e-6));
    // a model without pair lists needs no plain pairlist
    options_.params_.pluginPath   = "";
    options_.params_.modelPath_   = "gromacs-test-recorder-model";
    options_.params_.modelOptions = "";
    EXPECT_EQ(MetatomicForceProvider(options_, logger_, mpiComm_).plainPairlistRange(ranges), 0);
}

TEST_F(MetatomicLennardJonesTest, PlainPairlistRangeEndsAtRlist)
{
    setModelUnits("nm", "kJ/mol");
    gmx_mtop_t mtop;
    t_inputrec inputrec;
    inputrec.ensembleTemperatureSetting = EnsembleTemperatureSetting::NotAvailable;
    // the range and the warnings for rlist \p rlist, in dynamics without an ensemble
    // temperature (a buffer of 0.1 nm) or in energy minimization (no buffer needed)
    const auto range = [&](real rlist, IntegrationAlgorithm integrator)
    {
        inputrec.eI    = integrator;
        inputrec.rlist = rlist;
        StringOutputStream warnings;
        LoggerBuilder      builder;
        builder.addTargetStream(MDLogger::LogLevel::Warning, &warnings);
        const LoggerOwner logger = builder.build();
        const real        value  = MetatomicForceProvider(options_, logger.logger(), mpiComm_)
                                   .plainPairlistRange(PlainPairlistRanges(mtop, inputrec));
        return std::make_pair(value, warnings.toString());
    };

    // the model cutoff of 0.9 nm beyond rlist
    EXPECT_THAT([&] { range(0.8, IntegrationAlgorithm::MD); },
                ThrowsMessage<InconsistentInputError>(HasSubstr(
                        "The metatomic model reads pairs up to 0.900 nm, but the GROMACS pair "
                        "list only reaches rlist = 0.800 nm. Raise rlist to at least 1.000 nm")));

    // rlist leaves room for a buffer of 0.02 nm only
    const auto [shortRange, warnings] = range(0.92, IntegrationAlgorithm::MD);
    EXPECT_REAL_EQ_TOL(shortRange, 0.92, relativeToleranceAsFloatingPoint(1.0, 1e-6));
    EXPECT_THAT(warnings,
                HasSubstr("the buffer of the pair list is 0.020 nm, less than the 0.100 nm the "
                          "atoms can move between pair searches, because rlist is 0.920 nm"));

    // energy minimization needs no buffer, and a full buffer gives no warning
    EXPECT_THAT(range(0.92, IntegrationAlgorithm::Steep).second, Not(HasSubstr("buffer")));
    EXPECT_THAT(range(1.2, IntegrationAlgorithm::MD).second, Not(HasSubstr("buffer")));

    // rlist set to the cutoff is enough, although 0.9 rounds down in single precision
    EXPECT_REAL_EQ_TOL(range(0.9, IntegrationAlgorithm::Steep).first,
                       0.9,
                       relativeToleranceAsFloatingPoint(1.0, 1e-6));
}

TEST_F(MetatomicForceProviderTest, RequestsTheStrainGradientForTheVirialOnly)
{
    const int strainRequests = RecorderModel::strainRequests;
    evaluate(c_cubicBox, RVec(0.21, 0.24, 0.17), IVec(0, 0, 0), false);
    EXPECT_EQ(RecorderModel::strainRequests, strainRequests);
    evaluate(c_cubicBox, RVec(0.21, 0.24, 0.17), IVec(0, 0, 0), true);
    EXPECT_EQ(RecorderModel::strainRequests, strainRequests + 1);
}

TEST_F(MetatomicForceProviderTest, WarnsThatAModelWithoutStrainGradientAddsNoVirial)
{
    EXPECT_THAT(setupWarnings(), Not(HasSubstr("adds zero to the virial")));
    options_.params_.modelOptions = R"(without_strain="")";
    EXPECT_THAT(setupWarnings(), HasSubstr("strain gradient of 'energy', so it adds zero to the virial"));
}

TEST_F(MetatomicForceProviderTest, NeverAsksForAStrainGradientTheModelLacks)
{
    options_.params_.modelOptions = R"(without_strain="")";
    const int strainRequests      = RecorderModel::strainRequests;
    evaluate(c_cubicBox, RVec(0.21, 0.24, 0.17), IVec(0, 0, 0), true);
    EXPECT_EQ(RecorderModel::strainRequests, strainRequests);
}

TEST_F(MetatomicForceProviderTest, PressureCouplingNeedsTheStrainGradient)
{
    options_.params_.pressureCoupling = true;
    EXPECT_NO_THROW(MetatomicForceProvider provider(options_, logger_, mpiComm_));
    options_.params_.modelOptions = R"(without_strain="")";
    EXPECT_THROW_GMX(MetatomicForceProvider provider(options_, logger_, mpiComm_), InconsistentInputError);
}

TEST_F(MetatomicForceProviderTest, AppliesNonConservativeForcesAndStress)
{
    options_.params_.nonConservative  = true;
    options_.params_.modelOptions     = R"(non_conservative="")";
    options_.params_.checkConsistency = true;
    const double forceFactor =
            metatomic::unit_conversion_factor(RecorderModel::c_ncForceUnit, "kJ/mol/nm");
    const double stressFactor =
            metatomic::unit_conversion_factor(RecorderModel::c_ncStressUnit, "kJ/mol/nm^3");
    const double volume = det(c_cubicBox);

    for (const bool computeVirial : { false, true })
    {
        evaluate(c_cubicBox, RVec(0.21, 0.24, 0.17), IVec(0, 0, 0), computeVirial);
        for (size_t atom = 0; atom < 2; ++atom)
        {
            for (size_t d = 0; d < 3; ++d)
            {
                const double force = forceFactor * RecorderModel::ncForce(atom, d);
                EXPECT_NEAR(force, forces_[atom][d], c_tolerance * std::abs(force))
                        << "atom " << atom << ", dimension " << d;
            }
        }
        // the virial is half the stress times the volume, zero when not asked for
        for (size_t a = 0; a < 3; ++a)
        {
            for (size_t b = 0; b < 3; ++b)
            {
                const double virial =
                        computeVirial ? 0.5 * volume * stressFactor * RecorderModel::ncStress(a, b) : 0.0;
                EXPECT_NEAR(virial, virial_[a][b], c_tolerance * std::abs(virial) + 1e-12)
                        << "virial " << a << ", " << b << " with computeVirial " << computeVirial;
            }
        }
    }
}

TEST_F(MetatomicForceProviderTest, NonConservativeModeNeedsTheNonConservativeForce)
{
    options_.params_.nonConservative = true;
    EXPECT_THAT([&] { MetatomicForceProvider provider(options_, logger_, mpiComm_); },
                ThrowsMessage<InconsistentInputError>(
                        HasSubstr("does not provide a 'non_conservative_force' output")));
}

TEST_F(MetatomicForceProviderTest, WarnsAboutAtomsAboveTheUncertaintyThreshold)
{
    // atom 0 has an uncertainty of 6 kJ/mol and atom 1 of 12 kJ/mol, around the threshold of 10
    options_.params_.uncertaintyThreshold = "10";
    options_.params_.modelOptions         = R"(uncertainty="6")";
    options_.params_.checkConsistency     = true;
    evaluate(c_cubicBox, RVec(0.21, 0.24, 0.17), IVec(0, 0, 0));
    EXPECT_THAT(evaluationWarnings_,
                HasSubstr("uncertainty on atomic energies for 1 atoms is larger than the threshold "
                          "of 10.0000 kJ/mol"));

    options_.params_.modelOptions = R"(uncertainty="4")";
    evaluate(c_cubicBox, RVec(0.21, 0.24, 0.17), IVec(0, 0, 0));
    EXPECT_THAT(evaluationWarnings_, Not(HasSubstr("uncertainty on atomic energies")));

    options_.params_.modelOptions         = R"(uncertainty="6")";
    options_.params_.uncertaintyThreshold = "off";
    evaluate(c_cubicBox, RVec(0.21, 0.24, 0.17), IVec(0, 0, 0));
    EXPECT_THAT(evaluationWarnings_, Not(HasSubstr("uncertainty on atomic energies")));
}

TEST_F(MetatomicForceProviderTest, RejectsScrewPeriodicity)
{
    options_.params_.pbcType_ = std::make_unique<PbcType>(PbcType::Screw);
    EXPECT_THROW_GMX(MetatomicForceProvider provider(options_, logger_, mpiComm_), NotImplementedError);
}

TEST_F(MetatomicForceProviderTest, RejectsInputsGromacsCannotProvide)
{
    options_.params_.modelOptions = R"(request_charges="")";
    EXPECT_THROW_GMX(MetatomicForceProvider provider(options_, logger_, mpiComm_), InconsistentInputError);
}

TEST_F(MetatomicForceProviderTest, FailsOnUnknownModel)
{
    options_.params_.modelPath_ = "no-such-model";
    EXPECT_THROW_GMX(MetatomicForceProvider provider(options_, logger_, mpiComm_), InvalidInputError);
}

TEST_F(MetatomicForceProviderTest, RejectsAtomicNumbersTheModelDoesNotSupport)
{
    options_.params_.atoms_.atom[1].atomnumber = 8;
    EXPECT_THAT([&] { evaluate(c_cubicBox, RVec(0.21, 0.24, 0.17), IVec(0, 0, 0)); },
                ThrowsMessage<InvalidInputError>(HasSubstr(
                        "The metatomic model failed: invalid parameter: this model does not "
                        "support the following atomic types which are present in the input "
                        "systems: 8")));
}

/*! \brief Link caps on the atoms of linkcaps.h, with the Lennard-Jones model.
 *
 * The force provider gets the pairs of all atoms, as from the GROMACS pair list. The pairs of
 * the MM atoms are not those of their caps, and only the caps count.
 */
TEST(MetatomicLinkCapTest, SpreadsTheForcesOnTheCapsAndAddsTheirVirial)
{
    const auto plugin = std::filesystem::path(GMX_METATOMIC_PLUGINS_DIR) / "metatomic-lj-plugin.so";
    if (!std::filesystem::exists(plugin))
    {
        GTEST_SKIP() << plugin << " is not installed";
    }
    for (const auto& system : { LinkCapSystem(), LinkCapSystem::withANeighbourBeyondTheRangeOfTheMMAtom() })
    {
        SCOPED_TRACE(formatString("%zu atoms", system.positions.size()));
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
        ASSERT_EQ(options.params_.numEmbeddedAtoms(), system.numEmbedded);

        matrix            box;
        std::vector<RVec> x;
        for (int a = 0; a < DIM; ++a)
        {
            for (int b = 0; b < DIM; ++b)
            {
                box[a][b] = static_cast<real>(system.box[a][b]);
            }
        }
        for (const auto& position : system.positions)
        {
            x.push_back(position.toRVec());
        }

        const MDLogger         logger;
        const MpiComm          mpiComm(MpiComm::SingleRank{});
        MetatomicForceProvider provider(options, logger, mpiComm);
        provider.gatherAtomNumbersIndices(MDModulesAtomsRedistributedSignal(box, x, {}, {}, std::nullopt));
        std::vector<MDModulesPairlistConstructedSignal::PairlistEntry> pairlist;
        for (int i = 0; i < numAtoms; ++i)
        {
            for (int j = i + 1; j < numAtoms; ++j)
            {
                // GROMACS shifts the first atom of a pair, metatensor the second
                const IVec n = minimumImageCellShift(box, { 1, 1, 1 }, x[j] - x[i]);
                pairlist.push_back({ { i, j }, xyzToShiftIndex(-n[XX], -n[YY], -n[ZZ]) });
            }
        }
        provider.setPairlist(MDModulesPairlistConstructedSignal(pairlist, {}, {}));

        std::vector<RVec>   forces(numAtoms, RVec(0, 0, 0));
        ForceWithVirial     forceWithVirial(forces, true);
        gmx_enerdata_t      enerd(1, nullptr);
        ForceProviderOutput output(&forceWithVirial, &enerd);
        provider.calculateForces(
                ForceProviderInput(x, numAtoms, {}, {}, 0.0, 0, box, mpiComm, nullptr), &output);

        const double energy = system.energy();
        EXPECT_NEAR(enerd.term[InteractionFunction::MetatomicPotentialEnergy],
                    energy,
                    c_tolerance * std::abs(energy));
        const auto expectedForces = system.forces();
        for (int i = 0; i < numAtoms; ++i)
        {
            for (int d = 0; d < DIM; ++d)
            {
                EXPECT_NEAR(forces[i][d], expectedForces[i][d], 1e-3 * (1 + std::abs(expectedForces[i][d])))
                        << "atom " << i << ", dimension " << d;
            }
        }
        const auto expectedVirial = system.virial();
        for (int a = 0; a < DIM; ++a)
        {
            for (int b = 0; b < DIM; ++b)
            {
                EXPECT_NEAR(forceWithVirial.getVirial()[a][b],
                            expectedVirial[a][b],
                            1e-3 * (1 + std::abs(expectedVirial[a][b])))
                        << "virial " << a << ", " << b;
            }
        }
        done_atom(&options.params_.atoms_);
    }
}

TEST(MetatomicLinkCapTest, RejectsCappedBondsLongerThanItSupports)
{
    auto system = LinkCapSystem::withANeighbourBeyondTheRangeOfTheMMAtom();
    // the MM atom 0.35 nm from its embedded atom
    system.positions[2][XX]      = 2.1;
    const int           numAtoms = static_cast<int>(system.positions.size());
    MetatomicOptions    options;
    LocalAtomSetManager atomSets;
    init_t_atoms(&options.params_.atoms_, numAtoms, false);
    for (int i = 0; i < numAtoms; ++i)
    {
        options.params_.atoms_.atom[i].atomnumber = 18;
        options.params_.mtaIndices_.push_back(i);
    }
    ASSERT_TRUE(registerRecorderPlugin());
    options.params_.active     = true;
    options.params_.modelPath_ = "gromacs-test-recorder-model";
    options.params_.numAtoms_  = numAtoms;
    options.params_.mtaAtoms_  = std::make_unique<LocalAtomSet>(
            atomSets.add(ArrayRef<const Index>(options.params_.mtaIndices_)));
    options.params_.pbcType_      = std::make_unique<PbcType>(PbcType::Xyz);
    options.params_.linkFrontier_ = system.linkFrontier();
    const matrix      box         = { { 4, 0, 0 }, { 0, 4, 0 }, { 0, 0, 4 } };
    std::vector<RVec> x;
    for (const auto& position : system.positions)
    {
        x.push_back(position.toRVec());
    }

    const MDLogger         logger;
    const MpiComm          mpiComm(MpiComm::SingleRank{});
    MetatomicForceProvider provider(options, logger, mpiComm);
    provider.gatherAtomNumbersIndices(MDModulesAtomsRedistributedSignal(box, x, {}, {}, std::nullopt));
    std::vector<RVec>   forces(numAtoms, RVec(0, 0, 0));
    ForceWithVirial     forceWithVirial(forces, false);
    gmx_enerdata_t      enerd(1, nullptr);
    ForceProviderOutput output(&forceWithVirial, &enerd);
    EXPECT_THAT(
            [&]
            {
                provider.calculateForces(
                        ForceProviderInput(x, numAtoms, {}, {}, 0.0, 0, box, mpiComm, nullptr), &output);
            },
            ThrowsMessage<InconsistentInputError>(
                    HasSubstr("The bond between atoms 1 and 3, capped by a link atom, is 0.35 nm "
                              "long, longer than 0.3 nm")));
    done_atom(&options.params_.atoms_);
}

#endif // GMX_METATOMIC_ACTIVE

} // namespace
} // namespace test
} // namespace gmx
