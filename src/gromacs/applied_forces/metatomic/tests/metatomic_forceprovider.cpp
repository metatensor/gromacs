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
 * The model is a Lennard-Jones pair potential registered as an in-process
 * metatomic plugin, with positions and strain gradients, in a length unit
 * chosen per test. Energies, forces and virials are compared with the same
 * potential evaluated here in GROMACS units.
 *
 * \author Metatensor developers <https://github.com/metatensor>
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "config.h"

#include <cmath>

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/applied_forces/metatomic/metatomic_cellshift.h"
#include "gromacs/applied_forces/metatomic/metatomic_forceprovider.h"
#include "gromacs/applied_forces/metatomic/metatomic_options.h"
#include "gromacs/domdec/localatomset.h"
#include "gromacs/domdec/localatomsetmanager.h"
#include "gromacs/utility/vec.h"
#include "gromacs/mdrunutility/mdmodulesnotifiers.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/mdtypes/iforceprovider.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/topology/atoms.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/mpicomm.h"

#include "testutils/testasserts.h"

#if GMX_METATOMIC_ACTIVE
#    ifdef DIM
#        undef DIM
#    endif
#    include "testmodel.h"
#endif

namespace gmx
{
namespace test
{
namespace
{

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

//! Register the in-process plugin once per test binary.
void registerTestPlugin()
{
    static const bool registered = []
    {
        mta_plugin_t plugin{ MTA_ABI_VERSION, "gromacs-test", loadPairModel };
        return mta_register_plugin(plugin) == MTA_SUCCESS;
    }();
    ASSERT_TRUE(registered);
}

//! Two-atom system evaluated by the force provider.
class MetatomicForceProviderTest : public ::testing::Test
{
public:
    MetatomicForceProviderTest()
    {
        registerTestPlugin();
        init_t_atoms(&options_.params_.atoms_, 2, false);
        for (int i = 0; i < 2; ++i)
        {
            options_.params_.atoms_.atom[i].atomnumber = c_atomicNumber;
        }
        options_.params_.active      = true;
        options_.params_.modelPath_  = "gromacs-test-pair-model";
        options_.params_.mtaIndices_ = { 0, 1 };
        options_.params_.numAtoms_   = 2;
        options_.params_.mtaAtoms_   = std::make_unique<LocalAtomSet>(
                atomSets_.add(ArrayRef<const Index>(options_.params_.mtaIndices_)));
        options_.params_.pbcType_ = std::make_unique<PbcType>(PbcType::Xyz);
    }
    ~MetatomicForceProviderTest() override { done_atom(&options_.params_.atoms_); }

    void setModelUnits(const std::string& lengthUnit, const std::string& energyUnit)
    {
        options_.params_.modelOptions =
                R"({"length_unit": ")" + lengthUnit + R"(", "energy_unit": ")" + energyUnit + R"("})";
    }

    /*! \brief Evaluate the pair (0, 1) with metatensor cell shift \p cellShift
     *
     * \p x1 is chosen such that x1 + cellShift . box - x0 = \p distance.
     */
    void evaluate(const matrix box, const RVec& distance, const IVec& cellShift)
    {
        const DVec shift(cellShift[XX] * box[XX][XX] + cellShift[YY] * box[YY][XX] + cellShift[ZZ] * box[ZZ][XX],
                         cellShift[YY] * box[YY][YY] + cellShift[ZZ] * box[ZZ][YY],
                         cellShift[ZZ] * box[ZZ][ZZ]);
        x_ = { RVec(0.3, 0.4, 0.5), RVec(0.3, 0.4, 0.5) + distance - shift.toRVec() };
        // the pair vector the model sees, up to rounding inside the force provider
        distance_ = x_[1].toDVec() + shift - x_[0].toDVec();

        MetatomicForceProvider provider(options_, logger_, mpiComm_);
        provider.gatherAtomNumbersIndices(MDModulesAtomsRedistributedSignal(box, x_, {}, {}, std::nullopt));
        // GROMACS shifts the first atom of a pair, metatensor the second
        const std::vector<MDModulesPairlistConstructedSignal::PairlistEntry> pairlist = {
            { { 0, 1 }, xyzToShiftIndex(-cellShift[XX], -cellShift[YY], -cellShift[ZZ]) }
        };
        provider.setPairlist(MDModulesPairlistConstructedSignal(pairlist, {}, {}));

        forces_.assign(2, RVec(0, 0, 0));
        ForceWithVirial     forceWithVirial(forces_, true);
        gmx_enerdata_t      enerd(1, nullptr);
        ForceProviderOutput output(&forceWithVirial, &enerd);
        provider.calculateForces(ForceProviderInput(x_, 2, {}, {}, 0.0, 0, box, mpiComm_, nullptr), &output);
        energy_ = enerd.term[InteractionFunction::MetatomicPotentialEnergy];
        copy_mat(forceWithVirial.getVirial(), virial_);
    }

    //! Compare with the Lennard-Jones energy, forces and virial of the last evaluation.
    void checkAgainstLennardJones() const
    {
        const double r      = norm(distance_);
        const double sr6    = std::pow(c_sigma / r, 6);
        const double energy = 4 * c_epsilon * (sr6 * sr6 - sr6);
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
                const double virial = -0.5 * distance_[a] * -dEdr * distance_[b] / r;
                EXPECT_NEAR(virial, virial_[a][b], c_tolerance * std::abs(dEdr * r))
                        << "virial " << a << ", " << b;
            }
        }
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
};

const matrix c_cubicBox = { { 3, 0, 0 }, { 0, 3, 0 }, { 0, 0, 3 } };

TEST_F(MetatomicForceProviderTest, ModelInNanometre)
{
    setModelUnits("nm", "kJ/mol");
    evaluate(c_cubicBox, RVec(0.21, 0.24, 0.17), IVec(0, 0, 0));
    checkAgainstLennardJones();
}

/* The system is handed to metatomic in the model's length unit and the positions
 * gradient converted to kJ/mol/nm here. Converting it again in GROMACS for a system
 * passed in nm (the workaround metatensor/metatomic#369 makes wrong) would give
 * forces 10x too large with #369, and keeping the system in nm without converting
 * would give them 10x too small without it. */
TEST_F(MetatomicForceProviderTest, ModelInAngstromAndElectronvolt)
{
    setModelUnits("Angstrom", "eV");
    evaluate(c_cubicBox, RVec(0.21, 0.24, 0.17), IVec(0, 0, 0));
    checkAgainstLennardJones();
}

TEST_F(MetatomicForceProviderTest, ChecksConsistencyOfInputsAndOutputs)
{
    setModelUnits("Angstrom", "eV");
    options_.params_.checkConsistency = true;
    evaluate(c_cubicBox, RVec(0.21, 0.24, 0.17), IVec(0, 0, 0));
    checkAgainstLennardJones();
}

TEST_F(MetatomicForceProviderTest, PairAcrossTriclinicBoundary)
{
    // The lattice shift of a pair is n_a a + n_b b + n_c c, which differs from box . n
    // as soon as the box is not rectangular.
    const matrix box = { { 3, 0, 0 }, { 1.2, 3, 0 }, { -0.9, 1.1, 3 } };
    setModelUnits("Angstrom", "eV");
    evaluate(box, RVec(0.21, -0.24, 0.17), IVec(1, -1, 1));
    checkAgainstLennardJones();
}

TEST_F(MetatomicForceProviderTest, NonperiodicSystemWithBox)
{
    setModelUnits("nm", "kJ/mol");
    options_.params_.pbcType_ = std::make_unique<PbcType>(PbcType::No);
    options_.params_.checkConsistency = true;
    evaluate(c_cubicBox, RVec(0.21, 0.24, 0.17), IVec(0, 0, 0));
    checkAgainstLennardJones();
}

TEST_F(MetatomicForceProviderTest, TwoDimensionalPeriodicity)
{
    setModelUnits("Angstrom", "eV");
    options_.params_.pbcType_ = std::make_unique<PbcType>(PbcType::XY);
    options_.params_.checkConsistency = true;
    evaluate(c_cubicBox, RVec(0.21, 0.24, 0.17), IVec(1, -1, 0));
    checkAgainstLennardJones();
}

TEST_F(MetatomicForceProviderTest, RejectsScrewPeriodicity)
{
    setModelUnits("nm", "kJ/mol");
    options_.params_.pbcType_ = std::make_unique<PbcType>(PbcType::Screw);
    EXPECT_THROW_GMX(MetatomicForceProvider provider(options_, logger_, mpiComm_), NotImplementedError);
}

TEST_F(MetatomicForceProviderTest, RejectsInputsGromacsCannotProvide)
{
    options_.params_.modelOptions = R"({"length_unit": "nm", "energy_unit": "kJ/mol", "request_charges": ""})";
    EXPECT_THROW_GMX(MetatomicForceProvider provider(options_, logger_, mpiComm_), APIError);
}

TEST_F(MetatomicForceProviderTest, FailsOnUnknownModel)
{
    options_.params_.modelPath_ = "no-such-model";
    EXPECT_THROW_GMX(MetatomicForceProvider provider(options_, logger_, mpiComm_), APIError);
}

#else

TEST(MetatomicForceProviderTest, NeedsMetatomic)
{
    GTEST_SKIP() << "GROMACS was built without metatomic";
}

#endif // GMX_METATOMIC_ACTIVE

} // namespace
} // namespace test
} // namespace gmx
