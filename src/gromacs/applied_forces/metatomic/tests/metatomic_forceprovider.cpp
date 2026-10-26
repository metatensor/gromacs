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
#    include <metatomic.hpp>
#endif

namespace gmx
{
namespace test
{
namespace
{

#if GMX_METATOMIC_ACTIVE

//! Lennard-Jones parameters in GROMACS units: nm and kJ/mol.
constexpr double c_sigma   = 0.34;
constexpr double c_epsilon = 0.99;
constexpr double c_cutoff  = 0.9;
//! Argon, the only type the model accepts.
constexpr int c_atomicNumber = 18;
//! Relative tolerance: pair vectors are built from `real` positions.
constexpr double c_tolerance = GMX_DOUBLE ? 1e-8 : 1e-4;

/*! \brief Lennard-Jones pair model with explicit positions and strain gradients.
 *
 * Built with the length and energy units given as model options, so the same
 * physical potential can be declared in nm or in Angstrom.
 */
class PairModel final : public metatomic::BaseModel
{
public:
    PairModel(const std::string& lengthUnit, const std::string& energyUnit, bool requestCharges) :
        lengthUnit_(lengthUnit),
        energyUnit_(energyUnit),
        toLength_(metatomic::unit_conversion_factor("nm", lengthUnit)),
        toEnergy_(metatomic::unit_conversion_factor("kJ/mol", energyUnit)),
        requestCharges_(requestCharges),
        pairs_(metatomic::PairListOptions::builder().cutoff(c_cutoff * toLength_).full_list(false).strict(false).build())
    {
    }

    metatomic::ModelCapabilities capabilities() const override
    {
        auto energy = metatomic::Quantity::builder()
                              .name("energy")
                              .unit(energyUnit_)
                              .sample_kind(metatomic::SampleKind::System)
                              .add_gradient(metatomic::Gradients::Positions)
                              .add_gradient(metatomic::Gradients::Strain)
                              .build();
        return metatomic::ModelCapabilities::builder()
                .atomic_types({ c_atomicNumber })
                .interaction_range(c_cutoff * toLength_)
                .length_unit(lengthUnit_)
                .supported_devices({ metatomic::ModelCapabilities::Device::CPU })
                .dtype(metatomic::ModelCapabilities::DType::Float64)
                .add_output(energy)
                .build();
    }

    metatomic::ModelMetadata metadata() const override
    {
        return metatomic::ModelMetadata::builder().name("GROMACS test pair model").build();
    }

    std::vector<metatomic::PairListOptions>
    requested_pair_lists(const std::vector<metatomic::Quantity>& /*requestedOutputs*/) const override
    {
        return { pairs_ };
    }

    std::vector<metatomic::Quantity>
    requested_inputs(const std::vector<metatomic::Quantity>& /*requestedOutputs*/) const override
    {
        if (!requestCharges_)
        {
            return {};
        }
        return { metatomic::Quantity::builder()
                         .name("charge")
                         .unit("e")
                         .sample_kind(metatomic::SampleKind::Atom)
                         .build() };
    }

    std::vector<metatensor::TensorMap> execute_inner(const std::vector<metatomic::System>& systems,
                                                     std::optional<metatensor::Labels> selectedAtoms,
                                                     const std::vector<metatomic::Quantity>& /*requestedOutputs*/) override
    {
        if (systems.size() != 1 || selectedAtoms.has_value())
        {
            throw metatomic::Error("the test model takes one system and no atom selection");
        }
        const auto& system  = systems[0];
        const auto  nAtoms  = system.size();
        auto        pairs   = system.pairs(pairs_);
        const auto  samples = pairs.samples().values_cpu();
        const auto  vectors = pairs.values<double>();

        const double sigma   = c_sigma * toLength_;
        const double epsilon = c_epsilon * toEnergy_;
        const double cutoff  = c_cutoff * toLength_;

        double                         energy = 0;
        std::vector<double>            positionsGradient(3 * nAtoms, 0.0);
        std::array<double, 9>          strainGradient{};
        for (size_t pair = 0; pair < pairs.samples().count(); ++pair)
        {
            const double d[3] = { vectors(pair, 0, 0), vectors(pair, 1, 0), vectors(pair, 2, 0) };
            const double r    = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            if (r >= cutoff)
            {
                continue;
            }
            const double sr6 = std::pow(sigma / r, 6);
            energy += 4 * epsilon * (sr6 * sr6 - sr6);
            const double dEdr = 4 * epsilon * (-12 * sr6 * sr6 + 6 * sr6) / r;
            const auto   i    = static_cast<size_t>(samples(pair, 0));
            const auto   j    = static_cast<size_t>(samples(pair, 1));
            for (int a = 0; a < 3; ++a)
            {
                // d = x_j - x_i + shift
                positionsGradient[3 * j + a] += dEdr * d[a] / r;
                positionsGradient[3 * i + a] -= dEdr * d[a] / r;
                for (int b = 0; b < 3; ++b)
                {
                    strainGradient[3 * a + b] += d[a] * dEdr * d[b] / r;
                }
            }
        }

        auto block = metatensor::TensorBlock(
                std::make_unique<metatensor::SimpleDataArray<double>>(std::vector<uintptr_t>{ 1, 1 },
                                                                      std::vector<double>{ energy }),
                metatensor::Labels({ "system" }, { { 0 } }),
                {},
                metatensor::Labels({ "energy" }, { { 0 } }));

        std::vector<int32_t> gradientSamples;
        for (size_t atom = 0; atom < nAtoms; ++atom)
        {
            gradientSamples.insert(gradientSamples.end(), { 0, 0, static_cast<int32_t>(atom) });
        }
        block.add_gradient(
                "positions",
                metatensor::TensorBlock(std::make_unique<metatensor::SimpleDataArray<double>>(
                                                std::vector<uintptr_t>{ nAtoms, 3, 1 }, positionsGradient),
                                        metatensor::Labels({ "sample", "system", "atom" },
                                                           gradientSamples.data(),
                                                           nAtoms),
                                        { metatensor::Labels({ "xyz" }, { { 0 }, { 1 }, { 2 } }) },
                                        metatensor::Labels({ "energy" }, { { 0 } })));
        block.add_gradient(
                "strain",
                metatensor::TensorBlock(
                        std::make_unique<metatensor::SimpleDataArray<double>>(
                                std::vector<uintptr_t>{ 1, 3, 3, 1 },
                                std::vector<double>(strainGradient.begin(), strainGradient.end())),
                        metatensor::Labels({ "sample" }, { { 0 } }),
                        { metatensor::Labels({ "xyz_1" }, { { 0 }, { 1 }, { 2 } }),
                          metatensor::Labels({ "xyz_2" }, { { 0 }, { 1 }, { 2 } }) },
                        metatensor::Labels({ "energy" }, { { 0 } })));

        std::vector<metatensor::TensorBlock> blocks;
        blocks.push_back(std::move(block));
        std::vector<metatensor::TensorMap> outputs;
        outputs.emplace_back(metatensor::Labels({ "_" }, { { 0 } }), std::move(blocks));
        return outputs;
    }

private:
    std::string                lengthUnit_;
    std::string                energyUnit_;
    double                     toLength_;
    double                     toEnergy_;
    bool                       requestCharges_;
    metatomic::PairListOptions pairs_;
};

//! Plugin callback creating a PairModel from `{"length_unit": ..., "energy_unit": ...}`.
mta_status_t loadPairModel(const char* loadFrom, const char* optionsJson, mta_model_t* model)
{
    if (std::string(loadFrom) != "gromacs-test-pair-model")
    {
        return MTA_UNSUPPORTED_MODEL_ERROR;
    }
    try
    {
        const auto options = nlohmann::json::parse(optionsJson);
        *model             = metatomic::BaseModel::to_mta_model(
                std::make_unique<PairModel>(options.at("length_unit").get<std::string>(),
                                            options.at("energy_unit").get<std::string>(),
                                            options.contains("request_charges")));
        return MTA_SUCCESS;
    }
    catch (const std::exception& e)
    {
        mta_set_last_error(e.what(), "gromacs-test-pair-model", nullptr, nullptr);
        return MTA_MODEL_ERROR;
    }
}

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
