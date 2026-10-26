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
// Shared analytic model for unit tests and the dynamically loaded smoke-test plugin.
#ifndef GMX_METATOMIC_TESTMODEL_H
#define GMX_METATOMIC_TESTMODEL_H
#include <array>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <metatomic.hpp>
#include <nlohmann/json.hpp>
namespace gmx::test {
//! Lennard-Jones parameters in GROMACS units: nm and kJ/mol.
constexpr double c_sigma   = 0.34;
constexpr double c_epsilon = 0.99;
constexpr double c_cutoff  = 0.9;
//! Argon, the only type the model accepts.
constexpr int c_atomicNumber = 18;

/*! \brief Lennard-Jones pair model with explicit positions and strain gradients.
 *
 * Built with the length and energy units given as model options, so the same
 * physical potential can be declared in nm or in Angstrom. With selected atoms
 * (domain decomposition), the energy is that of the selected atoms: half of each
 * pair for each selected end, as metatomic-core's Lennard-Jones plugin does.
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
        if (systems.size() != 1)
        {
            throw metatomic::Error("the test model takes one system");
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
            const auto i      = static_cast<size_t>(samples(pair, 0));
            const auto j      = static_cast<size_t>(samples(pair, 1));
            const auto weight = [&]
            {
                if (!selectedAtoms.has_value())
                {
                    return 1.0;
                }
                const auto selected = [&](size_t atom)
                { return selectedAtoms->position({ 0, static_cast<int32_t>(atom) }).has_value(); };
                return 0.5 * (selected(i) + selected(j));
            }();
            if (weight == 0.0)
            {
                continue;
            }
            const double sr6 = std::pow(sigma / r, 6);
            energy += weight * 4 * epsilon * (sr6 * sr6 - sr6);
            const double dEdr = weight * 4 * epsilon * (-12 * sr6 * sr6 + 6 * sr6) / r;
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
inline mta_status_t loadPairModel(const char* loadFrom, const char* optionsJson, mta_model_t* model)
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


} // namespace gmx::test
#endif
