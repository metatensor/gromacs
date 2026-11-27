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
 * Model for the unit tests of what the Lennard-Jones model of metatomic-core does
 * not cover: the inputs a model requests, a model without pair lists, which
 * gradients GROMACS asks for, non-conservative forces and the energy uncertainty.
 *
 * \author Eric Boittier <eric.boittier@epfl.ch>
 * \ingroup module_applied_forces
 */
#ifndef GMX_METATOMIC_TESTMODEL_H
#define GMX_METATOMIC_TESTMODEL_H

#include <algorithm>
#include <memory>
#include <metatomic.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace gmx::test
{

/*! \brief Model with a zero energy per system, which declares positions and strain gradients.
 *
 * It reads no pairs, can request a charge input that GROMACS does not provide or leave
 * out the strain gradient, and counts the evaluations that ask it for the strain gradient.
 * It can also declare the non-conservative force and stress, with the values of
 * ncForce() and ncStress(), and a per-atom energy uncertainty of (atom + 1) times a
 * given value. With site inputs, it requests the total charge and spin multiplicity of each
 * system, and its energy is siteEnergy() of them.
 */
class RecorderModel final : public metatomic::BaseModel
{
public:
    RecorderModel(bool                  requestCharges,
                  bool                  declaresStrain,
                  bool                  nonConservative = false,
                  std::optional<double> uncertainty     = std::nullopt,
                  bool                  siteInputs      = false) :
        requestCharges_(requestCharges),
        declaresStrain_(declaresStrain),
        nonConservative_(nonConservative),
        uncertainty_(uncertainty),
        siteInputs_(siteInputs)
    {
    }

    //! Energy of a system of \p numAtoms atoms with total charge \p charge and spin multiplicity \p spin, in kJ/mol
    static double siteEnergy(double charge, double spin, size_t numAtoms)
    {
        return 1000 * charge + 100 * spin + numAtoms;
    }

    //! Non-conservative force on \p atom along \p d, in c_ncForceUnit.
    static double ncForce(size_t atom, size_t d)
    {
        return (d == 1 ? -1.0 : 1.0) * (atom + 1) * (d + 2);
    }
    //! Element (\p a, \p b) of the non-conservative stress, in c_ncStressUnit.
    static double                ncStress(size_t a, size_t b) { return 0.01 * (3 * a + b + 1); }
    static constexpr const char* c_ncForceUnit  = "eV/Angstrom";
    static constexpr const char* c_ncStressUnit = "eV/Angstrom^3";

    metatomic::ModelCapabilities capabilities() const override
    {
        auto outputs = extraOutputs();
        outputs.insert(
                outputs.begin(),
                metatomic::Quantity::builder()
                        .name("energy")
                        .unit("kJ/mol")
                        .sample_kind(metatomic::SampleKind::System)
                        .gradients(declaresStrain_ ? std::vector{ metatomic::Gradients::Positions,
                                                                  metatomic::Gradients::Strain }
                                                   : std::vector{ metatomic::Gradients::Positions })
                        .build());
        return metatomic::ModelCapabilities::builder()
                .atomic_types({ 18 })
                .interaction_range(0.0)
                .length_unit("nm")
                .supported_devices({ metatomic::ModelCapabilities::Device::CPU })
                .dtype(metatomic::ModelCapabilities::DType::Float64)
                .outputs(std::move(outputs))
                .build();
    }

    metatomic::ModelMetadata metadata() const override
    {
        return metatomic::ModelMetadata::builder().name("GROMACS test recorder model").build();
    }

    std::vector<metatomic::PairListOptions>
    requested_pair_lists(const std::vector<metatomic::Quantity>& /*requestedOutputs*/) const override
    {
        return {};
    }

    std::vector<metatomic::Quantity>
    requested_inputs(const std::vector<metatomic::Quantity>& /*requestedOutputs*/) const override
    {
        std::vector<metatomic::Quantity> inputs;
        if (requestCharges_)
        {
            inputs.push_back(metatomic::Quantity::builder()
                                     .name("charge")
                                     .unit("e")
                                     .sample_kind(metatomic::SampleKind::Atom)
                                     .build());
        }
        if (siteInputs_)
        {
            inputs.push_back(metatomic::Quantity::builder()
                                     .name("charge")
                                     .unit("e")
                                     .sample_kind(metatomic::SampleKind::System)
                                     .build());
            inputs.push_back(metatomic::Quantity::builder()
                                     .name("spin_multiplicity")
                                     .unit("")
                                     .sample_kind(metatomic::SampleKind::System)
                                     .build());
        }
        return inputs;
    }

    std::vector<metatensor::TensorMap> execute_inner(const std::vector<metatomic::System>& systems,
                                                     std::optional<metatensor::Labels> /*selectedAtoms*/,
                                                     const std::vector<metatomic::Quantity>& requestedOutputs) override
    {
        std::vector<metatensor::TensorMap> outputs;
        for (const auto& output : requestedOutputs)
        {
            const auto& name = output.name();
            if (name == "energy")
            {
                std::vector<double> energies;
                for (const auto& system : systems)
                {
                    const auto value = [&system](const char* input)
                    {
                        auto       data   = system.custom_data(input);
                        auto       block  = data.block_by_id(0);
                        const auto values = block.values<double>();
                        return values(0, 0);
                    };
                    energies.push_back(siteInputs_ ? siteEnergy(value("charge"),
                                                                value("spin_multiplicity"),
                                                                system.size())
                                                   : 0.0);
                }
                outputs.push_back(energy(systems, energies, output.gradients()));
            }
            else if (name == "non_conservative_force")
            {
                outputs.push_back(perAtom(systems, name, 3, ncForce));
            }
            else if (name == "non_conservative_stress")
            {
                outputs.push_back(stress(systems.size()));
            }
            else if (name == "energy_uncertainty")
            {
                outputs.push_back(perAtom(systems,
                                          "energy",
                                          1,
                                          [this](size_t atom, size_t /*d*/)
                                          { return uncertainty_.value() * (atom + 1); }));
            }
            else
            {
                throw std::runtime_error("the recorder model has no output " + name);
            }
        }
        return outputs;
    }

    //! Number of evaluations asked for the strain gradient, by all RecorderModel instances
    static inline int strainRequests = 0;

private:
    //! The outputs beyond the energy that the model declares
    std::vector<metatomic::Quantity> extraOutputs() const
    {
        std::vector<metatomic::Quantity> outputs;
        if (nonConservative_)
        {
            outputs.push_back(metatomic::Quantity::builder()
                                      .name("non_conservative_force")
                                      .unit(c_ncForceUnit)
                                      .sample_kind(metatomic::SampleKind::Atom)
                                      .build());
            outputs.push_back(metatomic::Quantity::builder()
                                      .name("non_conservative_stress")
                                      .unit(c_ncStressUnit)
                                      .sample_kind(metatomic::SampleKind::System)
                                      .build());
        }
        if (uncertainty_.has_value())
        {
            outputs.push_back(metatomic::Quantity::builder()
                                      .name("energy_uncertainty")
                                      .unit("kJ/mol")
                                      .sample_kind(metatomic::SampleKind::Atom)
                                      .build());
        }
        return outputs;
    }

    //! The \p energies of the systems, with zero gradients for those in \p gradients.
    static metatensor::TensorMap energy(const std::vector<metatomic::System>&    systems,
                                        std::vector<double>                      energies,
                                        const std::vector<metatomic::Gradients>& gradients)
    {
        const auto has = [&gradients](metatomic::Gradients gradient)
        { return std::find(gradients.begin(), gradients.end(), gradient) != gradients.end(); };
        const size_t         nSystems = systems.size();
        std::vector<int32_t> systemSamples;
        std::vector<int32_t> atomSamples;
        for (size_t s = 0; s < nSystems; ++s)
        {
            systemSamples.push_back(static_cast<int32_t>(s));
            for (size_t atom = 0; atom < systems[s].size(); ++atom)
            {
                atomSamples.insert(
                        atomSamples.end(),
                        { static_cast<int32_t>(s), static_cast<int32_t>(s), static_cast<int32_t>(atom) });
            }
        }
        const size_t nAtoms = atomSamples.size() / 3;
        auto         block  = metatensor::TensorBlock(
                std::make_unique<metatensor::SimpleDataArray<double>>(
                        std::vector<uintptr_t>{ nSystems, 1 }, std::move(energies)),
                metatensor::Labels({ "system" }, systemSamples.data(), nSystems),
                {},
                metatensor::Labels({ "energy" }, { { 0 } }));

        if (has(metatomic::Gradients::Positions))
        {
            block.add_gradient(
                    "positions",
                    metatensor::TensorBlock(
                            std::make_unique<metatensor::SimpleDataArray<double>>(
                                    std::vector<uintptr_t>{ nAtoms, 3, 1 }, std::vector<double>(3 * nAtoms)),
                            metatensor::Labels({ "sample", "system", "atom" }, atomSamples.data(), nAtoms),
                            { metatensor::Labels({ "xyz" }, { { 0 }, { 1 }, { 2 } }) },
                            metatensor::Labels({ "energy" }, { { 0 } })));
        }
        if (has(metatomic::Gradients::Strain))
        {
            ++strainRequests;
            block.add_gradient("strain",
                               metatensor::TensorBlock(
                                       std::make_unique<metatensor::SimpleDataArray<double>>(
                                               std::vector<uintptr_t>{ nSystems, 3, 3, 1 },
                                               std::vector<double>(9 * nSystems)),
                                       metatensor::Labels({ "sample" }, systemSamples.data(), nSystems),
                                       { metatensor::Labels({ "xyz_1" }, { { 0 }, { 1 }, { 2 } }),
                                         metatensor::Labels({ "xyz_2" }, { { 0 }, { 1 }, { 2 } }) },
                                       metatensor::Labels({ "energy" }, { { 0 } })));
        }

        return single(std::move(block));
    }

    /*! \brief Per-atom values of \p f(atom, d), with \p nComponents xyz components if it is 3.
     *
     * The atom is the index of the atom in its system.
     */
    template<typename F>
    static metatensor::TensorMap perAtom(const std::vector<metatomic::System>& systems,
                                         const std::string&                    property,
                                         size_t                                nComponents,
                                         F                                     f)
    {
        std::vector<int32_t> samples;
        std::vector<double>  values;
        for (size_t s = 0; s < systems.size(); ++s)
        {
            for (size_t atom = 0; atom < systems[s].size(); ++atom)
            {
                samples.insert(samples.end(), { static_cast<int32_t>(s), static_cast<int32_t>(atom) });
                for (size_t d = 0; d < nComponents; ++d)
                {
                    values.push_back(f(atom, d));
                }
            }
        }
        const size_t                    nAtoms = samples.size() / 2;
        std::vector<metatensor::Labels> components;
        std::vector<uintptr_t>          shape{ nAtoms, 1 };
        if (nComponents == 3)
        {
            components.push_back(metatensor::Labels({ "xyz" }, { { 0 }, { 1 }, { 2 } }));
            shape = { nAtoms, 3, 1 };
        }
        return single(metatensor::TensorBlock(
                std::make_unique<metatensor::SimpleDataArray<double>>(shape, std::move(values)),
                metatensor::Labels({ "system", "atom" }, samples.data(), nAtoms),
                components,
                metatensor::Labels({ property }, { { 0 } })));
    }

    //! ncStress() for each of \p nSystems systems
    static metatensor::TensorMap stress(size_t nSystems)
    {
        std::vector<double>  values;
        std::vector<int32_t> samples;
        for (size_t s = 0; s < nSystems; ++s)
        {
            samples.push_back(static_cast<int32_t>(s));
            for (size_t a = 0; a < 3; ++a)
            {
                for (size_t b = 0; b < 3; ++b)
                {
                    values.push_back(ncStress(a, b));
                }
            }
        }
        return single(metatensor::TensorBlock(
                std::make_unique<metatensor::SimpleDataArray<double>>(
                        std::vector<uintptr_t>{ nSystems, 3, 3, 1 }, std::move(values)),
                metatensor::Labels({ "system" }, samples.data(), nSystems),
                { metatensor::Labels({ "xyz_1" }, { { 0 }, { 1 }, { 2 } }),
                  metatensor::Labels({ "xyz_2" }, { { 0 }, { 1 }, { 2 } }) },
                metatensor::Labels({ "non_conservative_stress" }, { { 0 } })));
    }

    static metatensor::TensorMap single(metatensor::TensorBlock block)
    {
        std::vector<metatensor::TensorBlock> blocks;
        blocks.push_back(std::move(block));
        return metatensor::TensorMap(metatensor::Labels({ "_" }, { { 0 } }), std::move(blocks));
    }

    bool                  requestCharges_;
    bool                  declaresStrain_;
    bool                  nonConservative_;
    std::optional<double> uncertainty_;
    bool                  siteInputs_;
};

/*! \brief Plugin callback creating a RecorderModel.
 *
 * It requests a charge input with the option `request_charges`, declares no strain
 * gradient with `without_strain`, declares the non-conservative force and stress with
 * `non_conservative`, an energy uncertainty of (atom + 1) times x kJ/mol with
 * `uncertainty=x`, and requests the charge and spin multiplicity of each system with
 * `site_inputs`.
 */
inline mta_status_t loadRecorderModel(const char* loadFrom, const char* optionsJson, mta_model_t* model)
{
    if (std::string(loadFrom) != "gromacs-test-recorder-model")
    {
        return MTA_UNSUPPORTED_MODEL_ERROR;
    }
    try
    {
        const auto            options = nlohmann::json::parse(optionsJson);
        std::optional<double> uncertainty;
        if (options.contains("uncertainty"))
        {
            uncertainty = std::stod(options["uncertainty"].get<std::string>());
        }
        *model = metatomic::BaseModel::to_mta_model(
                std::make_unique<RecorderModel>(options.contains("request_charges"),
                                                !options.contains("without_strain"),
                                                options.contains("non_conservative"),
                                                uncertainty,
                                                options.contains("site_inputs")));
        return MTA_SUCCESS;
    }
    catch (const std::exception& e)
    {
        mta_set_last_error(e.what(), "gromacs-test-recorder-model", nullptr, nullptr);
        return MTA_MODEL_ERROR;
    }
}

//! Register the in-process plugin of the recorder model once per test binary.
inline bool registerRecorderPlugin()
{
    static const bool registered = []
    {
        mta_plugin_t plugin{ MTA_ABI_VERSION, "gromacs-test", loadRecorderModel };
        return mta_register_plugin(plugin) == MTA_SUCCESS;
    }();
    return registered;
}

} // namespace gmx::test

#endif
