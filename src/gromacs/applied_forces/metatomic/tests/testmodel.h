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
 * not cover: the inputs a model requests, a model without pair lists, and which
 * gradients GROMACS asks for.
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
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace gmx::test
{

/*! \brief Model with a zero energy per system, which declares positions and strain gradients.
 *
 * It reads no pairs, can request a charge input that GROMACS does not provide or leave
 * out the strain gradient, and counts the evaluations that ask it for the strain gradient.
 */
class RecorderModel final : public metatomic::BaseModel
{
public:
    RecorderModel(bool requestCharges, bool declaresStrain) :
        requestCharges_(requestCharges), declaresStrain_(declaresStrain)
    {
    }

    metatomic::ModelCapabilities capabilities() const override
    {
        return metatomic::ModelCapabilities::builder()
                .atomic_types({ 18 })
                .interaction_range(0.0)
                .length_unit("nm")
                .supported_devices({ metatomic::ModelCapabilities::Device::CPU })
                .dtype(metatomic::ModelCapabilities::DType::Float64)
                .add_output(metatomic::Quantity::builder()
                                    .name("energy")
                                    .unit("kJ/mol")
                                    .sample_kind(metatomic::SampleKind::System)
                                    .gradients(declaresStrain_
                                                       ? std::vector{ metatomic::Gradients::Positions,
                                                                      metatomic::Gradients::Strain }
                                                       : std::vector{ metatomic::Gradients::Positions })
                                    .build())
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
                                                     std::optional<metatensor::Labels> /*selectedAtoms*/,
                                                     const std::vector<metatomic::Quantity>& requestedOutputs) override
    {
        const auto nAtoms = systems.at(0).size();
        auto       block  = metatensor::TensorBlock(
                std::make_unique<metatensor::SimpleDataArray<double>>(
                        std::vector<uintptr_t>{ 1, 1 }, std::vector<double>{ 0.0 }),
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
                metatensor::TensorBlock(
                        std::make_unique<metatensor::SimpleDataArray<double>>(
                                std::vector<uintptr_t>{ nAtoms, 3, 1 }, std::vector<double>(3 * nAtoms)),
                        metatensor::Labels({ "sample", "system", "atom" }, gradientSamples.data(), nAtoms),
                        { metatensor::Labels({ "xyz" }, { { 0 }, { 1 }, { 2 } }) },
                        metatensor::Labels({ "energy" }, { { 0 } })));

        const auto& gradients = requestedOutputs.front().gradients();
        if (std::find(gradients.begin(), gradients.end(), metatomic::Gradients::Strain) != gradients.end())
        {
            ++strainRequests;
            block.add_gradient(
                    "strain",
                    metatensor::TensorBlock(
                            std::make_unique<metatensor::SimpleDataArray<double>>(
                                    std::vector<uintptr_t>{ 1, 3, 3, 1 }, std::vector<double>(9)),
                            metatensor::Labels({ "sample" }, { { 0 } }),
                            { metatensor::Labels({ "xyz_1" }, { { 0 }, { 1 }, { 2 } }),
                              metatensor::Labels({ "xyz_2" }, { { 0 }, { 1 }, { 2 } }) },
                            metatensor::Labels({ "energy" }, { { 0 } })));
        }

        std::vector<metatensor::TensorBlock> blocks;
        blocks.push_back(std::move(block));
        std::vector<metatensor::TensorMap> outputs;
        outputs.emplace_back(metatensor::Labels({ "_" }, { { 0 } }), std::move(blocks));
        return outputs;
    }

    //! Number of evaluations asked for the strain gradient, by all RecorderModel instances
    static inline int strainRequests = 0;

private:
    bool requestCharges_;
    bool declaresStrain_;
};

/*! \brief Plugin callback creating a RecorderModel.
 *
 * It requests a charge input with the option `request_charges`, and declares no strain
 * gradient with `without_strain`.
 */
inline mta_status_t loadRecorderModel(const char* loadFrom, const char* optionsJson, mta_model_t* model)
{
    if (std::string(loadFrom) != "gromacs-test-recorder-model")
    {
        return MTA_UNSUPPORTED_MODEL_ERROR;
    }
    try
    {
        const auto options = nlohmann::json::parse(optionsJson);
        *model             = metatomic::BaseModel::to_mta_model(std::make_unique<RecorderModel>(
                options.contains("request_charges"), !options.contains("without_strain")));
        return MTA_SUCCESS;
    }
    catch (const std::exception& e)
    {
        mta_set_last_error(e.what(), "gromacs-test-recorder-model", nullptr, nullptr);
        return MTA_MODEL_ERROR;
    }
}

} // namespace gmx::test

#endif
