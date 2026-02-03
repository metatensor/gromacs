/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2024- The GROMACS Authors
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
 * Implements Metatomic MDModule class
 *
 * \author Metatensor developers <https://github.com/metatensor>
 * \ingroup module_applied_forces
 */


#include "gmxpre.h"

#include "metatomic_mdmodule.h"

#include "gromacs/domdec/localatomset.h"
#include "gromacs/domdec/localatomsetmanager.h"
#include "gromacs/mdrunutility/mdmodulesnotifiers.h"
#include "gromacs/mdrunutility/plainpairlistranges.h"
#include "gromacs/mdtypes/imdmodule.h"
#include "gromacs/utility/basenetwork.h"
#include "gromacs/utility/keyvaluetreebuilder.h"

#include "metatomic_forceprovider.h"
#include "metatomic_options.h"
#ifdef DIM
#    undef DIM
#endif

#include <cmath>

#include <metatensor/torch.hpp>
#include <metatomic/torch.hpp>

namespace gmx
{

namespace
{

/*! \internal
 * \brief Metatomic Module
 *
 * Class that implements the metatomic MDModule.
 */
class MetatomicMDModule final : public IMDModule
{
public:
    explicit MetatomicMDModule() = default;

    /*! \brief Requests to be notified during preprocessing.
     *
     * \param[in] notifiers allows the module to subscribe to notifications from MdModules.
     *
     * The Metatomic module subscribes to the following notifications:
     * - The atom groups and their names from the index file (to specify the ML atoms)
     * by taking a const IndexGroupsAndNames& as a parameter.
     * - The system topology, which might be modified (e.g., to remove classical interactions).
     * by taking a gmx_mtop_t* as a parameter.
     * - Writing the module parameters to the KVT for storage in the .tpr file
     * by taking a KeyValueTreeObjectBuilder as a parameter.
     * - Accessing the MDLogger to log messages
     * by taking a const MDLogger& as a parameter.
     * - Accessing the WarningHandler to output warnings
     * by taking a WarningHandler* as a parameter.
     */
    void subscribeToPreProcessingNotifications(MDModulesNotifiers* notifiers) override
    {
        if (!options_.isActive())
        {
            return;
        }

        const auto setInputGroupIndicesFunction = [this](const IndexGroupsAndNames& indexGroupsAndNames)
        { options_.setInputGroupIndices(indexGroupsAndNames); };
        notifiers->preProcessingNotifier_.subscribe(setInputGroupIndicesFunction);

        const auto modifyTopologyFunction = [this](gmx_mtop_t* top) { options_.modifyTopology(top); };
        notifiers->preProcessingNotifier_.subscribe(modifyTopologyFunction);

        const auto writeParamsToKvtFunction = [this](KeyValueTreeObjectBuilder kvt)
        { options_.writeParamsToKvt(kvt); };
        notifiers->preProcessingNotifier_.subscribe(writeParamsToKvtFunction);

        const auto setLoggerFunction = [this](const MDLogger& logger) { options_.setLogger(logger); };
        notifiers->preProcessingNotifier_.subscribe(setLoggerFunction);

        const auto setWarningFunction = [this](WarningHandler* wi)
        { options_.setWarningHandler(wi); };
        notifiers->preProcessingNotifier_.subscribe(setWarningFunction);
    }

    /*! \brief Requests to be notified during simulation setup.
     *
     * \param[in] notifiers allows the module to subscribe to notifications from MdModules.
     *
     * The Metatomic module subscribes to the following notifications:
     * - Reading the module parameters from the KVT
     * by taking a const KeyValueTreeObject& as a parameter.
     * - The local atom set manager to construct a local atom set for the ML atoms.
     * by taking a LocalAtomSetManager* as a parameter.
     * - The system topology
     * by taking a const gmx_mtop_t& as a parameter.
     * - The PBC type
     * by taking a PbcType as a parameter.
     * - The MPI communicator
     * by taking a const MpiComm& as a parameter.
     * - The MDLogger to log messages
     * by taking a const MDLogger& as a parameter.
     */
    void subscribeToSimulationSetupNotifications(MDModulesNotifiers* notifiers) override
    {
        if (!options_.isActive())
        {
            return;
        }

        const auto readParamsFromKvtFunction = [this](const KeyValueTreeObject& kvt)
        { options_.readParamsFromKvt(kvt); };
        notifiers->simulationSetupNotifier_.subscribe(readParamsFromKvtFunction);

        const auto setLocalAtomSetFunction = [this](LocalAtomSetManager* localAtomSetManager)
        {
            // TODO(rg): why are these separate functions, they just create unique pointers..
            LocalAtomSet atomSet1 = localAtomSetManager->add(options_.parameters().mtaIndices_);
            options_.setLocalInputAtomSet(atomSet1);
            LocalAtomSet atomSet2 = localAtomSetManager->add(options_.parameters().mmIndices_);
            options_.setLocalgmxMMAtomSet(atomSet2);
        };
        notifiers->simulationSetupNotifier_.subscribe(setLocalAtomSetFunction);

        const auto setTopologyFunction = [this](const gmx_mtop_t& top) { options_.setTopology(top); };
        notifiers->simulationSetupNotifier_.subscribe(setTopologyFunction);

        const auto setPbcTypeFunction = [this](const PbcType& pbc) { options_.setPbcType(pbc); };
        notifiers->simulationSetupNotifier_.subscribe(setPbcTypeFunction);

        const auto setCommFunction = [this](const MpiComm& mpiComm) { options_.setComm(mpiComm); };
        notifiers->simulationSetupNotifier_.subscribe(setCommFunction);

        const auto setLoggerFunction = [this](const MDLogger& logger) { options_.setLogger(logger); };
        notifiers->simulationSetupNotifier_.subscribe(setLoggerFunction);

        // Request that GROMACS adds an energy term for our potential to the .edr file
        const auto requestEnergyOutput =
                [](MDModulesEnergyOutputToMetatomicPotRequestChecker* energyOutputRequest)
        { energyOutputRequest->energyOutputToMetatomicPot_ = true; };
        notifiers->simulationSetupNotifier_.subscribe(requestEnergyOutput);

        const auto setPlainPairlistRangeFunction = [this](PlainPairlistRanges* ranges)
        {
            // Temporary: Load model just to peek at cutoff.
            // TODO: can the whole model be loaded earlier..?
            double max_cutoff{ 0.0 };
            try
            {
                auto model = metatomic_torch::load_atomistic_model(options_.parameters().modelPath_);

                // Check strict interaction range
                auto capabilities =
                        model.run_method("capabilities").toCustomClass<metatomic_torch::ModelCapabilitiesHolder>();
                double interaction_range = capabilities->engine_interaction_range("nm");

                if (interaction_range < 0.0)
                {
                    GMX_THROW(InconsistentInputError(
                            "interaction_range is negative for this model."));
                }

                if (!std::isfinite(interaction_range))
                {
                    // Infinite range (global) is only supported on a single rank
                    if (gmx_node_num() > 1)
                    {
                        GMX_THROW(NotImplementedError(
                                "interaction_range is infinite for this model; "
                                "using multiple MPI domains is not supported."));
                    }
                    // For infinite range, check if specific NLs were requested
                    // effectively falling through to the loop below.
                }
                else
                {
                    max_cutoff = interaction_range;
                }

                // Check requested neighbor lists
                auto requested_nl = model.run_method("requested_neighbor_lists");
                for (const auto& ivalue : requested_nl.toList())
                {
                    auto options =
                            ivalue.get().toCustomClass<metatomic_torch::NeighborListOptionsHolder>();
                    double cutoff = options->engine_cutoff("nm");
                    max_cutoff    = std::max(max_cutoff, cutoff);
                }
            }
            catch (const std::exception& e)
            {
                GMX_THROW(InternalError("Failed to read cutoff from model: " + std::string(e.what())));
            }

            if (max_cutoff <= 0.0 || !std::isfinite(max_cutoff))
            {
                // If the model is purely global and requested no specific NL,
                // there's no need for a pairlist, so max_cutoff remains 0.
                if (max_cutoff == 0.0)
                {
                    GMX_THROW(InconsistentInputError("Metatomic model cutoff is 0.0 or invalid."));
                }
            }
            // Register the requirement with GROMACS
            ranges->addRange(max_cutoff);
        };
        // Register the callback
        notifiers->simulationSetupNotifier_.subscribe(setPlainPairlistRangeFunction);
    }

    /*! \brief Requests to be notified during the simulation.
     *
     * \param[in] notifiers allows the module to subscribe to notifications from MdModules.
     *
     * The Metatomic module subscribes to the following notifications:
     * - Atom redistribution due to domain decomposition
     * - Changes in the neighborlist
     * by taking a const MDModulesAtomsRedistributedSignal as a parameter.
     */
    void subscribeToSimulationRunNotifications(MDModulesNotifiers* notifiers) override
    {
        if (!options_.isActive())
        {
            return;
        }

        // After domain decomposition, the force provider needs to know which atoms are local.
        const auto notifyDDFunction = [this](const MDModulesAtomsRedistributedSignal& signal)
        { force_provider_->gatherAtomNumbersIndices(signal); };
        notifiers->simulationRunNotifier_.subscribe(notifyDDFunction);

        // subscribe to pairlist construction notification
        const auto notifyPairlistFunction = [this](const MDModulesPairlistConstructedSignal& signal)
        { force_provider_->setPairlist(signal); };
        notifiers->simulationRunNotifier_.subscribe(notifyPairlistFunction);
    }

    void initForceProviders(ForceProviders* forceProviders) override
    {
        if (!options_.isActive())
        {
            return;
        }

        force_provider_ = std::make_unique<MetatomicForceProvider>(
                options_, options_.logger(), options_.mpiComm());
        forceProviders->addForceProvider(force_provider_.get(), "Metatomic");
    }

    IMdpOptionProvider* mdpOptionProvider() override { return &options_; }
    IMDOutputProvider*  outputProvider() override { return nullptr; }

private:
    MetatomicOptions options_;

    std::unique_ptr<MetatomicForceProvider> force_provider_;
};

} // end anonymous namespace

std::unique_ptr<IMDModule> MetatomicModuleInfo::create()
{
    return std::make_unique<MetatomicMDModule>();
}

} // end namespace gmx
