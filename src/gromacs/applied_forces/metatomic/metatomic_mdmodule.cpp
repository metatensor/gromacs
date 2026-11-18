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
 * \author Eric Boittier <eric.boittier@epfl.ch>
 * \author Rohit Goswami <r95g10@gmail.com>
 * \author Guillaume Fraux <guillaume.fraux@epfl.ch>
 * \ingroup module_applied_forces
 */


#include "gmxpre.h"

#include "config.h"

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
     * A notifier calls the callbacks whose argument has the type it sends, given here
     * after each notification the Metatomic module subscribes to:
     * - the index groups, to find the atoms of the model: const IndexGroupsAndNames&
     * - the topology, to remove the classical interactions among these atoms: gmx_mtop_t*
     * - writing the module parameters to the .tpr file: KeyValueTreeObjectBuilder
     * - the logger: const MDLogger&
     * - the grompp warning handler: WarningHandler*
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
     * A notifier calls the callbacks whose argument has the type it sends, given here
     * after each notification the Metatomic module subscribes to:
     * - reading the module parameters from the .tpr file: const KeyValueTreeObject&
     * - the local atom set manager, to follow the atoms of the model: LocalAtomSetManager*
     * - the topology: const gmx_mtop_t&
     * - the periodic boundary conditions: const PbcType&
     * - the MPI communicator: const MpiComm&
     * - the logger: const MDLogger&
     * - the request for an energy term: MDModulesEnergyOutputToMetatomicRequestChecker*
     * - the range of the plain pairlist, which loads the model: PlainPairlistRanges*
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
            LocalAtomSet atomSet = localAtomSetManager->add(options_.parameters().mtaIndices_);
            options_.setLocalInputAtomSet(atomSet);
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
                [](MDModulesEnergyOutputToMetatomicRequestChecker* energyOutputRequest)
        { energyOutputRequest->energyOutputToMetatomic_ = true; };
        notifiers->simulationSetupNotifier_.subscribe(requestEnergyOutput);

#if GMX_METATOMIC_ACTIVE
        // Every setup notification the force provider needs comes before this one, so
        // the model is loaded here, once, and the force provider is kept for the run.
        const auto setPlainPairlistRangeFunction = [this](PlainPairlistRanges* ranges)
        {
            createForceProvider();
            const real range = force_provider_->plainPairlistRange(*ranges);
            // mdrun requires a range from the modules that subscribe to the pairlist
            needsPairlist_ = range > 0;
            if (needsPairlist_)
            {
                ranges->addRange(range);
            }
        };
        notifiers->simulationSetupNotifier_.subscribe(setPlainPairlistRangeFunction);
#endif // GMX_METATOMIC_ACTIVE
    }

    /*! \brief Requests to be notified during the simulation.
     *
     * \param[in] notifiers allows the module to subscribe to notifications from MdModules.
     *
     * A notifier calls the callbacks whose argument has the type it sends, given here
     * after each notification the Metatomic module subscribes to:
     * - the atoms of this rank, after domain decomposition: const MDModulesAtomsRedistributedSignal&
     * - the plain pairlist, unless the model reads no pairs: const MDModulesPairlistConstructedSignal&
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

        if (needsPairlist_)
        {
            const auto notifyPairlistFunction = [this](const MDModulesPairlistConstructedSignal& signal)
            { force_provider_->setPairlist(signal); };
            notifiers->simulationRunNotifier_.subscribe(notifyPairlistFunction);
        }
    }

    void initForceProviders(ForceProviders* forceProviders) override
    {
        if (!options_.isActive())
        {
            return;
        }

        createForceProvider();
        forceProviders->addForceProvider(force_provider_.get(), "Metatomic");
    }

    IMdpOptionProvider* mdpOptionProvider() override { return &options_; }
    IMDOutputProvider*  outputProvider() override { return nullptr; }

private:
    //! Load the model, unless that happened already during simulation setup.
    void createForceProvider()
    {
        if (!force_provider_)
        {
            force_provider_ = std::make_unique<MetatomicForceProvider>(
                    options_, options_.logger(), options_.mpiComm());
        }
    }

    MetatomicOptions options_;

    std::unique_ptr<MetatomicForceProvider> force_provider_;
    //! False for a model that reads no pairs
    bool needsPairlist_ = true;
};

} // end anonymous namespace

std::unique_ptr<IMDModule> MetatomicModuleInfo::create()
{
    return std::make_unique<MetatomicMDModule>();
}

} // end namespace gmx
