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
 * Implements the GPU-resident Metatomic MDModule.
 *
 * This module handles MDP option parsing for the `metatomic-gpu` section,
 * registers the model cutoff with the pairlist range system, and provides
 * the MetatomicParameters for runner.cpp to create the GPU force provider.
 * Unlike the CPU MetatomicMDModule, this does NOT register an IForceProvider.
 *
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "metatomic_gpu_mdmodule.h"

#include <cmath>
#include <cstdlib>

#include <set>
#include <string>

#include "gromacs/domdec/localatomset.h"
#include "gromacs/fileio/warninp.h"
#include "gromacs/mdrunutility/mdmodulesnotifiers.h"
#include "gromacs/mdrunutility/plainpairlistranges.h"
#include "gromacs/mdtypes/imdmodule.h"
#include "gromacs/mdtypes/imdpoptionprovider_helpers.h"
#include "gromacs/options/basicoptions.h"
#include "gromacs/options/optionsection.h"
#include "gromacs/selection/indexutil.h"
#include "gromacs/topology/embedded_system_preprocessing.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/basenetwork.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/keyvaluetreebuilder.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/strconvert.h"
#include "gromacs/utility/stringutil.h"

#include "metatomic_options.h"

#ifdef DIM
#    undef DIM
#endif

#include <metatensor/torch.hpp>
#include <metatomic/torch.hpp>

namespace gmx
{

namespace
{

/*! \internal
 * \brief MDP option provider for the metatomic-gpu section.
 *
 * Handles MDP parsing, topology preprocessing (NB exclusions for ML atoms),
 * and parameter storage. Shares the embedded-system preprocessing logic
 * with the CPU metatomic module.
 */
class MetatomicGpuOptions final : public IMdpOptionProvider
{
public:
    void initMdpTransform(IKeyValueTreeTransformRules* rules) override
    {
        const auto& stringIdentityTransform = [](std::string s) { return s; };
        addMdpTransformFromString<bool>(rules, &fromStdString<bool>, "metatomic-gpu", "active");
        addMdpTransformFromString<std::string>(
                rules, stringIdentityTransform, "metatomic-gpu", "model-path");
        addMdpTransformFromString<std::string>(
                rules, stringIdentityTransform, "metatomic-gpu", "extensions-directory");
        addMdpTransformFromString<std::string>(
                rules, stringIdentityTransform, "metatomic-gpu", "variant");
        addMdpTransformFromString<std::string>(
                rules, stringIdentityTransform, "metatomic-gpu", "input-group");
    }

    void initMdpOptions(IOptionsContainerWithSections* options) override
    {
        auto section = options->addSection(OptionSection("metatomic-gpu"));
        section.addOption(BooleanOption("active").store(&active_));
        section.addOption(StringOption("model-path").store(&modelPath_));
        section.addOption(StringOption("extensions-directory").store(&extensionsDir_));
        section.addOption(StringOption("variant").store(&variant_));
        section.addOption(StringOption("input-group").store(&inputGroup_));
    }

    void buildMdpOutput(KeyValueTreeObjectBuilder* builder) const override
    {
        addMdpOutputComment(builder, "metatomic-gpu", "empty-line", "");
        addMdpOutputComment(
                builder, "metatomic-gpu", "module", "; GPU-resident ML potential using metatomic");
        addMdpOutputValue(builder, "metatomic-gpu", "active", active_);
        if (active_)
        {
            addMdpOutputValue<std::string>(builder, "metatomic-gpu", "model-path", modelPath_);
            addMdpOutputValue<std::string>(
                    builder, "metatomic-gpu", "extensions-directory", extensionsDir_);
            addMdpOutputValue<std::string>(builder, "metatomic-gpu", "variant", variant_);
            addMdpOutputValue<std::string>(builder, "metatomic-gpu", "input-group", inputGroup_);
        }
    }

    bool        active() const { return active_; }
    std::string modelPath() const { return modelPath_; }
    std::string extensionsDir() const { return extensionsDir_; }
    std::string variant() const { return variant_; }

    //! Resolve input-group name to atom indices from index file
    void setInputGroupIndices(const IndexGroupsAndNames& indexGroupsAndNames)
    {
        mtaIndices_ = indexGroupsAndNames.indices(inputGroup_);
        if (mtaIndices_.empty())
        {
            GMX_THROW(InconsistentInputError(formatString(
                    "Group '%s' defining metatomic-gpu input atoms should not be empty.",
                    inputGroup_.c_str())));
        }
    }

    //! Modify topology: exclude classical NB for ML atoms, remove bonded interactions
    void modifyTopology(gmx_mtop_t* mtop)
    {
        std::set<int> mtaIndicesSet(mtaIndices_.begin(), mtaIndices_.end());
        int           numMTAAtoms     = static_cast<int>(mtaIndices_.size());
        int           numRegularAtoms = mtop->natoms - numMTAAtoms;

        GMX_LOG(logger().info)
                .appendText("Metatomic GPU potential interface is active, topology was modified!");
        GMX_LOG(logger().info)
                .appendTextFormatted(
                        "Number of embedded Metatomic-GPU atoms: %d\nNumber of regular atoms: %d\n",
                        numMTAAtoms,
                        numRegularAtoms);

        std::vector<bool> isMTABlock = splitEmbeddedBlocks(mtop, mtaIndicesSet);
        addEmbeddedNBExclusions(mtop, mtaIndicesSet, logger());
        buildEmbeddedAtomNumbers(*mtop);
        modifyEmbeddedTwoCenterInteractions(mtop, mtaIndicesSet, isMTABlock, logger());
        modifyEmbeddedThreeCenterInteractions(mtop, mtaIndicesSet, isMTABlock, logger());
        modifyEmbeddedFourCenterInteractions(mtop, mtaIndicesSet, isMTABlock, logger());
        checkConstrainedBonds(mtop, mtaIndicesSet, isMTABlock, wi_);
        mtop->finalize();
    }

    void setLogger(const MDLogger& logger) { logger_ = &logger; }
    void setWarningHandler(WarningHandler* wi) { wi_ = wi; }

    //! Write input-group indices to KVT for .tpr storage
    void writeInputGroupToKvt(KeyValueTreeObjectBuilder kvt)
    {
        auto indexAdder = kvt.addUniformArray<std::int64_t>("metatomic-gpu-input-group");
        for (const auto& idx : mtaIndices_)
        {
            indexAdder.addValue(idx);
        }
    }

    //! Build a MetatomicParameters struct for the GPU provider
    MetatomicParameters buildParams() const
    {
        MetatomicParameters params;
        params.active              = active_;
        params.modelPath_          = modelPath_;
        params.extensionsDirectory = extensionsDir_;
        params.variant             = variant_;
        params.device              = "cuda";
        params.nlMode              = "full";
        params.inputGroup          = inputGroup_;
        return params;
    }

private:
    const MDLogger& logger() const
    {
        GMX_RELEASE_ASSERT(logger_, "Logger not set for MetatomicGpuOptions.");
        return *logger_;
    }

    bool                active_     = false;
    std::string         modelPath_;
    std::string         extensionsDir_;
    std::string         variant_;
    std::string         inputGroup_ = "System";
    std::vector<Index>  mtaIndices_;
    const MDLogger*     logger_     = nullptr;
    WarningHandler*     wi_         = nullptr;
};


/*! \internal
 * \brief GPU Metatomic MDModule.
 *
 * Subscribes to PlainPairlistRanges to register the model cutoff.
 * Does NOT register an IForceProvider — the GPU force provider is created
 * directly in runner.cpp with the proper DeviceContext/DeviceStream.
 */
class MetatomicGpuMDModule final : public IMDModule
{
public:
    explicit MetatomicGpuMDModule() = default;

    void subscribeToPreProcessingNotifications(MDModulesNotifiers* notifiers) override
    {
        if (!options_.active())
        {
            return;
        }

        // Receive logger for topology modification messages
        notifiers->preProcessingNotifier_.subscribe(
                [this](const MDLogger& logger) { options_.setLogger(logger); });

        // Receive warning handler for constraint checks
        notifiers->preProcessingNotifier_.subscribe(
                [this](WarningHandler* wi) { options_.setWarningHandler(wi); });

        // Resolve input-group name to atom indices
        notifiers->preProcessingNotifier_.subscribe(
                [this](const IndexGroupsAndNames& idx) { options_.setInputGroupIndices(idx); });

        // Modify topology: exclude classical NB for ML atoms
        notifiers->preProcessingNotifier_.subscribe(
                [this](gmx_mtop_t* top) { options_.modifyTopology(top); });

        // Write GPU metatomic params + input-group indices to KVT for .tpr
        notifiers->preProcessingNotifier_.subscribe(
                [this](KeyValueTreeObjectBuilder kvt)
                {
                    auto section = kvt.addObject("metatomic-gpu");
                    section.addValue<bool>("active", options_.active());
                    section.addValue<std::string>("model-path", options_.modelPath());
                    section.addValue<std::string>("extensions-directory", options_.extensionsDir());
                    section.addValue<std::string>("variant", options_.variant());
                    options_.writeInputGroupToKvt(kvt);
                });
    }

    void subscribeToSimulationSetupNotifications(MDModulesNotifiers* notifiers) override
    {
        if (!options_.active())
        {
            return;
        }

        // Read params back from KVT (.tpr)
        notifiers->simulationSetupNotifier_.subscribe(
                [this](const KeyValueTreeObject& kvt)
                {
                    if (kvt.keyExists("metatomic-gpu"))
                    {
                        // Parameters already set from MDP; this block is
                        // a placeholder for future .tpr restore logic.
                        static_cast<void>(kvt["metatomic-gpu"]);
                    }
                });

        // Register model cutoff for pairlist range (same pattern as CPU module)
        notifiers->simulationSetupNotifier_.subscribe(
                [this](PlainPairlistRanges* ranges)
                {
                    double maxCutoff = 0.0;
                    try
                    {
                        auto model = metatomic_torch::load_atomistic_model(options_.modelPath());
                        auto caps  = model.run_method("capabilities")
                                             .toCustomClass<metatomic_torch::ModelCapabilitiesHolder>();
                        double range = caps->engine_interaction_range("nm");
                        if (range > 0.0 && std::isfinite(range))
                        {
                            maxCutoff = range;
                        }
                        auto nlRequests = model.run_method("requested_neighbor_lists");
                        for (const auto& req : nlRequests.toList())
                        {
                            auto nlOpt = req.get().toCustomClass<metatomic_torch::NeighborListOptionsHolder>();
                            maxCutoff  = std::max(maxCutoff, nlOpt->engine_cutoff("nm"));
                        }
                    }
                    catch (const std::exception& e)
                    {
                        GMX_THROW(InternalError(
                                "Failed to read cutoff from GPU metatomic model: " + std::string(e.what())));
                    }

                    if (maxCutoff <= 0.0 || !std::isfinite(maxCutoff))
                    {
                        GMX_THROW(InconsistentInputError(
                                "Metatomic GPU model cutoff is invalid."));
                    }
                    ranges->addRange(maxCutoff);
                });

        // Request "Metatomic Potential" energy term in .edr output
        notifiers->simulationSetupNotifier_.subscribe(
                [](MDModulesEnergyOutputToMetatomicPotRequestChecker* req)
                { req->energyOutputToMetatomicPot_ = true; });
    }

    void subscribeToSimulationRunNotifications(MDModulesNotifiers* /*notifiers*/) override
    {
        // GPU force provider handles atoms redistributed signal directly
        // from runner.cpp, not through the MDModule notification system.
    }

    void initForceProviders(ForceProviders* /*forceProviders*/) override
    {
        // Intentionally empty: GPU force provider is NOT an IForceProvider.
        // Created directly in runner.cpp with proper DeviceContext/DeviceStream.
    }

    IMdpOptionProvider* mdpOptionProvider() override { return &options_; }
    IMDOutputProvider*  outputProvider() override { return nullptr; }

private:
    MetatomicGpuOptions options_;
};

} // end anonymous namespace

std::unique_ptr<IMDModule> MetatomicGpuModuleInfo::create()
{
    return std::make_unique<MetatomicGpuMDModule>();
}

} // end namespace gmx
