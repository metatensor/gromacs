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
 * Implements the options for NNPot MDModule class.
 *
 * \author Metatensor developers <https://github.com/metatensor>
 * \ingroup module_applied_forces
 */
// TODO(rg): Figure out how to insert the model into the .tpr file
#include "gmxpre.h"

#include "metatomic_options.h"

#include "gromacs/domdec/localatomset.h"
#include "gromacs/fileio/warninp.h"
#include "gromacs/options/basicoptions.h"
#include "gromacs/mdtypes/imdpoptionprovider_helpers.h"
#include "gromacs/options/optionsection.h"
#include "gromacs/selection/indexutil.h"
#include "gromacs/topology/mtop_util.h"
#include "gromacs/utility/keyvaluetreebuilder.h"
#include "gromacs/utility/keyvaluetreetransform.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/mpicomm.h"
#include "gromacs/utility/strconvert.h"
#include "gromacs/utility/stringutil.h"

#include "metatomic_topologypreprocessor.h"

namespace gmx
{

static const std::string METATOMIC_MODULE_NAME = "metatomic";

/*! \brief Following Tags denotes names of parameters from .mdp file
 * \note Changing this strings will break .tpr backwards compatibility
 */

static const std::string ACTIVE_TAG      = "active";
static const std::string INPUT_GROUP_TAG = "input_group";

static const std::string MODEL_PATH_TAG           = "model";
static const std::string EXTENSIONS_DIRECTORY_TAG = "extensions";
static const std::string CHECK_CONSISTENCY_TAG    = "check_consistency";
static const std::string DEVICE_TAG               = "device";

/*! \brief \internal Helper to declare mdp transform rules.
 *
 * This local helper function is no longer needed, as we now use
 * the common addMdpTransformFromString from mdpopenumutil.h
 */
/*
template<class ToType, class TransformWithFunctionType>
void MetatomicMdpTransformFromString(IKeyValueTreeTransformRules* rules,
                                     TransformWithFunctionType    transformationFunction,
                                     const std::string&           optionTag)
{
    rules->addRule()
            .from<std::string>("/" + METATOMIC_MODULE_NAME + "-" + optionTag)
            .to<ToType>("/" + METATOMIC_MODULE_NAME + "/" + optionTag)
            .transformWith(transformationFunction);
}
*/

void MetatomicOptions::initMdpTransform(IKeyValueTreeTransformRules* rules)
{
    const auto& stringIdentityTransform = [](std::string s) { return s; };
    // Use the common helper function instead of the local one
    addMdpTransformFromString<bool>(rules, &fromStdString<bool>, METATOMIC_MODULE_NAME, ACTIVE_TAG);
    addMdpTransformFromString<std::string>(
            rules, stringIdentityTransform, METATOMIC_MODULE_NAME, INPUT_GROUP_TAG);

    addMdpTransformFromString<std::string>(
            rules, stringIdentityTransform, METATOMIC_MODULE_NAME, MODEL_PATH_TAG);
    addMdpTransformFromString<std::string>(
            rules, stringIdentityTransform, METATOMIC_MODULE_NAME, EXTENSIONS_DIRECTORY_TAG);
    addMdpTransformFromString<bool>(
            rules, &fromStdString<bool>, METATOMIC_MODULE_NAME, CHECK_CONSISTENCY_TAG);
    addMdpTransformFromString<std::string>(rules, stringIdentityTransform, METATOMIC_MODULE_NAME, DEVICE_TAG);
}

void MetatomicOptions::initMdpOptions(IOptionsContainerWithSections* options)
{
    auto section = options->addSection(OptionSection(METATOMIC_MODULE_NAME.c_str()));
    section.addOption(BooleanOption(ACTIVE_TAG.c_str()).store(&params_.active));
    section.addOption(StringOption(INPUT_GROUP_TAG.c_str()).store(&params_.inputGroup));

    section.addOption(StringOption(MODEL_PATH_TAG.c_str()).store(&params_.modelPath_));
    section.addOption(StringOption(EXTENSIONS_DIRECTORY_TAG.c_str()).store(&params_.extensionsDirectory));
    section.addOption(StringOption(DEVICE_TAG.c_str()).store(&params_.device));
    section.addOption(BooleanOption(CHECK_CONSISTENCY_TAG.c_str()).store(&params_.checkConsistency));
}

void MetatomicOptions::buildMdpOutput(KeyValueTreeObjectBuilder* builder) const
{
    // new empty line before writing mdp values
    // Use helper functions for MDP output
    addMdpOutputComment(builder, METATOMIC_MODULE_NAME, "empty-line", "");

    addMdpOutputComment(builder,
                        METATOMIC_MODULE_NAME,
                        "module",
                        "; Machine learning potential using metatomic");
    addMdpOutputValue(builder, METATOMIC_MODULE_NAME, ACTIVE_TAG, params_.active);

    if (params_.active)
    {
        addMdpOutputValue<std::string>(builder, METATOMIC_MODULE_NAME, INPUT_GROUP_TAG, params_.inputGroup);

        addMdpOutputValue<std::string>(builder, METATOMIC_MODULE_NAME, MODEL_PATH_TAG, params_.modelPath_);
        addMdpOutputValue<std::string>(
                builder, METATOMIC_MODULE_NAME, EXTENSIONS_DIRECTORY_TAG, params_.extensionsDirectory);
        addMdpOutputValue<std::string>(builder, METATOMIC_MODULE_NAME, DEVICE_TAG, params_.device);
        addMdpOutputValue<bool>(
                builder, METATOMIC_MODULE_NAME, CHECK_CONSISTENCY_TAG, params_.checkConsistency);
    }
}

const MetatomicParameters& MetatomicOptions::parameters()
{
    return params_;
}

bool MetatomicOptions::isActive() const
{
    return params_.active;
}


void MetatomicOptions::setInputGroupIndices(const IndexGroupsAndNames& indexGroupsAndNames)
{
    if (!params_.active)
    {
        return;
    }
    params_.mtaIndices_ = indexGroupsAndNames.indices(params_.inputGroup);

    if (params_.mtaIndices_.empty())
    {
        GMX_THROW(InconsistentInputError(formatString(
                "Group %s defining metatomic potential input atoms should not be empty.",
                params_.inputGroup.c_str())));
    }
}

void MetatomicOptions::modifyTopology(gmx_mtop_t* top)
{
    if (!params_.active)
    {
        return;
    }

    MetatomicTopologyPreprocessor topPrep(params_.mtaIndices_);
    topPrep.preprocess(top, logger(), wi_);
}

void MetatomicOptions::writeParamsToKvt(KeyValueTreeObjectBuilder treeBuilder)
{
    if (!params_.active)
    {
        return;
    }

    auto GroupIndexAdder =
            treeBuilder.addUniformArray<std::int64_t>(METATOMIC_MODULE_NAME + "-" + INPUT_GROUP_TAG);
    for (const auto& indexValue : params_.mtaIndices_)
    {
        GroupIndexAdder.addValue(indexValue);
    }
}

void MetatomicOptions::readParamsFromKvt(const KeyValueTreeObject& tree)
{
    if (!params_.active)
    {
        return;
    }

    std::string key = METATOMIC_MODULE_NAME + "-" + INPUT_GROUP_TAG;
    if (!tree.keyExists(key))
    {
        GMX_THROW(InconsistentInputError(
                "Cannot find input atoms index vector required for metatomic potential.\n"
                "This could be caused by incompatible or corrupted tpr input file."));
    }

    auto kvtIndexArray = tree[key].asArray().values();
    params_.mtaIndices_.resize(kvtIndexArray.size());
    std::transform(std::begin(kvtIndexArray),
                   std::end(kvtIndexArray),
                   std::begin(params_.mtaIndices_),
                   [](const KeyValueTreeValue& val) { return val.cast<std::int64_t>(); });
}


void MetatomicOptions::setLogger(const MDLogger& logger)
{
    logger_ = &logger;
}

void MetatomicOptions::setWarningHandler(WarningHandler* wi)
{
    wi_ = wi;
}

void MetatomicOptions::setTopology(const gmx_mtop_t& top)
{
    params_.atoms_    = gmx_mtop_global_atoms(top);
    params_.numAtoms_ = params_.atoms_.nr;
}

void MetatomicOptions::setPbcType(const PbcType& pbcType)
{
    params_.pbcType_ = std::make_unique<PbcType>(pbcType);
}

void MetatomicOptions::setComm(const MpiComm& mpiComm)
{
    mpiComm_ = &mpiComm;
}


const MDLogger& MetatomicOptions::logger() const
{
    GMX_RELEASE_ASSERT(logger_, "Logger not set for MetatomicOptions.");
    return *logger_;
}

const MpiComm& MetatomicOptions::mpiComm() const
{
    GMX_RELEASE_ASSERT(mpiComm_, "MPI communicator not set for MetatomicOptions.");
    return *mpiComm_;
}


void MetatomicOptions::setLocalInputAtomSet(const LocalAtomSet& localInputAtomSet)
{
    params_.mtaAtoms_ = std::make_unique<LocalAtomSet>(localInputAtomSet);
}

void MetatomicOptions::setLocalgmxMMAtomSet(const LocalAtomSet& localMMAtomSet)
{
    params_.gmxMMAtoms_ = std::make_unique<LocalAtomSet>(localMMAtomSet);
}


} // namespace gmx
