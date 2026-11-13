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
 * Implements the options for the Metatomic MDModule.
 *
 * \author Metatensor developers <https://github.com/metatensor>
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "metatomic_options.h"

#include "config.h"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <set>
#include <sstream>

#include "gromacs/domdec/localatomset.h"
#include "gromacs/fileio/warninp.h"
#include "gromacs/mdtypes/imdpoptionprovider_helpers.h"
#include "gromacs/options/basicoptions.h"
#include "gromacs/options/optionsection.h"
#include "gromacs/selection/indexutil.h"
#include "gromacs/topology/embedded_system_preprocessing.h"
#include "gromacs/topology/mtop_util.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/keyvaluetreebuilder.h"
#include "gromacs/utility/keyvaluetreetransform.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/mpicomm.h"
#include "gromacs/utility/strconvert.h"
#include "gromacs/utility/stringutil.h"


namespace gmx
{

static const std::string METATOMIC_MODULE_NAME = "metatomic";

/*! \brief Following Tags denotes names of parameters from .mdp file
 * \note Changing this strings will break .tpr backwards compatibility
 */

static const std::string ACTIVE_TAG      = "active";
static const std::string INPUT_GROUP_TAG = "input-group";

static const std::string MODEL_PATH_TAG           = "model";
static const std::string PLUGIN_TAG = "plugin";
static const std::string MODEL_OPTIONS_TAG        = "model-options";
static const std::string CHECK_CONSISTENCY_TAG    = "check-consistency";
static const std::string VARIANT_TAG              = "variant";
static const std::string UNCERTAINTY_THRESHOLD_TAG = "uncertainty-threshold";
static const std::string VARIANT_ENERGY_UQ_TAG     = "variant-energy-uq";
static const std::string NON_CONSERVATIVE_TAG      = "non-conservative";
static const std::string VARIANT_NC_FORCES_TAG     = "variant-nc-forces";
static const std::string VARIANT_NC_STRESS_TAG     = "variant-nc-stress";

namespace
{

//! \brief Preprocess the topology for the atoms given to the model.
void preprocessTopology(gmx_mtop_t* mtop, ArrayRef<const Index> mtaIndices, const MDLogger& logger, WarningHandler* wi)
{
    // convert mtaIndices to set for faster lookup
    std::set<int> mtaIndicesSet(mtaIndices.begin(), mtaIndices.end());
    int           numMTAAtoms     = static_cast<int>(mtaIndices.size());
    int           numRegularAtoms = mtop->natoms - numMTAAtoms;

    GMX_LOG(logger.info)
            .appendText("Metatomic potential interface is active, topology was modified!");
    GMX_LOG(logger.info)
            .appendTextFormatted(
                    "Number of embedded Metatomic atoms: %d\nNumber of regular atoms: %d\n",
                    numMTAAtoms,
                    numRegularAtoms);

    // 1) Split QM-containing molecules from other molecules in blocks
    std::vector<bool> isMTABlock = splitEmbeddedBlocks(mtop, mtaIndicesSet);

    // 2) Exclude non-bonded interactions between QM atoms
    addEmbeddedNBExclusions(mtop, mtaIndicesSet, logger);

    // 3) Build atomNumbers vector with atomic numbers of all atoms
    std::vector<int> atomNumbers = buildEmbeddedAtomNumbers(*mtop);

    // 4) Make F_CONNBOND between atoms within QM region
    modifyEmbeddedTwoCenterInteractions(mtop, mtaIndicesSet, isMTABlock, logger);

    // 5) Remove angles and settles containing 2 or more QM atoms
    modifyEmbeddedThreeCenterInteractions(mtop, mtaIndicesSet, isMTABlock, logger);

    // 6) Remove dihedrals containing 3 or more QM atoms
    modifyEmbeddedFourCenterInteractions(mtop, mtaIndicesSet, isMTABlock, logger);

    // 7) Check for constrained bonds in subsystem
    checkConstrainedBonds(mtop, mtaIndicesSet, isMTABlock, wi);

    // finalize topology
    mtop->finalize();
}
//! metatomic-uncertainty-threshold is auto, off, or a non-negative number of kJ/mol.
void checkUncertaintyThreshold(const std::string& value)
{
    if (value == "auto" || value == "off")
    {
        return;
    }
    try
    {
        if (doubleFromString(value.c_str()) >= 0)
        {
            return;
        }
    }
    catch (const InvalidInputError&)
    {
    }
    GMX_THROW(InconsistentInputError(formatString(
            "metatomic-uncertainty-threshold must be auto, off or a non-negative number in kJ/mol, "
            "not '%s'.",
            value.c_str())));
}

[[noreturn]] void modelOptionsError(const char* reason)
{
    GMX_THROW(InconsistentInputError(std::string("Invalid metatomic-model-options: ") + reason));
}
} // namespace

std::map<std::string, std::string> parseMetatomicModelOptions(const std::string& text)
{
    std::map<std::string, std::string> options;
    size_t position = 0;
    const auto isSpace = [](char c) { return std::isspace(static_cast<unsigned char>(c)) != 0; };
    const auto skipSpace = [&]()
    {
        while (position < text.size() && isSpace(text[position]))
        {
            ++position;
        }
    };
    skipSpace();
    while (position < text.size())
    {
        const size_t start = position;
        while (position < text.size())
        {
            const unsigned char c = static_cast<unsigned char>(text[position]);
            if (!std::isalnum(c) && c != '_' && c != '-' && c != '.')
            {
                break;
            }
            ++position;
        }
        const std::string key = text.substr(start, position - start);
        skipSpace();
        if (key.empty() || position == text.size() || text[position] != '=')
        {
            modelOptionsError("expected key=value with an alphanumeric, underscore, hyphen or dot key.");
        }
        ++position;
        skipSpace();
        if (position == text.size())
        {
            modelOptionsError("missing value; use an empty quoted string for an empty value.");
        }
        std::string value;
        if (text[position] == '\'' || text[position] == '"')
        {
            const char quote = text[position++];
            while (position < text.size() && text[position] != quote)
            {
                char c = text[position++];
                if (c == '\\')
                {
                    if (position == text.size()
                        || (text[position] != quote && text[position] != '\\'))
                    {
                        modelOptionsError("inside quotes, escape only the matching quote or a backslash.");
                    }
                    c = text[position++];
                }
                value += c;
            }
            if (position == text.size())
            {
                modelOptionsError("unterminated quoted value.");
            }
            ++position;
            if (position < text.size() && !isSpace(text[position]))
            {
                modelOptionsError("expected whitespace after a quoted value.");
            }
        }
        else
        {
            while (position < text.size() && !isSpace(text[position]))
            {
                if (text[position] == '\'' || text[position] == '"')
                {
                    modelOptionsError("quotes must enclose the entire value.");
                }
                value += text[position++];
            }
        }
        if (!options.emplace(key, value).second)
        {
            modelOptionsError("duplicate option key.");
        }
        skipSpace();
    }
    return options;
}

void MetatomicOptions::initMdpTransform(IKeyValueTreeTransformRules* rules)
{
    const auto& stringIdentityTransform = [](std::string s) { return s; };
    addMdpTransformFromString<bool>(rules, &fromStdString<bool>, METATOMIC_MODULE_NAME, ACTIVE_TAG);
    addMdpTransformFromString<std::string>(
            rules, stringIdentityTransform, METATOMIC_MODULE_NAME, INPUT_GROUP_TAG);

    addMdpTransformFromString<std::string>(
            rules, stringIdentityTransform, METATOMIC_MODULE_NAME, MODEL_PATH_TAG);
    addMdpTransformFromString<std::string>(
            rules, stringIdentityTransform, METATOMIC_MODULE_NAME, PLUGIN_TAG);
    addMdpTransformFromString<std::string>(
            rules, stringIdentityTransform, METATOMIC_MODULE_NAME, MODEL_OPTIONS_TAG);
    addMdpTransformFromString<bool>(
            rules, &fromStdString<bool>, METATOMIC_MODULE_NAME, CHECK_CONSISTENCY_TAG);
    addMdpTransformFromString<std::string>(
            rules, stringIdentityTransform, METATOMIC_MODULE_NAME, VARIANT_TAG);
    addMdpTransformFromString<std::string>(
            rules, stringIdentityTransform, METATOMIC_MODULE_NAME, UNCERTAINTY_THRESHOLD_TAG);
    addMdpTransformFromString<std::string>(
            rules, stringIdentityTransform, METATOMIC_MODULE_NAME, VARIANT_ENERGY_UQ_TAG);
    addMdpTransformFromString<bool>(
            rules, &fromStdString<bool>, METATOMIC_MODULE_NAME, NON_CONSERVATIVE_TAG);
    addMdpTransformFromString<std::string>(
            rules, stringIdentityTransform, METATOMIC_MODULE_NAME, VARIANT_NC_FORCES_TAG);
    addMdpTransformFromString<std::string>(
            rules, stringIdentityTransform, METATOMIC_MODULE_NAME, VARIANT_NC_STRESS_TAG);
}

void MetatomicOptions::initMdpOptions(IOptionsContainerWithSections* options)
{
    auto section = options->addSection(OptionSection(METATOMIC_MODULE_NAME.c_str()));
    section.addOption(BooleanOption(ACTIVE_TAG.c_str()).store(&params_.active));
    section.addOption(StringOption(INPUT_GROUP_TAG.c_str()).store(&params_.inputGroup));

    section.addOption(StringOption(MODEL_PATH_TAG.c_str()).store(&params_.modelPath_));
    section.addOption(StringOption(PLUGIN_TAG.c_str()).store(&params_.pluginPath));
    section.addOption(StringOption(MODEL_OPTIONS_TAG.c_str()).store(&params_.modelOptions));
    section.addOption(BooleanOption(CHECK_CONSISTENCY_TAG.c_str()).store(&params_.checkConsistency));
    section.addOption(StringOption(VARIANT_TAG.c_str()).store(&params_.variant));
    section.addOption(StringOption(UNCERTAINTY_THRESHOLD_TAG.c_str()).store(&params_.uncertaintyThreshold));
    section.addOption(StringOption(VARIANT_ENERGY_UQ_TAG.c_str()).store(&params_.variantEnergyUq));
    section.addOption(BooleanOption(NON_CONSERVATIVE_TAG.c_str()).store(&params_.nonConservative));
    section.addOption(StringOption(VARIANT_NC_FORCES_TAG.c_str()).store(&params_.variantNcForces));
    section.addOption(StringOption(VARIANT_NC_STRESS_TAG.c_str()).store(&params_.variantNcStress));
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
                builder, METATOMIC_MODULE_NAME, PLUGIN_TAG, params_.pluginPath);
        addMdpOutputValue<std::string>(
                builder, METATOMIC_MODULE_NAME, MODEL_OPTIONS_TAG, params_.modelOptions);
        addMdpOutputValue<bool>(
                builder, METATOMIC_MODULE_NAME, CHECK_CONSISTENCY_TAG, params_.checkConsistency);
        addMdpOutputValue<std::string>(builder, METATOMIC_MODULE_NAME, VARIANT_TAG, params_.variant);
        addMdpOutputValue<std::string>(
                builder, METATOMIC_MODULE_NAME, UNCERTAINTY_THRESHOLD_TAG, params_.uncertaintyThreshold);
        addMdpOutputValue<std::string>(
                builder, METATOMIC_MODULE_NAME, VARIANT_ENERGY_UQ_TAG, params_.variantEnergyUq);
        addMdpOutputValue<bool>(
                builder, METATOMIC_MODULE_NAME, NON_CONSERVATIVE_TAG, params_.nonConservative);
        addMdpOutputValue<std::string>(
                builder, METATOMIC_MODULE_NAME, VARIANT_NC_FORCES_TAG, params_.variantNcForces);
        addMdpOutputValue<std::string>(
                builder, METATOMIC_MODULE_NAME, VARIANT_NC_STRESS_TAG, params_.variantNcStress);
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
    preprocessTopology(top, params_.mtaIndices_, logger(), wi_);
}

void MetatomicOptions::writeParamsToKvt(KeyValueTreeObjectBuilder treeBuilder)
{
    if (!params_.active)
    {
        return;
    }
    // grompp rejects what mdrun would only find when it loads the model
    checkUncertaintyThreshold(params_.uncertaintyThreshold);
    const auto modelOptions = parseMetatomicModelOptions(params_.modelOptions);
    auto modelOptionsObject = treeBuilder.addObject("metatomic-model-loading-options");
    for (const auto& [key, value] : modelOptions)
    {
        modelOptionsObject.addValue(key, value);
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

    if (!tree.keyExists("metatomic-model-loading-options"))
    {
        GMX_THROW(InconsistentInputError("Cannot find metatomic model-loading options in the tpr file."));
    }
    std::ostringstream options;
    for (const auto& property : tree["metatomic-model-loading-options"].asObject().properties())
    {
        options << property.key() << '=' << std::quoted(property.value().cast<std::string>()) << ' ';
    }
    params_.modelOptions = options.str();
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


} // namespace gmx
