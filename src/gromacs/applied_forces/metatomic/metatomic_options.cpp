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
 * \author Eric Boittier <eric.boittier@epfl.ch>
 * \author Rohit Goswami <r95g10@gmail.com>
 * \author Guillaume Fraux <guillaume.fraux@epfl.ch>
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "metatomic_options.h"

#include "config.h"

#include <cctype>
#include <cmath>

#include <algorithm>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#include "gromacs/domdec/localatomset.h"
#include "gromacs/fileio/warninp.h"
#include "gromacs/mdrunutility/mdmodulesnotifiers.h"
#include "gromacs/mdtypes/imdpoptionprovider_helpers.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/options/basicoptions.h"
#include "gromacs/options/optionsection.h"
#include "gromacs/selection/indexutil.h"
#include "gromacs/topology/embedded_system_preprocessing.h"
#include "gromacs/topology/exclusionblocks.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/topology/mtop_lookup.h"
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

static const std::string MODEL_PATH_TAG            = "model";
static const std::string PLUGIN_TAG                = "plugin";
static const std::string MODEL_OPTIONS_TAG         = "model-options";
static const std::string CHECK_CONSISTENCY_TAG     = "check-consistency";
static const std::string VARIANT_TAG               = "variant";
static const std::string UNCERTAINTY_THRESHOLD_TAG = "uncertainty-threshold";
static const std::string VARIANT_ENERGY_UQ_TAG     = "variant-energy-uq";
static const std::string NON_CONSERVATIVE_TAG      = "non-conservative";
static const std::string VARIANT_NC_FORCES_TAG     = "variant-nc-forces";
static const std::string VARIANT_NC_STRESS_TAG     = "variant-nc-stress";
static const std::string ONIOM_TAG                 = "oniom";
static const std::string LINK_ATOMS_TAG            = "link-atoms";
static const std::string LINK_FRONTIER_KEY         = "metatomic-link-frontier";
static const std::string SITE_GROUPS_TAG           = "site-groups";
static const std::string SITE_CHARGES_TAG          = "site-charges";
static const std::string SITE_SPINS_TAG            = "site-spin-multiplicities";
static const std::string SITES_KEY                 = "metatomic-sites";

namespace
{

/*! \brief Excludes the non-bonded interactions within each site, keeping those between sites
 *
 * GROMACS has a single intermolecular exclusion group, which would exclude all pairs of
 * embedded atoms. With several sites, the pairs within a site are added to the exclusions
 * of its molecule instead, so each site must lie within one molecule. Must run after
 * splitEmbeddedBlocks(), which gives each molecule with embedded atoms its own block.
 */
void addSiteExclusions(gmx_mtop_t*           mtop,
                       ArrayRef<const Index> mtaIndices,
                       ArrayRef<const int>   sites,
                       int                   numSites,
                       const MDLogger&       logger)
{
    std::vector<std::vector<int>> siteAtoms(numSites);
    std::vector<int>              siteBlock(numSites, -1);
    MTopLookUp                    lookUp(*mtop);
    for (size_t k = 0; k < mtaIndices.size(); k++)
    {
        const auto location = lookUp.getMolblockAtomIndex(mtaIndices[k]);
        const int  block    = location.molBlock;
        const int  atom     = location.atomIndex;
        const int  s        = sites[k];
        if (siteBlock[s] >= 0 && siteBlock[s] != block)
        {
            GMX_THROW(InconsistentInputError(
                    formatString("Site %d spans more than one molecule; with several "
                                 "metatomic-site-groups each site must lie within one molecule.",
                                 s + 1)));
        }
        GMX_RELEASE_ASSERT(mtop->molblock[block].nmol == 1,
                           "Each molecule with embedded atoms has a block of its own");
        siteBlock[s] = block;
        siteAtoms[s].push_back(atom);
    }

    int                                        numPairs = 0;
    std::map<int, std::vector<ExclusionBlock>> newExclusions;
    for (int s = 0; s < numSites; s++)
    {
        if (siteBlock[s] < 0)
        {
            continue;
        }
        auto& blocks = newExclusions[siteBlock[s]];
        blocks.resize(mtop->moltype[mtop->molblock[siteBlock[s]].type].atoms.nr);
        for (const int i : siteAtoms[s])
        {
            for (const int j : siteAtoms[s])
            {
                if (i != j)
                {
                    blocks[i].atomNumber.push_back(j);
                }
            }
        }
        numPairs += siteAtoms[s].size() * (siteAtoms[s].size() - 1) / 2;
    }
    for (auto& [block, blocks] : newExclusions)
    {
        mergeExclusions(&mtop->moltype[mtop->molblock[block].type].excls, blocks);
    }
    GMX_LOG(logger.info)
            .appendTextFormatted(
                    "Excluded the %d atom pairs within %d sites; pairs between sites keep their "
                    "classical interactions\n",
                    numPairs,
                    numSites);
}

/*! \brief Rejects a chemical bond between atoms of two sites.
 *
 * The model sees each site alone, and the topology preprocessing removes the bond.
 */
void checkNoBondsBetweenSites(const gmx_mtop_t& mtop, ArrayRef<const Index> mtaIndices, ArrayRef<const int> sites)
{
    std::map<int, int> siteOf;
    for (size_t k = 0; k < mtaIndices.size(); k++)
    {
        siteOf.emplace(mtaIndices[k], sites[k]);
    }
    for (size_t b = 0; b < mtop.molblock.size(); ++b)
    {
        const auto& moltype = mtop.moltype[mtop.molblock[b].type];
        for (int mol = 0; mol < mtop.molblock[b].nmol; ++mol)
        {
            const int start = mtop.moleculeBlockIndices[b].globalAtomStart + mol * moltype.atoms.nr;
            for (const auto ftype : EnumerationWrapper<InteractionFunction>{})
            {
                if (!(interaction_function[ftype].categories & sc_ifcChemicalBond) || NRAL(ftype) != 2)
                {
                    continue;
                }
                const auto& iatoms = moltype.ilist[ftype].iatoms;
                for (size_t j = 0; j < iatoms.size(); j += 3)
                {
                    const auto first  = siteOf.find(iatoms[j + 1] + start);
                    const auto second = siteOf.find(iatoms[j + 2] + start);
                    if (first != siteOf.end() && second != siteOf.end() && first->second != second->second)
                    {
                        GMX_THROW(InconsistentInputError(formatString(
                                "Atoms %d and %d are bonded but in sites %d and %d. The model "
                                "evaluates each site alone, so the bond would have no energy; put "
                                "both atoms in one of metatomic-site-groups.",
                                first->first + 1,
                                second->first + 1,
                                first->second + 1,
                                second->second + 1)));
                    }
                }
            }
        }
    }
}

/*! \brief Preprocess the topology for the atoms given to the model, and return the bonds they cut.
 *
 * With several sites, \p sites gives the site of each atom, and only the pairs within a site
 * are excluded.
 */
std::vector<LinkFrontierAtom> preprocessTopology(gmx_mtop_t*           mtop,
                                                 ArrayRef<const Index> mtaIndices,
                                                 ArrayRef<const int>   sites,
                                                 int                   numSites,
                                                 bool                  oniom,
                                                 const MDLogger&       logger,
                                                 WarningHandler*       wi)
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

    // 2) Exclude non-bonded interactions between QM atoms, within each site with several sites
    if (numSites > 1)
    {
        checkNoBondsBetweenSites(*mtop, mtaIndices, sites);
        addSiteExclusions(mtop, mtaIndices, sites, numSites, logger);
    }
    else
    {
        addEmbeddedNBExclusions(mtop, mtaIndicesSet, logger);
    }

    // 3) Build atomNumbers vector with atomic numbers of all atoms
    std::vector<int> atomNumbers = buildEmbeddedAtomNumbers(*mtop);

    // 4) Make F_CONNBOND between atoms within QM region
    modifyEmbeddedTwoCenterInteractions(mtop, mtaIndicesSet, isMTABlock, logger);

    // 5) and 6) Remove angles and settles containing 2 or more QM atoms, and dihedrals
    // containing 3 or more; with ONIOM, only the angles and dihedrals among QM atoms
    const auto boundaryTerms = oniom ? EmbeddedBoundaryTerms::Keep : EmbeddedBoundaryTerms::Remove;
    modifyEmbeddedThreeCenterInteractions(mtop, mtaIndicesSet, isMTABlock, logger, boundaryTerms);
    modifyEmbeddedFourCenterInteractions(mtop, mtaIndicesSet, isMTABlock, logger, boundaryTerms);

    // 7) Check for constrained bonds in subsystem
    checkConstrainedBonds(mtop, mtaIndicesSet, isMTABlock, wi);

    // 8) The bonds between QM and MM atoms; the caller logs what it does with them
    const MDLogger silent;
    auto           linkFrontier = buildLinkFrontier(mtop, mtaIndicesSet, isMTABlock, silent);

    // finalize topology
    mtop->finalize();
    return linkFrontier;
}
//! metatomic-uncertainty-threshold is off or a non-negative number of kJ/mol.
void checkUncertaintyThreshold(const std::string& value)
{
    if (value == "off")
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
            "metatomic-uncertainty-threshold must be off or a non-negative number in kJ/mol, "
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
    size_t                             position = 0;
    const auto isSpace   = [](char c) { return std::isspace(static_cast<unsigned char>(c)) != 0; };
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
            modelOptionsError(
                    "expected key=value with an alphanumeric, underscore, hyphen or dot key.");
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
                    if (position == text.size() || (text[position] != quote && text[position] != '\\'))
                    {
                        modelOptionsError(
                                "inside quotes, escape only the matching quote or a backslash.");
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
    addMdpTransformFromString<std::string>(rules, stringIdentityTransform, METATOMIC_MODULE_NAME, PLUGIN_TAG);
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
    addMdpTransformFromString<bool>(rules, &fromStdString<bool>, METATOMIC_MODULE_NAME, NON_CONSERVATIVE_TAG);
    addMdpTransformFromString<std::string>(
            rules, stringIdentityTransform, METATOMIC_MODULE_NAME, VARIANT_NC_FORCES_TAG);
    addMdpTransformFromString<std::string>(
            rules, stringIdentityTransform, METATOMIC_MODULE_NAME, VARIANT_NC_STRESS_TAG);
    addMdpTransformFromString<bool>(rules, &fromStdString<bool>, METATOMIC_MODULE_NAME, ONIOM_TAG);
    addMdpTransformFromString<bool>(rules, &fromStdString<bool>, METATOMIC_MODULE_NAME, LINK_ATOMS_TAG);
    for (const auto& tag : { SITE_GROUPS_TAG, SITE_CHARGES_TAG, SITE_SPINS_TAG })
    {
        addMdpTransformFromString<std::string>(rules, stringIdentityTransform, METATOMIC_MODULE_NAME, tag);
    }
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
    section.addOption(BooleanOption(ONIOM_TAG.c_str()).store(&params_.oniom));
    section.addOption(BooleanOption(LINK_ATOMS_TAG.c_str()).store(&params_.linkAtoms));
    section.addOption(StringOption(SITE_GROUPS_TAG.c_str()).store(&params_.siteGroups));
    section.addOption(StringOption(SITE_CHARGES_TAG.c_str()).store(&params_.siteCharges));
    section.addOption(StringOption(SITE_SPINS_TAG.c_str()).store(&params_.siteSpinMultiplicities));
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
        addMdpOutputValue<std::string>(builder, METATOMIC_MODULE_NAME, PLUGIN_TAG, params_.pluginPath);
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
        addMdpOutputValue<bool>(builder, METATOMIC_MODULE_NAME, ONIOM_TAG, params_.oniom);
        addMdpOutputValue<bool>(builder, METATOMIC_MODULE_NAME, LINK_ATOMS_TAG, params_.linkAtoms);
        addMdpOutputValue<std::string>(builder, METATOMIC_MODULE_NAME, SITE_GROUPS_TAG, params_.siteGroups);
        addMdpOutputValue<std::string>(builder, METATOMIC_MODULE_NAME, SITE_CHARGES_TAG, params_.siteCharges);
        addMdpOutputValue<std::string>(
                builder, METATOMIC_MODULE_NAME, SITE_SPINS_TAG, params_.siteSpinMultiplicities);
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

    // Sites: every input atom in exactly one site group; one site without groups
    params_.mtaSites_.clear();
    const auto siteNames = splitString(params_.siteGroups);
    const int  numSites  = siteNames.empty() ? 1 : static_cast<int>(siteNames.size());
    if (!siteNames.empty())
    {
        std::map<Index, int> siteOf;
        for (int s = 0; s < numSites; s++)
        {
            for (const Index atom : indexGroupsAndNames.indices(siteNames[s]))
            {
                if (!siteOf.emplace(atom, s).second)
                {
                    GMX_THROW(InconsistentInputError(formatString(
                            "Atom %d is in more than one of metatomic-site-groups (%s and %s).",
                            static_cast<int>(atom) + 1,
                            siteNames[siteOf[atom]].c_str(),
                            siteNames[s].c_str())));
                }
            }
        }
        for (const Index atom : params_.mtaIndices_)
        {
            const auto it = siteOf.find(atom);
            if (it == siteOf.end())
            {
                GMX_THROW(InconsistentInputError(
                        formatString("Atom %d of %s is in none of metatomic-site-groups.",
                                     static_cast<int>(atom) + 1,
                                     params_.inputGroup.c_str())));
            }
            params_.mtaSites_.push_back(it->second);
        }
        if (siteOf.size() != params_.mtaIndices_.size())
        {
            GMX_THROW(InconsistentInputError(formatString(
                    "metatomic-site-groups contain %zu atoms, but %s has %zu; the sites must "
                    "divide the input group.",
                    siteOf.size(),
                    params_.inputGroup.c_str(),
                    params_.mtaIndices_.size())));
        }
    }

    // One value per site, or one value for all
    const auto perSite = [numSites](const std::string& option, const std::string& text, const auto fallback)
    {
        using T = std::decay_t<decltype(fallback)>;
        std::vector<T> values;
        for (const auto& word : splitString(text))
        {
            try
            {
                std::size_t  used  = 0;
                const double value = std::stod(word, &used);
                if (used != word.size() || (std::is_integral_v<T> && value != std::round(value)))
                {
                    throw std::invalid_argument(word);
                }
                values.push_back(static_cast<T>(value));
            }
            catch (const std::exception&)
            {
                GMX_THROW(InconsistentInputError(formatString(
                        "Cannot read '%s' in metatomic-%s.", word.c_str(), option.c_str())));
            }
        }
        if (values.empty())
        {
            values.assign(numSites, fallback);
        }
        else if (values.size() == 1)
        {
            // a copy, as assign() overwrites the value it would read
            const T value = values[0];
            values.assign(numSites, value);
        }
        else if (static_cast<int>(values.size()) != numSites)
        {
            GMX_THROW(InconsistentInputError(formatString(
                    "metatomic-%s has %zu values for %d sites; give one value, or one per site.",
                    option.c_str(),
                    values.size(),
                    numSites)));
        }
        return values;
    };
    params_.siteChargeValues_ = perSite(SITE_CHARGES_TAG, params_.siteCharges, real(0));
    params_.siteSpinValues_   = perSite(SITE_SPINS_TAG, params_.siteSpinMultiplicities, int(1));
    for (const int multiplicity : params_.siteSpinValues_)
    {
        if (multiplicity < 1)
        {
            GMX_THROW(InconsistentInputError("metatomic-site-spin-multiplicities must be >= 1."));
        }
    }
}

void MetatomicOptions::modifyTopology(gmx_mtop_t* top)
{
    if (!params_.active)
    {
        return;
    }
    if (params_.linkAtoms && !params_.oniom)
    {
        GMX_THROW(InconsistentInputError("metatomic-link-atoms requires metatomic-oniom = yes."));
    }
    const auto cutBonds = preprocessTopology(
            top, params_.mtaIndices_, params_.mtaSites_, params_.numSites(), params_.oniom, logger(), wi_);
    if (params_.oniom && !params_.linkAtoms && !cutBonds.empty() && wi_ != nullptr)
    {
        wi_->addWarning(formatString(
                "%zu covalent bonds between atoms of metatomic-input-group and other atoms are cut "
                "(the first between atoms %d and %d), but metatomic-link-atoms = no. The model "
                "then sees the atoms of the group at the cut with a missing neighbour, a dangling "
                "bond, which gives wrong energies and forces. Set metatomic-link-atoms = yes to "
                "cap these bonds with hydrogen atoms, or add the bonded atoms to the group.",
                cutBonds.size(),
                cutBonds.front().getEmbeddedIndex() + 1,
                cutBonds.front().getMMIndex() + 1));
    }
    if (params_.linkAtoms)
    {
        // The model sees a cap in place of each MM atom of a cut bond, so these atoms join the
        // atoms it sees, after those of the input group. The topology keeps their classical
        // interactions, including the bond.
        params_.linkFrontier_ = cutBonds;
        // the MM atom of a cap is in the site of its embedded atom
        std::map<Index, int> siteOf;
        for (size_t k = 0; k < params_.mtaIndices_.size(); k++)
        {
            siteOf.emplace(params_.mtaIndices_[k], params_.siteOf(k));
        }
        std::map<Index, int> capped;
        for (const auto& bond : cutBonds)
        {
            const int site            = siteOf.at(bond.getEmbeddedIndex());
            const auto [it, inserted] = capped.emplace(bond.getMMIndex(), site);
            if (!inserted && it->second != site)
            {
                GMX_THROW(InconsistentInputError(formatString(
                        "Atom %d is bonded to atoms of two sites; it cannot carry link atoms of "
                        "both.",
                        bond.getMMIndex() + 1)));
            }
        }
        for (const auto& [atom, site] : capped)
        {
            params_.mtaIndices_.push_back(atom);
            if (!params_.mtaSites_.empty())
            {
                params_.mtaSites_.push_back(site);
            }
        }
        GMX_LOG(logger().info)
                .appendTextFormatted("Metatomic: %zu link atoms cap the bonds to %zu MM atoms",
                                     cutBonds.size(),
                                     capped.size());
    }
}

void MetatomicOptions::addExclusionDistanceExemptions(ExclusionDistanceExemptions* exemptions) const
{
    if (!params_.active || !params_.oniom)
    {
        return;
    }
    std::vector<std::vector<int>> sites(params_.numSites());
    for (int k = 0; k < params_.numEmbeddedAtoms(); k++)
    {
        sites[params_.siteOf(k)].push_back(static_cast<int>(params_.mtaIndices_[k]));
    }
    for (auto& site : sites)
    {
        exemptions->groups.push_back(std::move(site));
    }
}

int MetatomicParameters::numEmbeddedAtoms() const
{
    std::set<int> capped;
    for (const auto& bond : linkFrontier_)
    {
        capped.insert(bond.getMMIndex());
    }
    return static_cast<int>(mtaIndices_.size() - capped.size());
}

void MetatomicOptions::writeParamsToKvt(KeyValueTreeObjectBuilder treeBuilder)
{
    if (!params_.active)
    {
        return;
    }
    // grompp rejects what mdrun would only find when it loads the model
    checkUncertaintyThreshold(params_.uncertaintyThreshold);
    const auto modelOptions       = parseMetatomicModelOptions(params_.modelOptions);
    auto       modelOptionsObject = treeBuilder.addObject("metatomic-model-loading-options");
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

    // The site of each atom given to the model, and the charge and spin multiplicity of each site
    if (!params_.mtaSites_.empty())
    {
        auto sitesAdder = treeBuilder.addUniformArray<std::int64_t>(SITES_KEY);
        for (const int site : params_.mtaSites_)
        {
            sitesAdder.addValue(site);
        }
    }
    auto chargeAdder = treeBuilder.addUniformArray<real>(METATOMIC_MODULE_NAME + "-" + SITE_CHARGES_TAG);
    for (const real charge : params_.siteChargeValues_)
    {
        chargeAdder.addValue(charge);
    }
    auto spinAdder =
            treeBuilder.addUniformArray<std::int64_t>(METATOMIC_MODULE_NAME + "-" + SITE_SPINS_TAG);
    for (const int multiplicity : params_.siteSpinValues_)
    {
        spinAdder.addValue(multiplicity);
    }

    // The bonds that link atoms cap, as (embedded, MM) global atom indices
    if (!params_.linkFrontier_.empty())
    {
        auto linkAdder = treeBuilder.addUniformArray<std::int64_t>(LINK_FRONTIER_KEY);
        for (const auto& bond : params_.linkFrontier_)
        {
            linkAdder.addValue(bond.getEmbeddedIndex());
            linkAdder.addValue(bond.getMMIndex());
        }
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
        GMX_THROW(InconsistentInputError(
                "Cannot find metatomic model-loading options in the tpr file."));
    }
    std::ostringstream options;
    for (const auto& property : tree["metatomic-model-loading-options"].asObject().properties())
    {
        options << property.key() << '=' << std::quoted(property.value().cast<std::string>()) << ' ';
    }
    params_.modelOptions = options.str();

    // Sites; without them, one site of charge 0 and multiplicity 1
    params_.mtaSites_.clear();
    params_.siteChargeValues_ = { 0 };
    params_.siteSpinValues_   = { 1 };
    if (tree.keyExists(SITES_KEY))
    {
        for (const auto& value : tree[SITES_KEY].asArray().values())
        {
            params_.mtaSites_.push_back(static_cast<int>(value.cast<std::int64_t>()));
        }
    }
    const std::string chargesKey = METATOMIC_MODULE_NAME + "-" + SITE_CHARGES_TAG;
    if (tree.keyExists(chargesKey))
    {
        params_.siteChargeValues_.clear();
        for (const auto& value : tree[chargesKey].asArray().values())
        {
            params_.siteChargeValues_.push_back(value.cast<real>());
        }
    }
    const std::string spinsKey = METATOMIC_MODULE_NAME + "-" + SITE_SPINS_TAG;
    if (tree.keyExists(spinsKey))
    {
        params_.siteSpinValues_.clear();
        for (const auto& value : tree[spinsKey].asArray().values())
        {
            params_.siteSpinValues_.push_back(static_cast<int>(value.cast<std::int64_t>()));
        }
    }
    if ((!params_.mtaSites_.empty() && params_.mtaSites_.size() != params_.mtaIndices_.size())
        || params_.siteChargeValues_.size() != params_.siteSpinValues_.size()
        || std::any_of(params_.mtaSites_.begin(),
                       params_.mtaSites_.end(),
                       [this](int site) { return site < 0 || site >= params_.numSites(); }))
    {
        GMX_THROW(InconsistentInputError("Inconsistent metatomic site data in the tpr file."));
    }

    params_.linkFrontier_.clear();
    if (tree.keyExists(LINK_FRONTIER_KEY))
    {
        const auto bonds = tree[LINK_FRONTIER_KEY].asArray().values();
        for (size_t i = 0; i + 1 < bonds.size(); i += 2)
        {
            params_.linkFrontier_.emplace_back(static_cast<int>(bonds[i].cast<std::int64_t>()),
                                               static_cast<int>(bonds[i + 1].cast<std::int64_t>()));
        }
    }
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
    params_.atoms_           = gmx_mtop_global_atoms(top);
    params_.numAtoms_        = params_.atoms_.nr;
    params_.numEnergyGroups_ = top.groups.groups[SimulationAtomGroupType::EnergyOutput].size();
    params_.energyGroups_.resize(params_.numAtoms_);
    for (int i = 0; i < params_.numAtoms_; i++)
    {
        params_.energyGroups_[i] = getGroupType(top.groups, SimulationAtomGroupType::EnergyOutput, i);
    }
}

void MetatomicOptions::setPbcType(const PbcType& pbcType)
{
    params_.pbcType_ = std::make_unique<PbcType>(pbcType);
}

void MetatomicOptions::setPressureCoupling(const PressureCoupling& pressureCoupling)
{
    params_.pressureCoupling = pressureCoupling != PressureCoupling::No;
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
