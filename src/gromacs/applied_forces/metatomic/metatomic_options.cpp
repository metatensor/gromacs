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

#include <array>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "gromacs/domdec/localatomset.h"
#include "gromacs/fileio/warninp.h"
#include "gromacs/mdrunutility/mdmodulesnotifiers.h"
#include "gromacs/mdtypes/imdpoptionprovider_helpers.h"
#include "gromacs/options/basicoptions.h"
#include "gromacs/options/optionsection.h"
#include "gromacs/selection/indexutil.h"
#include "gromacs/topology/embedded_system_preprocessing.h"
#include "gromacs/topology/exclusionblocks.h"
#include "gromacs/topology/idef.h"
#include "gromacs/topology/ifunc.h"
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
static const std::string EXTENSIONS_DIRECTORY_TAG = "extensions";
static const std::string CHECK_CONSISTENCY_TAG    = "check-consistency";
static const std::string DEVICE_TAG               = "device";
static const std::string VARIANT_TAG              = "variant";
static const std::string UNCERTAINTY_THRESHOLD_TAG = "uncertainty-threshold";
static const std::string VARIANT_ENERGY_UQ_TAG     = "variant-energy-uq";
static const std::string NON_CONSERVATIVE_TAG      = "non-conservative";
static const std::string VARIANT_NC_FORCES_TAG     = "variant-nc-forces";
static const std::string VARIANT_NC_STRESS_TAG     = "variant-nc-stress";
static const std::string ONIOM_TAG                  = "oniom";
static const std::string LINK_ATOMS_TAG            = "link-atoms";
static const std::string MM_CHARGES_TAG            = "mm-charges";
static const std::string SITE_GROUPS_TAG           = "site-groups";
static const std::string SITE_CHARGES_TAG          = "site-charges";
static const std::string SITE_SPINS_TAG            = "site-spin-multiplicities";

namespace
{
// TODO(rg): this is duplicated from the nnpotoptions

void addLinkFrontierAtom(std::set<int>*                 boundaryMM,
                         std::vector<LinkFrontierAtom>* linkFrontier,
                         int                            embeddedIndex,
                         int                            mmIndex)
{
    boundaryMM->insert(mmIndex);
    linkFrontier->emplace_back(embeddedIndex, mmIndex);
}

//! Returns the chemical bonds between an atom in \p mlSet and one outside it, as (ML, MM) pairs
std::vector<std::pair<int, int>> findCutBonds(const gmx_mtop_t& mtop, const std::set<int>& mlSet)
{
    // This runs before splitEmbeddedBlocks, so a block may still hold many
    // molecules of the same type (one block of 34 lipids, say). Every
    // molecule in the block therefore has to be visited with its own atom
    // offset: using only the block's globalAtomStart finds the boundary of
    // the first molecule and silently misses all the others.
    std::vector<std::pair<int, int>> cut;
    for (size_t mb = 0; mb < mtop.molblock.size(); ++mb)
    {
        const auto& moltype     = mtop.moltype[mtop.molblock[mb].type];
        const int   blockStart  = mtop.moleculeBlockIndices[mb].globalAtomStart;
        const int   numAtomsMol = moltype.atoms.nr;

        for (int mol = 0; mol < mtop.molblock[mb].nmol; ++mol)
        {
            const int start = blockStart + mol * numAtomsMol;

            for (const auto ftype : gmx::EnumerationWrapper<InteractionFunction>{})
            {
                if (!(interaction_function[ftype].flags & IF_CHEMBOND) || NRAL(ftype) != 2
                    || moltype.ilist[ftype].empty())
                {
                    continue;
                }
                for (int j = 0; j < moltype.ilist[ftype].size(); j += 3)
                {
                    const int  a1    = moltype.ilist[ftype].iatoms[j + 1] + start;
                    const int  a2    = moltype.ilist[ftype].iatoms[j + 2] + start;
                    const bool a1_ml = mlSet.count(a1) > 0;
                    const bool a2_ml = mlSet.count(a2) > 0;
                    if (a1_ml != a2_ml)
                    {
                        cut.emplace_back(a1_ml ? a1 : a2, a1_ml ? a2 : a1);
                    }
                }
            }
        }
    }
    return cut;
}

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
    // (block, molecule in block, atom in molecule) of every atom, from the split blocks;
    // mtop->moleculeBlockIndices is only updated by finalize()
    const auto locate = [mtop](const Index atom)
    {
        Index start = 0;
        for (size_t b = 0; b < mtop->molblock.size(); b++)
        {
            const int   numAtoms = mtop->moltype[mtop->molblock[b].type].atoms.nr;
            const Index size     = Index(mtop->molblock[b].nmol) * numAtoms;
            if (atom < start + size)
            {
                return std::array<int, 3>{ int(b), int((atom - start) / numAtoms), int((atom - start) % numAtoms) };
            }
            start += size;
        }
        GMX_THROW(InternalError("Embedded atom index beyond the topology"));
    };

    std::vector<std::vector<int>> siteAtoms(numSites);
    std::vector<int>              siteBlock(numSites, -1);
    for (size_t k = 0; k < mtaIndices.size(); k++)
    {
        const auto [b, mol, local] = locate(mtaIndices[k]);
        const int s                = sites[k];
        if (siteBlock[s] >= 0 && siteBlock[s] != b)
        {
            GMX_THROW(InconsistentInputError(formatString(
                    "ML site %d spans more than one molecule; with several "
                    "metatomic-site-groups each site must lie within one molecule.",
                    s + 1)));
        }
        if (mtop->molblock[b].nmol != 1 || mol != 0)
        {
            GMX_THROW(InternalError("Expected each molecule with embedded atoms in its own block"));
        }
        siteBlock[s] = b;
        siteAtoms[s].push_back(local);
    }

    int numPairs = 0;
    std::map<int, std::vector<ExclusionBlock>> newExclusions;
    for (int s = 0; s < numSites; s++)
    {
        auto& moltype = mtop->moltype[mtop->molblock[siteBlock[s]].type];
        auto& blocks  = newExclusions[siteBlock[s]];
        blocks.resize(moltype.atoms.nr);
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
    for (auto& [b, blocks] : newExclusions)
    {
        mergeExclusions(&mtop->moltype[mtop->molblock[b].type].excls, blocks);
    }
    GMX_LOG(logger.info)
            .appendTextFormatted("Excluded the %d atom pairs within %d ML sites; pairs between "
                                 "sites keep their MM interactions\n",
                                 numPairs,
                                 numSites);
}

//! \brief Helper function to preprocess topology for MTA
void preprocessTopology(gmx_mtop_t*                    mtop,
                        ArrayRef<const Index>           mtaIndices,
                        ArrayRef<const int>             sites,
                        int                             numSites,
                        const MDLogger&                 logger,
                        WarningHandler*                 wi,
                        bool                            buildLinks,
                        std::vector<LinkFrontierAtom>*  linkFrontierOut)
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

    // 2) Exclude non-bonded interactions between QM atoms, per site with several sites
    if (numSites > 1)
    {
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

    // 5) Remove angles and settles containing all-ML atoms (ONIOM)
    modifyEmbeddedThreeCenterInteractions(mtop, mtaIndicesSet, isMTABlock, logger);

    // 6) Remove dihedrals containing all-ML atoms (ONIOM)
    modifyEmbeddedFourCenterInteractions(mtop, mtaIndicesSet, isMTABlock, logger);

    // 7) Check for constrained bonds in subsystem
    checkConstrainedBonds(mtop, mtaIndicesSet, isMTABlock, wi);

    // 8) Build link frontier atoms at ML/MM boundary bonds
    if (buildLinks && linkFrontierOut != nullptr)
    {
        *linkFrontierOut = buildLinkFrontier(mtop, mtaIndicesSet, isMTABlock, logger);
        GMX_LOG(logger.info)
                .appendTextFormatted("Number of link frontier atoms: %zu",
                                     linkFrontierOut->size());
    }

    // finalize topology
    mtop->finalize();
}
} // namespace

void MetatomicOptions::initMdpTransform(IKeyValueTreeTransformRules* rules)
{
    const auto& stringIdentityTransform = [](std::string s) { return s; };
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
    addMdpTransformFromString<bool>(
            rules, &fromStdString<bool>, METATOMIC_MODULE_NAME, ONIOM_TAG);
    addMdpTransformFromString<bool>(
            rules, &fromStdString<bool>, METATOMIC_MODULE_NAME, LINK_ATOMS_TAG);
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
    section.addOption(StringOption(EXTENSIONS_DIRECTORY_TAG.c_str()).store(&params_.extensionsDirectory));
    section.addOption(StringOption(DEVICE_TAG.c_str()).store(&params_.device));
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
        addMdpOutputValue<std::string>(
                builder, METATOMIC_MODULE_NAME, EXTENSIONS_DIRECTORY_TAG, params_.extensionsDirectory);
        addMdpOutputValue<std::string>(builder, METATOMIC_MODULE_NAME, DEVICE_TAG, params_.device);
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
        addMdpOutputValue<bool>(
                builder, METATOMIC_MODULE_NAME, LINK_ATOMS_TAG, params_.linkAtoms);
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
    params_.mtaSites_.assign(params_.mtaIndices_.size(), 0);
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
                            int(atom) + 1,
                            siteNames[siteOf[atom]].c_str(),
                            siteNames[s].c_str())));
                }
            }
        }
        for (size_t k = 0; k < params_.mtaIndices_.size(); k++)
        {
            const auto it = siteOf.find(params_.mtaIndices_[k]);
            if (it == siteOf.end())
            {
                GMX_THROW(InconsistentInputError(formatString(
                        "Atom %d of %s is in none of metatomic-site-groups.",
                        int(params_.mtaIndices_[k]) + 1,
                        params_.inputGroup.c_str())));
            }
            params_.mtaSites_[k] = it->second;
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
        using T            = std::decay_t<decltype(fallback)>;
        const auto words   = splitString(text);
        std::vector<T> out;
        for (const auto& w : words)
        {
            try
            {
                std::size_t used = 0;
                const double v   = std::stod(w, &used);
                if (used != w.size() || (std::is_integral_v<T> && v != std::round(v)))
                {
                    throw std::invalid_argument(w);
                }
                out.push_back(static_cast<T>(v));
            }
            catch (const std::exception&)
            {
                GMX_THROW(InconsistentInputError(
                        formatString("Cannot read '%s' in metatomic-%s.", w.c_str(), option.c_str())));
            }
        }
        if (out.empty())
        {
            out.assign(numSites, fallback);
        }
        else if (out.size() == 1)
        {
            out.assign(numSites, out[0]);
        }
        else if (static_cast<int>(out.size()) != numSites)
        {
            GMX_THROW(InconsistentInputError(formatString(
                    "metatomic-%s has %zu values for %d sites; give one value, or one per site.",
                    option.c_str(),
                    out.size(),
                    numSites)));
        }
        return out;
    };
    params_.siteChargeValues_ = perSite(SITE_CHARGES_TAG, params_.siteCharges, real(0));
    params_.siteSpinValues_   = perSite(SITE_SPINS_TAG, params_.siteSpinMultiplicities, int(1));
    for (const int m : params_.siteSpinValues_)
    {
        if (m < 1)
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

    params_.mmCharges_.clear();
    params_.linkFrontier_.clear();

    // Topology charges come from the unmodified topology so model-requested
    // charge inputs remain independent of embedded-system preprocessing.
    for (const auto& molblock : top->molblock)
    {
        const auto& moltype = top->moltype[molblock.type];
        for (int m = 0; m < molblock.nmol; m++)
        {
            for (int a = 0; a < moltype.atoms.nr; a++)
            {
                params_.mmCharges_.push_back(moltype.atoms.atom[a].q);
            }
        }
    }

    if (!params_.oniom)
    {
        if (params_.linkAtoms)
        {
            GMX_THROW(InconsistentInputError(
                    "metatomic-link-atoms requires metatomic-oniom = yes."));
        }

        GMX_LOG(logger().info)
                .appendText("Metatomic potential interface is active, topology was not modified.");
        return;
    }

    if (!params_.linkAtoms && wi_ != nullptr)
    {
        const std::set<int> mlSet(params_.mtaIndices_.begin(), params_.mtaIndices_.end());
        const auto          cut = findCutBonds(*top, mlSet);
        if (!cut.empty())
        {
            wi_->addWarning(formatString(
                    "%zu covalent bond(s) between ML and MM atoms are cut (the first between "
                    "atoms %d and %d, 1-based), but metatomic-link-atoms = no. The model then "
                    "sees the ML atoms at the cut with a missing neighbour (a dangling bond), "
                    "which gives wrong energies and forces. Set metatomic-link-atoms = yes to "
                    "cap them with hydrogens, or include the bonded MM atoms in the ML group.",
                    cut.size(),
                    cut.front().first + 1,
                    cut.front().second + 1));
        }
    }

    if (params_.linkAtoms)
    {
        // NNPot-style: identify boundary MM atoms first (by scanning bonds
        // between ML and non-ML atoms), add them to the embedded set, THEN
        // run topology surgery on the expanded set.  This ensures:
        // - NB exclusions include boundary MM atoms (no double-counting)
        // - Bonded terms between ML and boundary-MM are properly handled
        // - buildLinkFrontier finds zero cut bonds (all boundary atoms are embedded)
        //
        // The link frontier is built from the ORIGINAL ML set (before expansion)
        // so we know which embedded atoms are "real ML" vs "boundary MM".
        std::set<int> origMtaSet(params_.mtaIndices_.begin(), params_.mtaIndices_.end());

        std::set<int> boundaryMM;
        for (const auto& [embedded, mm] : findCutBonds(*top, origMtaSet))
        {
            addLinkFrontierAtom(&boundaryMM, &params_.linkFrontier_, embedded, mm);
        }

        // Add boundary MM atoms to the embedded set, in the site of their embedded partner
        std::map<Index, int> siteOfOriginal;
        for (size_t k = 0; k < params_.mtaIndices_.size(); k++)
        {
            siteOfOriginal[params_.mtaIndices_[k]] = params_.mtaSites_[k];
        }
        std::map<int, int> siteOfBoundary;
        for (const auto& link : params_.linkFrontier_)
        {
            const int s                       = siteOfOriginal.at(link.getEmbeddedIndex());
            const auto [it, inserted] = siteOfBoundary.emplace(link.getMMIndex(), s);
            if (!inserted && it->second != s)
            {
                GMX_THROW(InconsistentInputError(formatString(
                        "MM atom %d is bonded to ML atoms of two sites; it cannot be a link "
                        "atom of both.",
                        link.getMMIndex() + 1)));
            }
        }
        for (int mmIdx : boundaryMM)
        {
            params_.mtaIndices_.push_back(mmIdx);
            params_.mtaSites_.push_back(siteOfBoundary.at(mmIdx));
        }

        GMX_LOG(logger().info)
                .appendTextFormatted("Metatomic: expanded embedded set from %zu to %zu atoms "
                                     "(%zu boundary MM for link atoms)",
                                     origMtaSet.size(),
                                     params_.mtaIndices_.size(),
                                     boundaryMM.size());
    }

    // Run topology surgery on the (possibly expanded) embedded set
    preprocessTopology(top, params_.mtaIndices_, params_.mtaSites_, params_.numSites(), logger(), wi_,
                       /*buildLinks=*/false, nullptr);
    // Note: buildLinkFrontier is not called inside preprocessTopology because
    // we already built it above (and with the expanded set, there are no
    // cut bonds -- all boundary atoms are now embedded).
}

void MetatomicOptions::addExclusionDistanceExemptions(ExclusionDistanceExemptions* exemptions) const
{
    if (!params_.active || !params_.oniom)
    {
        return;
    }
    std::vector<std::vector<int>> sites(std::max(params_.numSites(), 1));
    for (size_t k = 0; k < params_.mtaIndices_.size(); k++)
    {
        sites[params_.mtaSites_[k]].push_back(static_cast<int>(params_.mtaIndices_[k]));
    }
    for (auto& site : sites)
    {
        exemptions->groups.push_back(std::move(site));
    }
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

    if (!params_.mmCharges_.empty())
    {
        auto chargesAdder =
                treeBuilder.addUniformArray<real>(METATOMIC_MODULE_NAME + "-" + MM_CHARGES_TAG);
        for (const auto& charge : params_.mmCharges_)
        {
            chargesAdder.addValue(charge);
        }
    }

    // Sites: the site of each embedded atom, and the charge and spin multiplicity of each site
    {
        auto sitesAdder = treeBuilder.addUniformArray<std::int64_t>(METATOMIC_MODULE_NAME + "-sites");
        for (const int s : params_.mtaSites_)
        {
            sitesAdder.addValue(s);
        }
        auto chargeAdder = treeBuilder.addUniformArray<real>(METATOMIC_MODULE_NAME + "-" + SITE_CHARGES_TAG);
        for (const real q : params_.siteChargeValues_)
        {
            chargeAdder.addValue(q);
        }
        auto spinAdder = treeBuilder.addUniformArray<std::int64_t>(METATOMIC_MODULE_NAME + "-" + SITE_SPINS_TAG);
        for (const int m : params_.siteSpinValues_)
        {
            spinAdder.addValue(m);
        }
    }

    // Serialize link frontier as flat [embIdx, mmIdx, ...] pairs
    if (!params_.linkFrontier_.empty())
    {
        auto linkAdder = treeBuilder.addUniformArray<std::int64_t>(
                METATOMIC_MODULE_NAME + "-link-frontier");
        for (const auto& link : params_.linkFrontier_)
        {
            linkAdder.addValue(link.getEmbeddedIndex());
            linkAdder.addValue(link.getMMIndex());
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

    std::string chargeKey = METATOMIC_MODULE_NAME + "-" + MM_CHARGES_TAG;
    if (tree.keyExists(chargeKey))
    {
        auto chargeArray = tree[chargeKey].asArray().values();
        params_.mmCharges_.resize(chargeArray.size());
        std::transform(std::begin(chargeArray),
                       std::end(chargeArray),
                       std::begin(params_.mmCharges_),
                       [](const KeyValueTreeValue& val) { return val.cast<real>(); });
    }

    // Sites; tpr files without them have one site with charge 0 and multiplicity 1
    params_.mtaSites_.assign(params_.mtaIndices_.size(), 0);
    params_.siteChargeValues_ = { 0 };
    params_.siteSpinValues_   = { 1 };
    const auto readArray = [&tree](const std::string& k, auto* out)
    {
        using T = typename std::decay_t<decltype(*out)>::value_type;
        if (!tree.keyExists(k))
        {
            return;
        }
        out->clear();
        for (const auto& v : tree[k].asArray().values())
        {
            if constexpr (std::is_integral_v<T>)
            {
                out->push_back(static_cast<T>(v.cast<std::int64_t>()));
            }
            else
            {
                out->push_back(v.cast<real>());
            }
        }
    };
    readArray(METATOMIC_MODULE_NAME + "-sites", &params_.mtaSites_);
    readArray(METATOMIC_MODULE_NAME + "-" + SITE_CHARGES_TAG, &params_.siteChargeValues_);
    readArray(METATOMIC_MODULE_NAME + "-" + SITE_SPINS_TAG, &params_.siteSpinValues_);
    if (params_.mtaSites_.size() != params_.mtaIndices_.size()
        || params_.siteChargeValues_.size() != params_.siteSpinValues_.size())
    {
        GMX_THROW(InconsistentInputError("Inconsistent metatomic site data in the tpr file."));
    }

    // Deserialize link frontier
    std::string linkKey = METATOMIC_MODULE_NAME + "-link-frontier";
    if (tree.keyExists(linkKey))
    {
        auto linkArray = tree[linkKey].asArray().values();
        params_.linkFrontier_.clear();
        for (size_t i = 0; i + 1 < linkArray.size(); i += 2)
        {
            int embIdx = static_cast<int>(linkArray[i].cast<std::int64_t>());
            int mmIdx  = static_cast<int>(linkArray[i + 1].cast<std::int64_t>());
            params_.linkFrontier_.emplace_back(embIdx, mmIdx);
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
    params_.atoms_    = gmx_mtop_global_atoms(top);
    params_.numAtoms_ = params_.atoms_.nr;
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
