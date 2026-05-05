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

#include <set>

#include "gromacs/domdec/localatomset.h"
#include "gromacs/fileio/warninp.h"
#include "gromacs/mdtypes/imdpoptionprovider_helpers.h"
#include "gromacs/options/basicoptions.h"
#include "gromacs/options/optionsection.h"
#include "gromacs/selection/indexutil.h"
#include "gromacs/topology/embedded_system_preprocessing.h"
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
static const std::string LINK_ATOMS_TAG            = "link-atoms";
static const std::string MM_CHARGES_TAG            = "mm-charges";

namespace
{
// TODO(rg): this is duplicated from the nnpotoptions

//! \brief Helper function to preprocess topology for MTA
void preprocessTopology(gmx_mtop_t*                    mtop,
                        ArrayRef<const Index>           mtaIndices,
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

    // 2) Exclude non-bonded interactions between QM atoms
    addEmbeddedNBExclusions(mtop, mtaIndicesSet, logger);

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
            rules, &fromStdString<bool>, METATOMIC_MODULE_NAME, LINK_ATOMS_TAG);
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
    section.addOption(BooleanOption(LINK_ATOMS_TAG.c_str()).store(&params_.linkAtoms));
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
        addMdpOutputValue<bool>(
                builder, METATOMIC_MODULE_NAME, LINK_ATOMS_TAG, params_.linkAtoms);
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

        // Scan bonds to find direct MM neighbors of ML atoms
        std::set<int> boundaryMM;
        for (size_t mb = 0; mb < top->molblock.size(); ++mb)
        {
            const auto& moltype = top->moltype[top->molblock[mb].type];
            int start = top->moleculeBlockIndices[mb].globalAtomStart;

            for (const auto ftype : gmx::EnumerationWrapper<InteractionFunction>{})
            {
                if (!(interaction_function[ftype].flags & IF_CHEMBOND) || NRAL(ftype) != 2
                    || moltype.ilist[ftype].empty())
                {
                    continue;
                }
                for (int j = 0; j < moltype.ilist[ftype].size(); j += 3)
                {
                    int a1 = moltype.ilist[ftype].iatoms[j + 1] + start;
                    int a2 = moltype.ilist[ftype].iatoms[j + 2] + start;
                    bool a1_ml = origMtaSet.count(a1) > 0;
                    bool a2_ml = origMtaSet.count(a2) > 0;
                    if (a1_ml && !a2_ml && boundaryMM.count(a2) == 0)
                    {
                        boundaryMM.insert(a2);
                        // Store as LinkFrontierAtom: a1=embedded, a2=MM
                        params_.linkFrontier_.emplace_back(a1, a2);
                    }
                    else if (a2_ml && !a1_ml && boundaryMM.count(a1) == 0)
                    {
                        boundaryMM.insert(a1);
                        params_.linkFrontier_.emplace_back(a2, a1);
                    }
                }
            }
        }

        // Add boundary MM atoms to the embedded set
        for (int mmIdx : boundaryMM)
        {
            params_.mtaIndices_.push_back(mmIdx);
        }

        GMX_LOG(logger().info)
                .appendTextFormatted("Metatomic: expanded embedded set from %zu to %zu atoms "
                                     "(%zu boundary MM for link atoms)",
                                     origMtaSet.size(),
                                     params_.mtaIndices_.size(),
                                     boundaryMM.size());
    }

    // Run topology surgery on the (possibly expanded) embedded set
    preprocessTopology(top, params_.mtaIndices_, logger(), wi_,
                       /*buildLinks=*/false, nullptr);
    // Note: buildLinkFrontier is not called inside preprocessTopology because
    // we already built it above (and with the expanded set, there are no
    // cut bonds -- all boundary atoms are now embedded).
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
