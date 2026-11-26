
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
 * Tests for functionality of the MetatomicOptions
 *
 * \author Eric Boittier <eric.boittier@epfl.ch>
 * \author Rohit Goswami <r95g10@gmail.com>
 * \author Philip Loche <ploche@physik.fu-berlin.de>
 * \ingroup module_applied_forces
 */

#include "gmxpre.h"

#include "gromacs/applied_forces/metatomic/metatomic_options.h"

#include "config.h"

#include <algorithm>
#include <filesystem>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <tuple>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "gromacs/applied_forces/nnpot/nnpot.h"
#include "gromacs/domdec/localatomset.h"
#include "gromacs/fileio/confio.h"
#include "gromacs/fileio/warninp.h"
#include "gromacs/gmxpreprocess/grompp.h"
#include "gromacs/mdrunutility/mdmodulesnotifiers.h"
#include "gromacs/mdtypes/imdpoptionprovider_test_helper.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/selection/indexutil.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/topology/index.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/keyvaluetree.h"
#include "gromacs/utility/keyvaluetreebuilder.h"
#include "gromacs/utility/keyvaluetreemdpwriter.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/loggerbuilder.h"
#include "gromacs/utility/stringstream.h"
#include "gromacs/utility/textwriter.h"

#include "testutils/cmdlinetest.h"
#include "testutils/refdata.h"
#include "testutils/testasserts.h"
#include "testutils/testfilemanager.h"
#include "testutils/testmatchers.h"

namespace gmx
{

namespace test
{

// Module name from metatomic_options.cpp
static const std::string METATOMIC_MODULE_NAME = "metatomic";

class MetatomicOptionsTest : public ::testing::Test
{
public:
    static KeyValueTreeObject metatomicBuildDefaultMdpValues()
    {
        // Prepare MDP inputs
        KeyValueTreeBuilder mdpValueBuilder;
        mdpValueBuilder.rootObject().addValue(METATOMIC_MODULE_NAME + "-active", std::string("true"));
        return mdpValueBuilder.build();
    }

    static KeyValueTreeObject metatomicBuildInputMdpValues()
    {
        // Prepare MDP inputs
        KeyValueTreeBuilder mdpValueBuilder;
        mdpValueBuilder.rootObject().addValue(METATOMIC_MODULE_NAME + "-active", std::string("true"));
        mdpValueBuilder.rootObject().addValue(METATOMIC_MODULE_NAME + "-input-group",
                                              std::string("System"));
        mdpValueBuilder.rootObject().addValue(
                METATOMIC_MODULE_NAME + "-model",
                gmx::test::TestFileManager::getInputFilePath("model.pt").string());

        mdpValueBuilder.rootObject().addValue(METATOMIC_MODULE_NAME + "-plugin", std::string("./ext"));
        mdpValueBuilder.rootObject().addValue(METATOMIC_MODULE_NAME + "-check-consistency",
                                              std::string("false"));
        mdpValueBuilder.rootObject().addValue(METATOMIC_MODULE_NAME + "-variant", std::string(""));
        mdpValueBuilder.rootObject().addValue(METATOMIC_MODULE_NAME + "-uncertainty-threshold",
                                              std::string("10"));
        mdpValueBuilder.rootObject().addValue(METATOMIC_MODULE_NAME + "-variant-energy-uq",
                                              std::string(""));
        mdpValueBuilder.rootObject().addValue(METATOMIC_MODULE_NAME + "-non-conservative",
                                              std::string("false"));
        mdpValueBuilder.rootObject().addValue(METATOMIC_MODULE_NAME + "-variant-nc-forces",
                                              std::string(""));
        mdpValueBuilder.rootObject().addValue(METATOMIC_MODULE_NAME + "-variant-nc-stress",
                                              std::string(""));
        return mdpValueBuilder.build();
    }

    static IndexGroupsAndNames indexGroupsAndNamesGeneric()
    {
        // System group is default
        std::vector<IndexGroup> indexGroups;
        indexGroups.push_back({ "A", { 1 } });
        indexGroups.push_back({ "System", { 1, 2, 3 } });
        indexGroups.push_back({ "C", { 2, 3 } });

        return IndexGroupsAndNames(indexGroups);
    }
};

TEST_F(MetatomicOptionsTest, DefaultParameters)
{
    MetatomicOptions                metatomicOptions;
    const MetatomicParameters&      defaultParams = metatomicOptions.parameters();
    gmx::test::TestReferenceData    data;
    gmx::test::TestReferenceChecker checker(data.rootChecker());

    checker.checkBoolean(defaultParams.active, "active");
    checker.checkString(defaultParams.inputGroup, "inputGroup");
    checker.checkString(defaultParams.modelPath_, "modelPath");
    checker.checkString(defaultParams.pluginPath, "pluginPath");
    checker.checkBoolean(defaultParams.checkConsistency, "checkConsistency");
}

TEST_F(MetatomicOptionsTest, OptionSetsActive)
{
    MetatomicOptions metatomicOptions;
    EXPECT_FALSE(metatomicOptions.parameters().active);
    test::fillOptionsFromMdpValues(metatomicBuildDefaultMdpValues(), &metatomicOptions);
    EXPECT_TRUE(metatomicOptions.parameters().active);
}

TEST_F(MetatomicOptionsTest, OutputNoDefaultValuesWhenInactive)
{
    // Transform module data into a flat key-value tree for output.
    StringOutputStream        stream;
    KeyValueTreeBuilder       builder;
    KeyValueTreeObjectBuilder builderObject = builder.rootObject();

    MetatomicOptions metatomicOptions;
    metatomicOptions.buildMdpOutput(&builderObject);
    {
        TextWriter writer(&stream);
        writeKeyValueTreeAsMdp(&writer, builder.build());
    }
    stream.close();

    gmx::test::TestReferenceData    data;
    gmx::test::TestReferenceChecker checker(data.rootChecker());

    checker.checkString(stream.toString(), "Mdp output");
}

TEST_F(MetatomicOptionsTest, OutputDefaultValuesWhenActive)
{
    // Set metatomic-active = true
    MetatomicOptions metatomicOptions;
    test::fillOptionsFromMdpValues(metatomicBuildDefaultMdpValues(), &metatomicOptions);

    // Transform module data into a flat key-value tree for output.
    StringOutputStream        stream;
    KeyValueTreeBuilder       builder;
    KeyValueTreeObjectBuilder builderObject = builder.rootObject();

    metatomicOptions.buildMdpOutput(&builderObject);
    {
        TextWriter writer(&stream);
        writeKeyValueTreeAsMdp(&writer, builder.build());
    }
    stream.close();

    gmx::test::TestReferenceData    data;
    gmx::test::TestReferenceChecker checker(data.rootChecker());

    checker.checkString(stream.toString(), "Mdp output");
}

TEST_F(MetatomicOptionsTest, GromppChecksUncertaintyThresholdAndModelOptions)
{
    MetatomicOptions metatomicOptions;
    fillOptionsFromMdpValues(metatomicBuildInputMdpValues(), &metatomicOptions);
    metatomicOptions.setInputGroupIndices(indexGroupsAndNamesGeneric());

    const auto write = [&metatomicOptions]()
    {
        KeyValueTreeBuilder builder;
        metatomicOptions.writeParamsToKvt(builder.rootObject());
    };
    for (const char* threshold : { "off", "0", "2.5", "10" })
    {
        metatomicOptions.params_.uncertaintyThreshold = threshold;
        EXPECT_NO_THROW(write()) << threshold;
    }
    for (const char* threshold : { "auto", "no", "0.1 eV", "-1", "" })
    {
        metatomicOptions.params_.uncertaintyThreshold = threshold;
        EXPECT_THROW_GMX(write(), InconsistentInputError) << threshold;
    }
    metatomicOptions.params_.uncertaintyThreshold = "10";

    metatomicOptions.params_.modelOptions = "cutoff=5.0";
    EXPECT_NO_THROW(write());
    for (const char* options : { "cutoff", "=5", "cutoff=", "cutoff=5 cutoff=6", "cutoff=\"5" })
    {
        metatomicOptions.params_.modelOptions = options;
        EXPECT_THROW_GMX(write(), InconsistentInputError) << options;
    }
}

TEST(MetatomicModelOptionsTest, ParsesStringsAndQuotedValues)
{
    const auto options = parseMetatomicModelOptions(
            R"(sigma=3 epsilon = 1.2 device=cpu label="argon model" empty='' path='a=b' quote="a\"b" slash="a\\b")");
    EXPECT_EQ(options.at("sigma"), "3");
    EXPECT_EQ(options.at("epsilon"), "1.2");
    EXPECT_EQ(options.at("device"), "cpu");
    EXPECT_EQ(options.at("label"), "argon model");
    EXPECT_EQ(options.at("empty"), "");
    EXPECT_EQ(options.at("path"), "a=b");
    EXPECT_EQ(options.at("quote"), "a\"b");
    EXPECT_EQ(options.at("slash"), "a\\b");
    EXPECT_TRUE(parseMetatomicModelOptions(" \t").empty());
}

TEST(MetatomicModelOptionsTest, RejectsMalformedOptions)
{
    for (const char* text : { "key",
                              "=value",
                              "key=",
                              "key=one key=two",
                              "key='open",
                              "key=\"open",
                              "key='value'tail",
                              "key=un'quoted",
                              "key='a\\q'",
                              "bad/key=value",
                              R"({"key":"value"})" })
    {
        EXPECT_THROW_GMX(parseMetatomicModelOptions(text), InconsistentInputError) << text;
    }
}

/*! \brief Topology of \p simulationName in the simulation database, as grompp writes it
 *
 * grompp reads the mdp file \p mdpContent and, when not empty, the index file \p ndxContent.
 */
static std::unique_ptr<gmx_mtop_t> makeMtopFromFile(const std::string& simulationName,
                                                    const std::string& mdpContent = "",
                                                    const std::string& ndxContent = "")
{
    const std::filesystem::path simData = TestFileManager::getTestSimulationDatabaseDirectory();
    TestFileManager             fileManager;
    const auto                  path = [&](const char* suffix)
    { return fileManager.getTemporaryFilePath(simulationName + suffix).string(); };
    TextWriter::writeFileFromString(path(".mdp"), mdpContent);
    CommandLine grompp;
    grompp.append("grompp");
    grompp.addOption("-f", path(".mdp"));
    grompp.addOption("-p", (simData / simulationName).replace_extension(".top").string());
    grompp.addOption("-c", (simData / simulationName).replace_extension(".gro").string());
    grompp.addOption("-po", path("-mdout.mdp"));
    grompp.addOption("-o", path(".tpr"));
    if (!ndxContent.empty())
    {
        TextWriter::writeFileFromString(path(".ndx"), ndxContent);
        grompp.addOption("-n", path(".ndx"));
    }
    EXPECT_EQ(0, gmx_grompp(grompp.argc(), grompp.argv()));

    bool    fullTopology;
    PbcType pbcType;
    matrix  box;
    auto    mtop = std::make_unique<gmx_mtop_t>();
    readConfAndTopology(path(".tpr").c_str(), &fullTopology, mtop.get(), &pbcType, nullptr, nullptr, box);
    return mtop;
}

/*! \brief Number of interactions of \p numAtoms atoms with at least \p minEmbedded atoms in \p embedded
 *
 * The interactions in all of \p category: bonded interactions by default.
 */
static int countInteractions(const gmx_mtop_t&                mtop,
                             int                              numAtoms,
                             const std::set<int>&             embedded,
                             int                              minEmbedded,
                             InteractionFunctionCategoryFlags category = sc_ifcBondedForce)
{
    int count = 0;
    for (size_t b = 0; b < mtop.molblock.size(); ++b)
    {
        const auto& moltype = mtop.moltype[mtop.molblock[b].type];
        for (int mol = 0; mol < mtop.molblock[b].nmol; ++mol)
        {
            const int start = mtop.moleculeBlockIndices[b].globalAtomStart + mol * moltype.atoms.nr;
            for (const auto ftype : EnumerationWrapper<InteractionFunction>{})
            {
                if ((interaction_function[ftype].categories & category) != category || NRAL(ftype) != numAtoms)
                {
                    continue;
                }
                const auto& iatoms = moltype.ilist[ftype].iatoms;
                for (size_t j = 0; j < iatoms.size(); j += numAtoms + 1)
                {
                    int n = 0;
                    for (int k = 1; k <= numAtoms; ++k)
                    {
                        n += static_cast<int>(embedded.count(iatoms[j + k] + start));
                    }
                    count += static_cast<int>(n >= minEmbedded);
                }
            }
        }
    }
    return count;
}

TEST_F(MetatomicOptionsTest, OniomKeepsTheAnglesAndDihedralsWithAnMMAtom)
{
    // CA, HA and the methyl group of alanine_vacuo
    const std::vector<int> ml = { 8, 9, 10, 11, 12, 13 };
    const std::set<int>    mlSet(ml.begin(), ml.end());
    for (const bool oniom : { false, true })
    {
        SCOPED_TRACE(oniom ? "with ONIOM" : "without ONIOM");
        auto       mtop   = makeMtopFromFile("alanine_vacuo");
        const auto before = [&](int numAtoms, int minEmbedded)
        { return countInteractions(*mtop, numAtoms, mlSet, minEmbedded); };
        const int angles = before(3, 0), boundaryAngles = before(3, 2) - before(3, 3);
        const int dihedrals = before(4, 0), boundaryDihedrals = before(4, 3) - before(4, 4);
        // the test tells the two rules apart only with terms that have both kinds of atoms
        ASSERT_GT(boundaryAngles, 0);
        ASSERT_GT(boundaryDihedrals, 0);
        const int removedAngles    = before(3, 3) + (oniom ? 0 : boundaryAngles);
        const int removedDihedrals = before(4, 4) + (oniom ? 0 : boundaryDihedrals);

        MetatomicOptions options;
        fillOptionsFromMdpValues(metatomicBuildDefaultMdpValues(), &options);
        options.params_.oniom                = oniom;
        const std::vector<IndexGroup> groups = { { "System", ml } };
        options.setInputGroupIndices(IndexGroupsAndNames(groups));
        MDLogger logger;
        options.setLogger(logger);
        WarningHandler wi(true, 0);
        options.setWarningHandler(&wi);
        options.modifyTopology(mtop.get());

        EXPECT_EQ(countInteractions(*mtop, 3, mlSet, 0), angles - removedAngles);
        EXPECT_EQ(countInteractions(*mtop, 4, mlSet, 0), dihedrals - removedDihedrals);
        EXPECT_EQ(countInteractions(*mtop, 3, mlSet, 2), oniom ? boundaryAngles : 0);
        EXPECT_EQ(countInteractions(*mtop, 4, mlSet, 3), oniom ? boundaryDihedrals : 0);
    }
}

TEST_F(MetatomicOptionsTest, LinkAtomsCapTheCutBondsAndKeepThem)
{
    // CA, HA and the methyl group of alanine_vacuo: the bonds CA-N and CA-C are cut
    const std::vector<int> ml = { 8, 9, 10, 11, 12, 13 };
    const std::set<int>    mlSet(ml.begin(), ml.end());
    auto                   mtop          = makeMtopFromFile("alanine_vacuo");
    const auto             boundaryBonds = [&]
    {
        // bonds with parameters, not the connections that replace removed bonds
        const auto bonds = sc_ifcChemicalBond | sc_ifcBType;
        return countInteractions(*mtop, 2, mlSet, 1, bonds) - countInteractions(*mtop, 2, mlSet, 2, bonds);
    };
    ASSERT_EQ(boundaryBonds(), 2);

    MetatomicOptions options;
    fillOptionsFromMdpValues(metatomicBuildDefaultMdpValues(), &options);
    options.params_.oniom                = true;
    options.params_.linkAtoms            = true;
    const std::vector<IndexGroup> groups = { { "System", ml } };
    options.setInputGroupIndices(IndexGroupsAndNames(groups));
    MDLogger logger;
    options.setLogger(logger);
    WarningHandler wi(true, 0);
    options.setWarningHandler(&wi);
    options.modifyTopology(mtop.get());

    // the model sees N and C after the input group, with a cap in place of each
    const auto& params = options.parameters();
    EXPECT_EQ(params.mtaIndices_, (std::vector<Index>{ 8, 9, 10, 11, 12, 13, 6, 14 }));
    EXPECT_EQ(params.numEmbeddedAtoms(), 6);
    ASSERT_EQ(params.linkFrontier_.size(), 2U);
    EXPECT_EQ(params.linkFrontier_[0].getEmbeddedIndex(), 8);
    EXPECT_EQ(params.linkFrontier_[0].getMMIndex(), 6);
    EXPECT_EQ(params.linkFrontier_[1].getEmbeddedIndex(), 8);
    EXPECT_EQ(params.linkFrontier_[1].getMMIndex(), 14);
    // the topology keeps the cut bonds, and the non-bonded pairs of N and C with the input group
    EXPECT_EQ(boundaryBonds(), 2);
    std::vector<int> excluded(mtop->intermolecularExclusionGroup.begin(),
                              mtop->intermolecularExclusionGroup.end());
    std::sort(excluded.begin(), excluded.end());
    EXPECT_EQ(excluded, ml);

    // the tpr file keeps the capped bonds
    KeyValueTreeBuilder builder;
    options.writeParamsToKvt(builder.rootObject());
    options.params_.linkFrontier_.clear();
    options.readParamsFromKvt(builder.build());
    ASSERT_EQ(options.parameters().linkFrontier_.size(), 2U);
    EXPECT_EQ(options.parameters().linkFrontier_[1].getEmbeddedIndex(), 8);
    EXPECT_EQ(options.parameters().linkFrontier_[1].getMMIndex(), 14);
    EXPECT_EQ(options.parameters().numEmbeddedAtoms(), 6);
}

TEST_F(MetatomicOptionsTest, OniomWarnsAboutCutBondsWithoutLinkAtoms)
{
    // CA, HA and the methyl group of alanine_vacuo cut the bonds CA-N and CA-C; the whole
    // molecule cuts none
    std::vector<int> molecule(22);
    std::iota(molecule.begin(), molecule.end(), 0);
    for (const auto& [ml, oniom, linkAtoms, warnings] :
         std::vector<std::tuple<std::vector<int>, bool, bool, int>>{
                 { { 8, 9, 10, 11, 12, 13 }, true, false, 1 },
                 { { 8, 9, 10, 11, 12, 13 }, true, true, 0 },
                 { { 8, 9, 10, 11, 12, 13 }, false, false, 0 },
                 { molecule, true, false, 0 } })
    {
        auto             mtop = makeMtopFromFile("alanine_vacuo");
        MetatomicOptions options;
        fillOptionsFromMdpValues(metatomicBuildDefaultMdpValues(), &options);
        options.params_.oniom                = oniom;
        options.params_.linkAtoms            = linkAtoms;
        const std::vector<IndexGroup> groups = { { "System", ml } };
        options.setInputGroupIndices(IndexGroupsAndNames(groups));
        MDLogger logger;
        options.setLogger(logger);
        WarningHandler wi(true, 0);
        options.setWarningHandler(&wi);
        options.modifyTopology(mtop.get());
        EXPECT_EQ(wi.warningCount(), warnings)
                << "first atom " << ml[0] << ", oniom " << oniom << ", link atoms " << linkAtoms;
    }
}

TEST_F(MetatomicOptionsTest, LinkAtomsRequireOniom)
{
    auto             mtop = makeMtopFromFile("alanine_vacuo");
    MetatomicOptions options;
    fillOptionsFromMdpValues(metatomicBuildDefaultMdpValues(), &options);
    options.params_.linkAtoms            = true;
    const std::vector<IndexGroup> groups = { { "System", { 8, 9, 10 } } };
    options.setInputGroupIndices(IndexGroupsAndNames(groups));
    MDLogger logger;
    options.setLogger(logger);
    EXPECT_THAT([&] { options.modifyTopology(mtop.get()); },
                ::testing::ThrowsMessage<InconsistentInputError>(::testing::HasSubstr(
                        "metatomic-link-atoms requires metatomic-oniom = yes.")));
}

//! Whether \p statement throws an \p Exception whose message is exactly \p message
template<typename Exception, typename Statement>
static ::testing::AssertionResult throwsWithMessage(Statement&& statement, const std::string& message)
{
    try
    {
        statement();
    }
    catch (const Exception& e)
    {
        if (message == e.what())
        {
            return ::testing::AssertionSuccess();
        }
        return ::testing::AssertionFailure() << "message:  " << e.what() << "\nexpected: " << message;
    }
    catch (const std::exception& e)
    {
        return ::testing::AssertionFailure() << "another exception: " << e.what();
    }
    return ::testing::AssertionFailure() << "no exception; expected: " << message;
}

//! Options with the input group ML, and the given index groups and mdp values, sites resolved
static MetatomicOptions siteOptions(const std::vector<IndexGroup>&            groups,
                                    const std::map<std::string, std::string>& values)
{
    KeyValueTreeBuilder mdp;
    mdp.rootObject().addValue("metatomic-active", std::string("true"));
    mdp.rootObject().addValue("metatomic-input-group", std::string("ML"));
    for (const auto& [key, value] : values)
    {
        mdp.rootObject().addValue("metatomic-" + key, value);
    }
    MetatomicOptions options;
    fillOptionsFromMdpValues(mdp.build(), &options);
    options.setInputGroupIndices(IndexGroupsAndNames(groups));
    return options;
}

//! Whether resolving the sites of siteOptions() throws exactly \p message
static ::testing::AssertionResult siteError(const std::vector<IndexGroup>&            groups,
                                            const std::map<std::string, std::string>& values,
                                            const std::string&                        message)
{
    return throwsWithMessage<InconsistentInputError>([&] { siteOptions(groups, values); }, message);
}

const std::vector<IndexGroup> c_twoSites = { { "ML", { 4, 5, 6, 7 } },
                                             { "SITE1", { 6, 7 } },
                                             { "SITE2", { 4, 5 } } };

TEST_F(MetatomicOptionsTest, SiteGroupsGiveEachInputAtomItsSite)
{
    const auto  options = siteOptions(c_twoSites, { { "site-groups", "SITE1 SITE2" } });
    const auto& params  = options.params_;
    EXPECT_EQ(params.numSites(), 2);
    EXPECT_EQ(params.mtaSites_, (std::vector<int>{ 1, 1, 0, 0 })); // in input-group order
    EXPECT_EQ(params.siteChargeValues_, (std::vector<real>{ 0, 0 }));
    EXPECT_EQ(params.siteSpinValues_, (std::vector<int>{ 1, 1 }));
}

TEST_F(MetatomicOptionsTest, OneSiteValueIsUsedForEverySite)
{
    const auto options = siteOptions(c_twoSites,
                                     { { "site-groups", "SITE1 SITE2" },
                                       { "site-charges", "-2" },
                                       { "site-spin-multiplicities", "5" } });
    EXPECT_EQ(options.params_.siteChargeValues_, (std::vector<real>{ -2, -2 }));
    EXPECT_EQ(options.params_.siteSpinValues_, (std::vector<int>{ 5, 5 }));
}

TEST_F(MetatomicOptionsTest, SiteValuesCanBeGivenPerSite)
{
    const auto options = siteOptions(c_twoSites,
                                     { { "site-groups", "SITE1 SITE2" },
                                       { "site-charges", "-1 0.5" },
                                       { "site-spin-multiplicities", "1 3" } });
    EXPECT_EQ(options.params_.siteChargeValues_, (std::vector<real>{ -1, 0.5 }));
    EXPECT_EQ(options.params_.siteSpinValues_, (std::vector<int>{ 1, 3 }));
}

TEST_F(MetatomicOptionsTest, WithoutSiteGroupsTheInputGroupIsOneSite)
{
    const auto options = siteOptions(
            c_twoSites, { { "site-charges", "-2" }, { "site-spin-multiplicities", "5" } });
    EXPECT_EQ(options.params_.numSites(), 1);
    EXPECT_TRUE(options.params_.mtaSites_.empty());
    EXPECT_EQ(options.params_.siteOf(3), 0);
    EXPECT_EQ(options.params_.siteChargeValues_, (std::vector<real>{ -2 }));
    EXPECT_EQ(options.params_.siteSpinValues_, (std::vector<int>{ 5 }));
}

TEST_F(MetatomicOptionsTest, RejectsAnUnknownSiteGroup)
{
    EXPECT_TRUE(
            siteError(c_twoSites,
                      { { "site-groups", "SITE1 SITE3" } },
                      "Group SITE3 referenced in the .mdp file was not found in the list of index "
                      "groups.\nGroup names must match either [moleculetype] names or custom index "
                      "group\nnames, in which case you must supply an index file to the '-n' "
                      "option\nof grompp."));
}

TEST_F(MetatomicOptionsTest, RejectsAnAtomInTwoSites)
{
    EXPECT_TRUE(
            siteError({ { "ML", { 4, 5, 6, 7 } }, { "SITE1", { 4, 5 } }, { "SITE2", { 5, 6, 7 } } },
                      { { "site-groups", "SITE1 SITE2" } },
                      "Atom 6 is in more than one of metatomic-site-groups (SITE1 and SITE2)."));
}

TEST_F(MetatomicOptionsTest, RejectsAnInputAtomInNoSite)
{
    EXPECT_TRUE(siteError({ { "ML", { 4, 5, 6, 7 } }, { "SITE1", { 4, 5 } }, { "SITE2", { 6 } } },
                          { { "site-groups", "SITE1 SITE2" } },
                          "Atom 8 of ML is in none of metatomic-site-groups."));
}

TEST_F(MetatomicOptionsTest, RejectsSiteAtomsOutsideTheInputGroup)
{
    EXPECT_TRUE(
            siteError({ { "ML", { 4, 5, 6, 7 } }, { "SITE1", { 4, 5, 9 } }, { "SITE2", { 6, 7 } } },
                      { { "site-groups", "SITE1 SITE2" } },
                      "metatomic-site-groups contain 5 atoms, but ML has 4; the sites must divide "
                      "the input group."));
}

TEST_F(MetatomicOptionsTest, RejectsSiteValuesThatAreNotNumbers)
{
    const auto with = [](const std::string& key, const std::string& value) {
        return std::map<std::string, std::string>{ { "site-groups", "SITE1 SITE2" }, { key, value } };
    };
    EXPECT_TRUE(siteError(c_twoSites,
                          with("site-charges", "-1 two"),
                          "Cannot read 'two' in metatomic-site-charges."));
    EXPECT_TRUE(siteError(
            c_twoSites, with("site-charges", "1x"), "Cannot read '1x' in metatomic-site-charges."));
    EXPECT_TRUE(siteError(c_twoSites,
                          with("site-spin-multiplicities", "2.5"),
                          "Cannot read '2.5' in metatomic-site-spin-multiplicities."));
}

TEST_F(MetatomicOptionsTest, RejectsAWrongNumberOfSiteValues)
{
    EXPECT_TRUE(siteError(c_twoSites,
                          { { "site-groups", "SITE1 SITE2" }, { "site-charges", "-1 0 1" } },
                          "metatomic-site-charges has 3 values for 2 sites; give one value, or one "
                          "per site."));
    EXPECT_TRUE(siteError(c_twoSites,
                          { { "site-spin-multiplicities", "1 3" } },
                          "metatomic-site-spin-multiplicities has 2 values for 1 sites; give one "
                          "value, or one per site."));
}

TEST_F(MetatomicOptionsTest, RejectsSpinMultiplicitiesBelowOne)
{
    for (const char* spins : { "0", "3 -1" })
    {
        EXPECT_TRUE(siteError(c_twoSites,
                              { { "site-groups", "SITE1 SITE2" }, { "site-spin-multiplicities", spins } },
                              "metatomic-site-spin-multiplicities must be >= 1."));
    }
}

TEST_F(MetatomicOptionsTest, SitesSurviveTheTprFile)
{
    auto                options = siteOptions(c_twoSites,
                                              { { "site-groups", "SITE1 SITE2" },
                                                { "site-charges", "-2 1" },
                                                { "site-spin-multiplicities", "5 2" } });
    const auto          sites   = options.params_.mtaSites_;
    const auto          charges = options.params_.siteChargeValues_;
    const auto          spins   = options.params_.siteSpinValues_;
    KeyValueTreeBuilder builder;
    options.writeParamsToKvt(builder.rootObject());
    options.params_.mtaSites_.clear();
    options.params_.siteChargeValues_.clear();
    options.params_.siteSpinValues_.clear();
    options.readParamsFromKvt(builder.build());
    EXPECT_EQ(options.params_.mtaSites_, sites);
    EXPECT_EQ(options.params_.siteChargeValues_, charges);
    EXPECT_EQ(options.params_.siteSpinValues_, spins);
}

TEST_F(MetatomicOptionsTest, RejectsInconsistentSitesInTheTprFile)
{
    auto options = siteOptions(c_twoSites, { { "site-groups", "SITE1 SITE2" } });
    options.params_.mtaSites_.pop_back(); // sites for three of the four atoms
    KeyValueTreeBuilder builder;
    options.writeParamsToKvt(builder.rootObject());
    const auto tree = builder.build();
    EXPECT_TRUE(throwsWithMessage<InconsistentInputError>(
            [&] { options.readParamsFromKvt(tree); },
            "Inconsistent metatomic site data in the tpr file."));
}

//! Atoms excluded from \p atom in the (only) molecule of alanine_vacuo
static std::vector<int> exclusionsOf(const gmx_mtop_t& mtop, int atom)
{
    const auto excluded = mtop.moltype[mtop.molblock[0].type].excls[atom];
    return { excluded.begin(), excluded.end() };
}

TEST_F(MetatomicOptionsTest, SitesExcludeThePairsWithinThemAndKeepThoseBetweenThem)
{
    // Site A is the ACE methyl and HA, four bonds from its carbon; site B the NME methyl, four
    // bonds from HA and six from the ACE methyl (nrexcl = 3). Pairs further apart than three
    // bonds are only excluded by the embedding.
    const std::vector<int> siteA = { 0, 1, 2, 3, 9 }, siteB = { 18, 19, 20, 21 };
    std::vector<int>       ml = siteA;
    ml.insert(ml.end(), siteB.begin(), siteB.end());
    std::sort(ml.begin(), ml.end());
    const auto excludes = [](const gmx_mtop_t& mtop, int a, int b)
    {
        const auto list = exclusionsOf(mtop, a);
        return std::find(list.begin(), list.end(), b) != list.end();
    };
    const auto withoutEmbedding = makeMtopFromFile("alanine_vacuo");
    ASSERT_FALSE(excludes(*withoutEmbedding, 0, 9));
    ASSERT_FALSE(excludes(*withoutEmbedding, 9, 18));

    for (const bool sites : { false, true })
    {
        SCOPED_TRACE(sites ? "two sites" : "one site");
        auto                               mtop   = makeMtopFromFile("alanine_vacuo");
        std::map<std::string, std::string> values = { { "oniom", "true" } };
        if (sites)
        {
            values.emplace("site-groups", "A B");
        }
        auto options = siteOptions({ { "ML", ml }, { "A", siteA }, { "B", siteB } }, values);
        StringOutputStream log;
        LoggerBuilder      loggerBuilder;
        loggerBuilder.addTargetStream(MDLogger::LogLevel::Info, &log);
        const LoggerOwner logger = loggerBuilder.build();
        options.setLogger(logger.logger());
        WarningHandler wi(true, 0);
        options.setWarningHandler(&wi);
        ASSERT_NO_THROW(options.modifyTopology(mtop.get()));
        const std::string sitesLine =
                "Excluded the 16 atom pairs within 2 sites; pairs between sites keep their "
                "classical interactions\n";
        EXPECT_EQ(log.toString().find(sitesLine) != std::string::npos, sites) << log.toString();

        // one site: all embedded pairs, through the single intermolecular exclusion group;
        // several sites: the pairs within each site, through the exclusions of the molecule
        std::vector<int> group(mtop->intermolecularExclusionGroup.begin(),
                               mtop->intermolecularExclusionGroup.end());
        std::sort(group.begin(), group.end());
        EXPECT_EQ(group, sites ? std::vector<int>{} : ml);
        for (const auto& site : { siteA, siteB })
        {
            for (const int a : site)
            {
                for (const int b : site)
                {
                    EXPECT_EQ(excludes(*mtop, a, b), sites || excludes(*withoutEmbedding, a, b))
                            << a + 1 << " " << b + 1;
                }
            }
        }
        for (const int a : siteA)
        {
            for (const int b : siteB)
            {
                EXPECT_FALSE(excludes(*mtop, a, b)) << a + 1 << " " << b + 1;
            }
        }
    }
}

TEST_F(MetatomicOptionsTest, RejectsASiteSpanningTwoMolecules)
{
    auto mtop    = makeMtopFromFile("spc2");
    auto options = siteOptions(
            { { "ML", { 0, 1, 2, 3, 4, 5 } }, { "SITE1", { 0, 1, 2, 3 } }, { "SITE2", { 4, 5 } } },
            { { "oniom", "true" }, { "site-groups", "SITE1 SITE2" } });
    MDLogger logger;
    options.setLogger(logger);
    WarningHandler wi(true, 0);
    options.setWarningHandler(&wi);
    EXPECT_TRUE(throwsWithMessage<InconsistentInputError>(
            [&] { options.modifyTopology(mtop.get()); },
            "Site 1 spans more than one molecule; with several metatomic-site-groups each site "
            "must lie within one molecule."));
}

TEST_F(MetatomicOptionsTest, RejectsABondBetweenSites)
{
    // CA and HA in one site, the methyl group CB bonded to CA in another
    auto mtop    = makeMtopFromFile("alanine_vacuo");
    auto options = siteOptions(
            { { "ML", { 8, 9, 10, 11, 12, 13 } }, { "A", { 8, 9 } }, { "B", { 10, 11, 12, 13 } } },
            { { "oniom", "true" }, { "site-groups", "A B" } });
    MDLogger logger;
    options.setLogger(logger);
    WarningHandler wi(true, 0);
    options.setWarningHandler(&wi);
    EXPECT_TRUE(throwsWithMessage<InconsistentInputError>(
            [&] { options.modifyTopology(mtop.get()); },
            "Atoms 9 and 11 are bonded but in sites 1 and 2. The model evaluates each site alone, "
            "so the bond would have no energy; put both atoms in one of metatomic-site-groups."));
}

TEST_F(MetatomicOptionsTest, LinkAtomsJoinTheSiteOfTheirEmbeddedAtom)
{
    // the methyl groups of ACE and NME, bonded to the C of ACE and the N of NME
    auto mtop    = makeMtopFromFile("alanine_vacuo");
    auto options = siteOptions(
            { { "ML", { 0, 1, 2, 3, 18, 19, 20, 21 } }, { "A", { 0, 1, 2, 3 } }, { "B", { 18, 19, 20, 21 } } },
            { { "oniom", "true" }, { "link-atoms", "true" }, { "site-groups", "A B" } });
    MDLogger logger;
    options.setLogger(logger);
    WarningHandler wi(true, 0);
    options.setWarningHandler(&wi);
    options.modifyTopology(mtop.get());
    EXPECT_EQ(options.params_.mtaIndices_, (std::vector<Index>{ 0, 1, 2, 3, 18, 19, 20, 21, 4, 16 }));
    EXPECT_EQ(options.params_.mtaSites_, (std::vector<int>{ 0, 0, 0, 0, 1, 1, 1, 1, 0, 1 }));

    // the input group of each site is exempt from the exclusion distance check
    ExclusionDistanceExemptions exemptions;
    options.addExclusionDistanceExemptions(&exemptions);
    EXPECT_EQ(exemptions.groups, (std::vector<std::vector<int>>{ { 0, 1, 2, 3 }, { 18, 19, 20, 21 } }));
}

TEST_F(MetatomicOptionsTest, RejectsAnMMAtomCappingTwoSites)
{
    // the C of ACE is bonded to its methyl group and to its O
    auto mtop = makeMtopFromFile("alanine_vacuo");
    auto options =
            siteOptions({ { "ML", { 0, 1, 2, 3, 5 } }, { "A", { 0, 1, 2, 3 } }, { "B", { 5 } } },
                        { { "oniom", "true" }, { "link-atoms", "true" }, { "site-groups", "A B" } });
    MDLogger logger;
    options.setLogger(logger);
    WarningHandler wi(true, 0);
    options.setWarningHandler(&wi);
    EXPECT_TRUE(throwsWithMessage<InconsistentInputError>(
            [&] { options.modifyTopology(mtop.get()); },
            "Atom 5 is bonded to atoms of two sites; it cannot carry link atoms of both."));
}

TEST_F(MetatomicOptionsTest, GromppAcceptsSitesWiderThanTheCutoffWithOniom)
{
    // Site A, the ACE methyl and HA, is 0.48 nm across, beyond rcoulomb = 0.4 nm; the
    // bonded exclusions of alanine_vacuo reach 0.38 nm. The embedded Coulomb correction
    // completes the exclusions within a site at any distance, so grompp accepts them.
    const std::string mdp =
            "integrator = steep\n"
            "nsteps = 0\n"
            "cutoff-scheme = Verlet\n"
            "coulombtype = PME\n"
            "rcoulomb = 0.4\n"
            "rvdw = 0.4\n"
            "metatomic-active = yes\n"
            "metatomic-input-group = ML\n"
            "metatomic-model = model\n"
            "metatomic-oniom = yes\n"
            "metatomic-link-atoms = yes\n"
            "metatomic-site-groups = A B\n";
    const std::string ndx =
            "[ ML ]\n1 2 3 4 10 19 20 21 22\n[ A ]\n1 2 3 4 10\n[ B ]\n19 20 21 22\n";
    const auto mtop = makeMtopFromFile("alanine_vacuo", mdp, ndx);
    EXPECT_EQ(mtop->natoms, 22);
}

TEST_F(MetatomicOptionsTest, InternalsToKvtAndBack)
{
    // Set metatomic-active = true and other params
    MetatomicOptions metatomicOptions;
    fillOptionsFromMdpValues(metatomicBuildInputMdpValues(), &metatomicOptions);

    // Set indices
    const IndexGroupsAndNames indexGroupAndNames = indexGroupsAndNamesGeneric();
    metatomicOptions.setInputGroupIndices(indexGroupAndNames);

    // Set dummy logger and warning handler
    MDLogger logger;
    metatomicOptions.setLogger(logger);
    WarningHandler warninp(true, 0);
    metatomicOptions.setWarningHandler(&warninp);

    metatomicOptions.params_.modelOptions = R"(sigma=3 label="argon model" empty='')";
    const auto expectedOptions = parseMetatomicModelOptions(metatomicOptions.params_.modelOptions);

    // Copy internal parameters
    const MetatomicParameters& params           = metatomicOptions.parameters();
    auto                       mtaIndicesBefore = params.mtaIndices_;

    KeyValueTreeBuilder builder;
    EXPECT_NO_THROW(metatomicOptions.writeParamsToKvt(builder.rootObject()));
    const auto inputTree = builder.build();

    // The parsed values, including empty strings, are carried by the tpr KVT.
    const auto& stored = inputTree["metatomic-model-loading-options"].asObject();
    EXPECT_EQ(stored["label"].cast<std::string>(), "argon model");
    EXPECT_EQ(stored["empty"].cast<std::string>(), "");
    metatomicOptions.params_.modelOptions.clear();
    EXPECT_NO_THROW(metatomicOptions.readParamsFromKvt(inputTree));
    EXPECT_EQ(parseMetatomicModelOptions(metatomicOptions.params_.modelOptions), expectedOptions);

    // Check Internal parameters taken back from KVT
    const MetatomicParameters& params2 = metatomicOptions.parameters();
    EXPECT_EQ(mtaIndicesBefore, params2.mtaIndices_);
}

} // namespace test

} // namespace gmx
