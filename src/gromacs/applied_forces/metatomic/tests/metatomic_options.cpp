
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

#include <filesystem>
#include <memory>
#include <set>

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

//! Topology of \p simulationName in the simulation database, as grompp writes it with an empty mdp
static std::unique_ptr<gmx_mtop_t> makeMtopFromFile(const std::string& simulationName)
{
    const std::filesystem::path simData = TestFileManager::getTestSimulationDatabaseDirectory();
    TestFileManager             fileManager;
    const std::string mdp = fileManager.getTemporaryFilePath(simulationName + ".mdp").string();
    TextWriter::writeFileFromString(mdp, "");
    const std::string tpr = fileManager.getTemporaryFilePath(simulationName + ".tpr").string();
    CommandLine       grompp;
    grompp.append("grompp");
    grompp.addOption("-f", mdp);
    grompp.addOption("-p", (simData / simulationName).replace_extension(".top").string());
    grompp.addOption("-c", (simData / simulationName).replace_extension(".gro").string());
    grompp.addOption("-o", tpr);
    EXPECT_EQ(0, gmx_grompp(grompp.argc(), grompp.argv()));

    bool    fullTopology;
    PbcType pbcType;
    matrix  box;
    auto    mtop = std::make_unique<gmx_mtop_t>();
    readConfAndTopology(tpr.c_str(), &fullTopology, mtop.get(), &pbcType, nullptr, nullptr, box);
    return mtop;
}

//! Number of bonded interactions of \p numAtoms atoms with at least \p minEmbedded atoms in \p embedded
static int countInteractions(const gmx_mtop_t& mtop, int numAtoms, const std::set<int>& embedded, int minEmbedded)
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
                if (!(interaction_function[ftype].categories & sc_ifcBondedForce) || NRAL(ftype) != numAtoms)
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
