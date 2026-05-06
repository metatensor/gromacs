
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
 * \author Metatensor developers <https://github.com/metatensor>
 * \ingroup module_applied_forces
 */

#include "gmxpre.h"

#include "gromacs/applied_forces/metatomic/metatomic_options.h"

#include <filesystem>
#include <map>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/applied_forces/nnpot/nnpot.h"
#include "gromacs/domdec/localatomset.h"
#include "gromacs/fileio/confio.h"
#include "gromacs/fileio/warninp.h"
#include "gromacs/gmxpreprocess/grompp.h"
#include "gromacs/mdrunutility/mdmodulesnotifiers.h"
#include "gromacs/mdtypes/imdpoptionprovider_test_helper.h"
#include "gromacs/selection/indexutil.h"
#include "gromacs/topology/index.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/topology/mtop_util.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/enumerationhelpers.h"
#include "gromacs/utility/keyvaluetree.h"
#include "gromacs/utility/keyvaluetreebuilder.h"
#include "gromacs/utility/keyvaluetreemdpwriter.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/stringstream.h"
#include "gromacs/utility/textwriter.h"

#include "testutils/refdata.h"
#include "testutils/cmdlinetest.h"
#include "testutils/testasserts.h"
#include "testutils/testfilemanager.h"
#include "testutils/testmatchers.h"

enum class PbcType : int;

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

    static KeyValueTreeObject metatomicBuildMdpValues(
            const std::map<std::string, std::string>& additionalValues)
    {
        KeyValueTreeBuilder mdpValueBuilder;
        mdpValueBuilder.rootObject().addValue(METATOMIC_MODULE_NAME + "-active", std::string("true"));
        for (const auto& [key, value] : additionalValues)
        {
            mdpValueBuilder.rootObject().addValue(METATOMIC_MODULE_NAME + "-" + key, value);
        }
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

        mdpValueBuilder.rootObject().addValue(METATOMIC_MODULE_NAME + "-extensions", std::string("./ext"));
        mdpValueBuilder.rootObject().addValue(METATOMIC_MODULE_NAME + "-check-consistency",
                                              std::string("false"));
        mdpValueBuilder.rootObject().addValue(METATOMIC_MODULE_NAME + "-device", std::string("cpu"));
        mdpValueBuilder.rootObject().addValue(METATOMIC_MODULE_NAME + "-variant",
                                              std::string(""));
        mdpValueBuilder.rootObject().addValue(METATOMIC_MODULE_NAME + "-uncertainty-threshold",
                                              std::string("auto"));
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

    static IndexGroupsAndNames indexGroupsAndNames(const std::vector<int>& metatomicAtomIndices)
    {
        std::vector<IndexGroup> indexGroups;
        indexGroups.push_back({ "System", metatomicAtomIndices });
        return IndexGroupsAndNames(indexGroups);
    }

    static std::unique_ptr<gmx_mtop_t> makeMtopFromFile(const std::string& simulationName,
                                                        const std::string& mdpContent)
    {
        const std::filesystem::path simData =
                gmx::test::TestFileManager::getTestSimulationDatabaseDirectory();
        TestFileManager fileManager;

        const std::string mdpInputFileName =
                fileManager.getTemporaryFilePath(simulationName + ".mdp").string();
        gmx::TextWriter::writeFileFromString(mdpInputFileName, mdpContent);

        const std::string tprName = fileManager.getTemporaryFilePath(simulationName + ".tpr").string();
        {
            gmx::test::CommandLine caller;
            caller.append("grompp");
            caller.addOption("-f", mdpInputFileName);
            caller.addOption("-p", (simData / simulationName).replace_extension(".top").string());
            caller.addOption("-c", (simData / simulationName).replace_extension(".gro").string());
            caller.addOption("-o", tprName);
            EXPECT_EQ(0, gmx_grompp(caller.argc(), caller.argv()));
        }

        bool                        fullTopology;
        PbcType                     pbcType;
        matrix                      box;
        std::unique_ptr<gmx_mtop_t> mtop(std::make_unique<gmx_mtop_t>());
        readConfAndTopology(tprName.c_str(), &fullTopology, mtop.get(), &pbcType, nullptr, nullptr, box);
        return mtop;
    }

    static std::map<InteractionFunction, int> listedInteractionSizes(const gmx_mtop_t& mtop)
    {
        std::map<InteractionFunction, int> sizes;
        for (const auto ftype : gmx::EnumerationWrapper<InteractionFunction>{})
        {
            int size = 0;
            for (const auto& molblock : mtop.molblock)
            {
                const auto& moltype = mtop.moltype[molblock.type];
                size += molblock.nmol * moltype.ilist[ftype].size();
            }
            sizes.emplace(ftype, size);
        }
        return sizes;
    }

    MetatomicOptions buildOptions(const std::vector<int>&                   metatomicAtomIndices,
                                  WarningHandler*                           wi,
                                  const std::map<std::string, std::string>& additionalValues)
    {
        MetatomicOptions options;
        fillOptionsFromMdpValues(metatomicBuildMdpValues(additionalValues), &options);
        options.setLogger(logHelper_.logger());
        options.setWarningHandler(wi);
        options.setInputGroupIndices(indexGroupsAndNames(metatomicAtomIndices));
        return options;
    }

    void expectLogMessage(const char* msg)
    {
        logHelper_.expectEntryMatchingRegex(MDLogger::LogLevel::Info, msg);
    }

protected:
    LoggerTestHelper logHelper_;
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
    checker.checkString(defaultParams.extensionsDirectory, "extensionsDirectory");
    checker.checkString(defaultParams.device, "device");
    checker.checkBoolean(defaultParams.checkConsistency, "checkConsistency");
    checker.checkBoolean(defaultParams.oniom, "oniom");
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

    // Copy internal parameters
    const MetatomicParameters& params           = metatomicOptions.parameters();
    auto                       mtaIndicesBefore = params.mtaIndices_;

    KeyValueTreeBuilder builder;
    // MetatomicOptions::writeParamsToKvt doesn't have external dependencies like
    // GMX_TORCH, so we can call it directly.
    EXPECT_NO_THROW(metatomicOptions.writeParamsToKvt(builder.rootObject()));
    const auto inputTree = builder.build();

    EXPECT_NO_THROW(metatomicOptions.readParamsFromKvt(inputTree));

    // Check Internal parameters taken back from KVT
    const MetatomicParameters& params2 = metatomicOptions.parameters();
    EXPECT_EQ(mtaIndicesBefore, params2.mtaIndices_);
}

TEST_F(MetatomicOptionsTest, ChargesToKvtAndBack)
{
    MetatomicOptions metatomicOptions;
    fillOptionsFromMdpValues(metatomicBuildInputMdpValues(), &metatomicOptions);

    const std::vector<real> charges = { 0.1_real, -0.2_real, 0.3_real };
    metatomicOptions.params_.mmCharges_ = charges;

    KeyValueTreeBuilder builder;
    EXPECT_NO_THROW(metatomicOptions.writeParamsToKvt(builder.rootObject()));

    metatomicOptions.params_.mmCharges_.clear();
    EXPECT_NO_THROW(metatomicOptions.readParamsFromKvt(builder.build()));

    EXPECT_EQ(charges, metatomicOptions.parameters().mmCharges_);
}

TEST_F(MetatomicOptionsTest, AdditiveMechanicalEmbeddingLeavesTopologyUnmodified)
{
    const std::vector<int> metatomicAtomIndices = { 8, 9, 10, 11, 12, 13 };
    auto                   mtop                 = makeMtopFromFile("alanine_vacuo", "");
    const auto             listedBefore         = listedInteractionSizes(*mtop);
    const auto             exclusionsBefore     = mtop->intermolecularExclusionGroup.atomNumbers().size();

    WarningHandler   wi(true, 0);
    MetatomicOptions options = buildOptions(metatomicAtomIndices, &wi, {});

    expectLogMessage("Metatomic potential interface is active, topology was not modified.");
    EXPECT_NO_THROW(options.modifyTopology(mtop.get()));

    EXPECT_EQ(listedBefore, listedInteractionSizes(*mtop));
    EXPECT_EQ(exclusionsBefore, mtop->intermolecularExclusionGroup.atomNumbers().size());
    EXPECT_TRUE(options.parameters().linkFrontier_.empty());
    EXPECT_EQ(options.parameters().mmCharges_.size(), gmx_mtop_global_atoms(*mtop).nr);
}

TEST_F(MetatomicOptionsTest, OniomEmbeddingAppliesSubtractiveTopologyPreprocessing)
{
    const std::vector<int> metatomicAtomIndices = { 8, 9, 10, 11, 12, 13 };
    auto                   mtop                 = makeMtopFromFile("alanine_vacuo", "");

    WarningHandler   wi(true, 0);
    MetatomicOptions options =
            buildOptions(metatomicAtomIndices, &wi, { { "oniom", "true" } });

    expectLogMessage("Metatomic potential interface is active, topology was modified!");
    expectLogMessage("Number of embedded Metatomic atoms: 6\nNumber of regular atoms: 16\n");
    expectLogMessage("Number of exclusions made: 6\n");
    expectLogMessage("Number of bonds removed: 8\n");
    expectLogMessage("Number of InteractionFunction::ConnectBonds \\(type 5 bonds\\) added: 5\n");
    expectLogMessage("Number of angles removed: 7\n");
    expectLogMessage("Number of dihedrals removed: 3\n");
    EXPECT_NO_THROW(options.modifyTopology(mtop.get()));
}

TEST_F(MetatomicOptionsTest, LinkAtomsRequireOniomEmbedding)
{
    const std::vector<int> metatomicAtomIndices = { 8, 9, 10, 11, 12, 13 };
    auto                   mtop                 = makeMtopFromFile("alanine_vacuo", "");

    WarningHandler   wi(true, 0);
    MetatomicOptions options =
            buildOptions(metatomicAtomIndices, &wi, { { "link-atoms", "true" } });

    EXPECT_THROW_GMX(options.modifyTopology(mtop.get()), InconsistentInputError);
}

} // namespace test

} // namespace gmx
