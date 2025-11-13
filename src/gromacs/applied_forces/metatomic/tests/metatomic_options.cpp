
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
 * Tests for functionality of the NNPotOptions
 *
 * \author Lukas Müllender <lukas.muellender@gmail.com>
 * \ingroup module_applied_forces
 */

#include "gmxpre.h"

#include "gromacs/applied_forces/metatomic/metatomic_options.h"

#include <gtest/gtest.h>

#include "gromacs/applied_forces/nnpot/nnpot.h"
#include "gromacs/domdec/localatomset.h"
#include "gromacs/fileio/warninp.h"
#include "gromacs/mdrunutility/mdmodulesnotifiers.h"
#include "gromacs/mdtypes/imdpoptionprovider_test_helper.h"
#include "gromacs/selection/indexutil.h"
#include "gromacs/topology/index.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/keyvaluetree.h"
#include "gromacs/utility/keyvaluetreebuilder.h"
#include "gromacs/utility/keyvaluetreemdpwriter.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/stringstream.h"
#include "gromacs/utility/textwriter.h"

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
        mdpValueBuilder.rootObject().addValue(
                METATOMIC_MODULE_NAME + "-model",
                gmx::test::TestFileManager::getInputFilePath("model.pt").string());
        mdpValueBuilder.rootObject().addValue(METATOMIC_MODULE_NAME + "-input_group",
                                              std::string("System"));
        mdpValueBuilder.rootObject().addValue(METATOMIC_MODULE_NAME + "-extensions", std::string("./ext"));
        mdpValueBuilder.rootObject().addValue(METATOMIC_MODULE_NAME + "-device", std::string("cpu"));
        mdpValueBuilder.rootObject().addValue(METATOMIC_MODULE_NAME + "-check_consistency",
                                              std::string("true"));
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
    checker.checkString(defaultParams.extensionsDirectory, "extensionsDirectory");
    checker.checkString(defaultParams.device, "device");
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

} // namespace test

} // namespace gmx
