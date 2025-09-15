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
#include "gmxpre.h"

#include "metatomic_options.h"

#include "gromacs/options/basicoptions.h"
#include "gromacs/options/optionsection.h"
#include "gromacs/utility/keyvaluetreebuilder.h"
#include "gromacs/utility/keyvaluetreetransform.h"
#include "gromacs/utility/strconvert.h"

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
 * Enforces uniform mdp options that are always prepended with the correct
 * string for the NNPot mdp options.
 *
 * \tparam ToType type to be transformed to
 * \tparam TransformWithFunctionType type of transformation function to be used
 *
 * \param[in] rules KVT transformation rules
 * \param[in] transformationFunction the function to transform the flat kvt tree
 * \param[in] optionTag string tag that describes the mdp option, appended to the
 *                      default string for the metatomic options
 */
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

void MetatomicOptions::initMdpTransform(IKeyValueTreeTransformRules* rules)
{
    const auto& stringIdentityTransform = [](std::string s) { return s; };
    MetatomicMdpTransformFromString<bool>(rules, &fromStdString<bool>, ACTIVE_TAG);
    MetatomicMdpTransformFromString<std::string>(rules, stringIdentityTransform, INPUT_GROUP_TAG);

    MetatomicMdpTransformFromString<std::string>(rules, stringIdentityTransform, MODEL_PATH_TAG);
    MetatomicMdpTransformFromString<std::string>(rules, stringIdentityTransform, EXTENSIONS_DIRECTORY_TAG);
    MetatomicMdpTransformFromString<bool>(rules, &fromStdString<bool>, CHECK_CONSISTENCY_TAG);
    MetatomicMdpTransformFromString<std::string>(rules, stringIdentityTransform, DEVICE_TAG);
}

void MetatomicOptions::initMdpOptions(IOptionsContainerWithSections* options)
{
    auto section = options->addSection(OptionSection(METATOMIC_MODULE_NAME.c_str()));
    section.addOption(BooleanOption(ACTIVE_TAG.c_str()).store(&params_.active));
    section.addOption(StringOption(INPUT_GROUP_TAG.c_str()).store(&params_.inputGroup));

    section.addOption(StringOption(MODEL_PATH_TAG.c_str()).store(&params_.modelPath));
    section.addOption(StringOption(EXTENSIONS_DIRECTORY_TAG.c_str()).store(&params_.extensionsDirectory));
    section.addOption(StringOption(DEVICE_TAG.c_str()).store(&params_.device));
    section.addOption(BooleanOption(CHECK_CONSISTENCY_TAG.c_str()).store(&params_.checkConsistency));
}

void MetatomicOptions::buildMdpOutput(KeyValueTreeObjectBuilder* builder) const
{
    // new empty line before writing mdp values
    builder->addValue<std::string>("comment-" + METATOMIC_MODULE_NAME + "empty-line", "");

    builder->addValue<std::string>("comment-" + METATOMIC_MODULE_NAME + "-module",
                                   "; Machine learning potential using metatomic");
    builder->addValue<bool>(METATOMIC_MODULE_NAME + "-" + ACTIVE_TAG, params_.active);

    if (params_.active)
    {
        builder->addValue<std::string>(METATOMIC_MODULE_NAME + "-" + INPUT_GROUP_TAG, params_.inputGroup);

        builder->addValue<std::string>(METATOMIC_MODULE_NAME + "-" + MODEL_PATH_TAG, params_.modelPath);
        builder->addValue<std::string>(METATOMIC_MODULE_NAME + "-" + EXTENSIONS_DIRECTORY_TAG,
                                       params_.extensionsDirectory);
        builder->addValue<std::string>(METATOMIC_MODULE_NAME + "-" + DEVICE_TAG, params_.device);
        builder->addValue<bool>(METATOMIC_MODULE_NAME + "-" + CHECK_CONSISTENCY_TAG,
                                params_.checkConsistency);
    }
}

const MetatomicParameters& MetatomicOptions::parameters()
{
    return params_;
}


} // namespace gmx
