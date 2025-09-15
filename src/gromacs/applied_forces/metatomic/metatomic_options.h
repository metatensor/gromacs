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
 * Declares the options for Metatomic MDModule class,
 * set during pre-processing in the .mdp-file.
 *
 * \author Metatensor developers <https://github.com/metatensor>
 * \ingroup module_applied_forces
 */
#pragma once

#include <string>

#include "gromacs/mdtypes/imdpoptionprovider.h"

namespace gmx
{

class MDLogger;
class IOptionsContainerWithSections;
class IKeyValueTreeTransformRules;
class KeyValueTreeObjectBuilder;

//! TODO
struct MetatomicParameters
{
    //! Is the metatomic force provider enabled?
    bool active = false;

    //! path to the exported metatomic model file
    std::string modelPath;
    //! path to a directory where extensions will be present at MD time
    std::string extensionsDirectory;
    //! should metatomic run additional checks on the models inputs & outputs?
    bool checkConsistency = true;
    //! Torch device to use to run the model. If left empty, this is defined
    //! based on the model declared preferences
    std::string device;

    // TODO: how should we translate atomic types?

    //! stores atom group name for which metatomic should compute the energy
    //! (default whole System)
    std::string inputGroup = "System";
};

class MetatomicOptions final : public IMdpOptionProvider
{
public:
    void initMdpTransform(IKeyValueTreeTransformRules* rules) override;
    void initMdpOptions(IOptionsContainerWithSections* options) override;
    void buildMdpOutput(KeyValueTreeObjectBuilder* builder) const override;

    // //! modify topology of the system during preprocessing
    // void modifyTopology(gmx_mtop_t*) const;

    // //! set topology of the system during simulation setup
    // void setTopology(const gmx_mtop_t&);

    // //! set communication record during simulation setup
    // void setCommRec(const t_commrec&);

    // //! Store the paramers that are not mdp options in the tpr file
    // // This is needed to retain data from preprocessing to simulation setup
    // void writeParamsToKvt(KeyValueTreeObjectBuilder);

    // //! Set the internal parameters that are stored in the tpr file
    // void readParamsFromKvt(const KeyValueTreeObject&);

    //! TODO
    const MetatomicParameters& parameters();

private:
    MetatomicParameters params_;
};

} // namespace gmx
