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

#include <memory>
#include <string>
#include <vector>

#include "gromacs/mdtypes/imdpoptionprovider.h"
#include "gromacs/topology/atoms.h"
#include "gromacs/utility/vectypes.h"

// some forward declarations
struct gmx_mtop_t;
class WarningHandler;
enum class PbcType;


namespace gmx
{
class MDLogger;
class IOptionsContainerWithSections;
class IKeyValueTreeTransformRules;
class KeyValueTreeObjectBuilder;
class KeyValueTreeObject;
class IndexGroupsAndNames;
class LocalAtomSet;
class MpiComm;


//! TODO
struct MetatomicParameters
{
    //! Is the metatomic force provider enabled?
    bool active = false;

    //! path to the exported metatomic model file
    std::string modelPath_;
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

    std::vector<Index>            metatomicIndices;
    std::unique_ptr<LocalAtomSet> mtaAtoms_;
    std::unique_ptr<LocalAtomSet> gmxMMAtoms_;
    t_atoms                       atoms_;
    int                           numAtoms_;
    std::unique_ptr<PbcType>      pbcType_;
};

class MetatomicOptions final : public IMdpOptionProvider
{
public:
    MetatomicParameters params_;
    // ^---------- lol
    void initMdpTransform(IKeyValueTreeTransformRules* rules) override;
    void initMdpOptions(IOptionsContainerWithSections* options) override;
    void buildMdpOutput(KeyValueTreeObjectBuilder* builder) const override;

    bool isActive() const;
    void setInputGroupIndices(const IndexGroupsAndNames&);
    void modifyTopology(gmx_mtop_t*);
    void writeParamsToKvt(KeyValueTreeObjectBuilder);
    void readParamsFromKvt(const KeyValueTreeObject&);
    void setLogger(const MDLogger&);
    void setWarningHandler(WarningHandler*);
    void setTopology(const gmx_mtop_t&);
    void setLocalAtomSet(const LocalAtomSet&);
    void setPbcType(const PbcType&);
    void setComm(const MpiComm&);

    const MetatomicParameters& parameters();
    const MDLogger&            logger() const;
    const MpiComm&             mpiComm() const;


private:
    const MDLogger* logger_  = nullptr;
    const MpiComm*  mpiComm_ = nullptr;
    WarningHandler* wi_      = nullptr;
};

} // namespace gmx
