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
#include "gromacs/topology/embedded_system_preprocessing.h"

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


//!\brief \internal Data structure to store Metatomic input parameters
struct MetatomicParameters
{
    //! Is the metatomic force provider enabled?
    bool active = false;

    //! path to the exported metatomic model file
    std::string modelPath_;
    //! path to a directory where extensions will be present at MD time
    std::string extensionsDirectory;
    //! should metatomic run additional checks on the models inputs & outputs?
    bool checkConsistency = false;
    //! Torch device to use to run the model. If left empty, this is defined
    //! based on the model declared preferences
    std::string device;
    //! specifies which variant of the model outputs should be uses for making
    //! predictions
    std::string variant;

    //! Uncertainty threshold: "auto" (100 meV/atom), "off", or a number in kJ/mol
    std::string uncertaintyThreshold = "auto";
    //! Variant override for energy_uncertainty output
    std::string variantEnergyUq;

    //! Enable non-conservative mode (forces/stress predicted directly, no backward pass)
    bool nonConservative = false;
    //! Variant overrides for non-conservative outputs
    std::string variantNcForces;
    std::string variantNcStress;

    // TODO: how should we translate atomic types?

    //! stores atom group name for which metatomic should compute the energy
    //! (default whole System)
    std::string inputGroup = "System";

    //! Enable subtractive ONIOM topology preprocessing
    bool oniom = false;

    //! Enable ONIOM link atoms at cut bonds between ML and MM regions
    bool linkAtoms = false;

    //! Index groups, one per ML site, each evaluated by the model as its own system
    std::string siteGroups;
    //! Total charge of each site (one value for all sites), for models that request it
    std::string siteCharges;
    //! Spin multiplicity of each site (one value for all sites), for models that request it
    std::string siteSpinMultiplicities;

    std::vector<Index>            mtaIndices_;
    std::vector<Index>            mmIndices_;
    std::unique_ptr<LocalAtomSet> mtaAtoms_;
    std::unique_ptr<LocalAtomSet> gmxMMAtoms_;
    t_atoms                       atoms_;
    int                           numAtoms_;
    std::unique_ptr<PbcType>      pbcType_;

    //! Link frontier atoms (bonds crossing ML/MM boundary)
    std::vector<LinkFrontierAtom> linkFrontier_;
    //! Topology charges indexed by global atom
    std::vector<real>             mmCharges_;
    //! Energy group of every atom, from the simulation topology
    std::vector<int> energyGroups_;
    //! Number of energy groups
    int numEnergyGroups_ = 1;

    //! Site of each atom in mtaIndices_ (all 0 without site groups)
    std::vector<int> mtaSites_;
    //! Total charge of each site
    std::vector<real> siteChargeValues_;
    //! Spin multiplicity of each site
    std::vector<int> siteSpinValues_;
    //! Number of sites
    int numSites() const { return static_cast<int>(siteChargeValues_.size()); }
};

struct ExclusionDistanceExemptions;

class MetatomicOptions final : public IMdpOptionProvider
{
public:
    MetatomicParameters params_;
    void initMdpTransform(IKeyValueTreeTransformRules* rules) override;
    void initMdpOptions(IOptionsContainerWithSections* options) override;
    void buildMdpOutput(KeyValueTreeObjectBuilder* builder) const override;

    bool isActive() const;
    void setInputGroupIndices(const IndexGroupsAndNames&);
    void modifyTopology(gmx_mtop_t*);
    //! With ONIOM, lists each site: the embedded Coulomb correction handles its exclusions at any distance
    void addExclusionDistanceExemptions(ExclusionDistanceExemptions* exemptions) const;
    void writeParamsToKvt(KeyValueTreeObjectBuilder);
    void readParamsFromKvt(const KeyValueTreeObject&);
    void setLogger(const MDLogger&);
    void setWarningHandler(WarningHandler*);
    void setTopology(const gmx_mtop_t&);
    void setPbcType(const PbcType&);
    void setComm(const MpiComm&);
    //! set local atom set for Metatomic input during simulation setup
    void setLocalInputAtomSet(const LocalAtomSet&);

    //! set local MM atom set during simulation setup
    void setLocalgmxMMAtomSet(const LocalAtomSet&);

    const MetatomicParameters& parameters();
    const MDLogger&            logger() const;
    const MpiComm&             mpiComm() const;


private:
    const MDLogger* logger_  = nullptr;
    const MpiComm*  mpiComm_ = nullptr;
    WarningHandler* wi_      = nullptr;
};

} // namespace gmx
