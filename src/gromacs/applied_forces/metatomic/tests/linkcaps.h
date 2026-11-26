/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2026- The GROMACS Authors
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
 * Atoms with link caps for the tests of the force provider, and the energy, forces and
 * virial that the Lennard-Jones model of metatomic-core gives them.
 *
 * The caps are argon, like the other atoms, as the model knows one atomic type.
 *
 * \author Eric Boittier <eric.boittier@epfl.ch>
 * \ingroup module_applied_forces
 */
#ifndef GMX_METATOMIC_LINKCAPS_H
#define GMX_METATOMIC_LINKCAPS_H

#include <cmath>

#include <array>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "gromacs/topology/embedded_system_preprocessing.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/vectypes.h"

namespace gmx::test
{

using DMatrix = std::array<DVec, DIM>;

/*! \brief Embedded atoms and MM atoms bonded to them, with the Lennard-Jones model.
 *
 * By default, three embedded atoms and two MM atoms in a triclinic box. MM atom 3 is bonded
 * to embedded atoms 0 and 2, so it carries two caps. MM atom 4 is bonded to embedded atom 1
 * across the periodic boundary along a. All atoms are within 0.55 nm of each other, through
 * the minimum image, and the cutoff is 0.6 nm.
 */
struct LinkCapSystem
{
    //! Lennard-Jones parameters (nm, kJ/mol), and the distance of a cap from its embedded atom
    double sigma       = 0.08;
    double epsilon     = 2.0;
    double cutoff      = 0.6;
    double capDistance = 0.12;

    DMatrix           box       = { DVec(3, 0, 0), DVec(0.5, 2.9, 0), DVec(-0.4, 0.6, 2.8) };
    std::vector<DVec> positions = { { 0.10, 1.00, 0.90 },
                                    { 0.02, 0.85, 1.05 },
                                    { 0.25, 0.80, 0.80 },
                                    { 0.27, 0.98, 0.78 },
                                    { 2.87, 0.90, 1.10 } };
    std::vector<std::pair<int, int>> bonds       = { { 0, 3 }, { 2, 3 }, { 1, 4 } };
    int                              numEmbedded = 3;

    /*! \brief A cap with a neighbour beyond the interaction range of the cap's MM atom.
     *
     * In a cubic box of 4 nm, along x: embedded atom 0, MM atom 1 bonded to it 0.25 nm
     * further, and embedded atom 2 0.57 nm before the cap of the bond, within the cutoff of
     * 0.6 nm, but 0.7 nm from MM atom 1. A large epsilon makes their pair count.
     */
    static LinkCapSystem withANeighbourBeyondTheRangeOfTheMMAtom()
    {
        LinkCapSystem system;
        system.epsilon     = 100;
        system.box         = { DVec(4, 0, 0), DVec(0, 4, 0), DVec(0, 0, 4) };
        system.positions   = { { 1.75, 2, 2 }, { 1.30, 2.01, 2 }, { 2.0, 2, 2 } };
        system.bonds       = { { 0, 2 } };
        system.numEmbedded = 2;
        return system;
    }

    //! Options of metatomic-core's Lennard-Jones model with these parameters
    std::string modelOptions() const
    {
        return formatString(
                "sigma=%.17g epsilon=%.17g cutoff=%.17g atomic_type=18 length_unit=nm "
                "energy_unit=kJ/mol",
                sigma,
                epsilon,
                cutoff);
    }

    //! The capped bonds, as the topology preprocessing gives them
    std::vector<LinkFrontierAtom> linkFrontier() const
    {
        std::vector<LinkFrontierAtom> frontier;
        for (const auto& [embedded, mm] : bonds)
        {
            frontier.emplace_back(embedded, mm);
            frontier.back().setLinkDistance(capDistance);
            frontier.back().setLinkAtomNumber(18);
        }
        return frontier;
    }

    //! Minimum image of \p v among the 27 nearest, in any box
    static DVec minimumImage(const DVec& v, const DMatrix& box)
    {
        DVec best = v;
        for (double a = -1; a <= 1; ++a)
        {
            for (double b = -1; b <= 1; ++b)
            {
                for (double c = -1; c <= 1; ++c)
                {
                    const DVec image = v + a * box[XX] + b * box[YY] + c * box[ZZ];
                    if (image.norm2() < best.norm2())
                    {
                        best = image;
                    }
                }
            }
        }
        return best;
    }

    //! Energy of the model on the embedded atoms and the caps, for \p x and \p box
    double energy(const std::vector<DVec>& x, const DMatrix& box) const
    {
        std::vector<DVec> atoms(x.begin(), x.begin() + numEmbedded);
        for (const auto& [embedded, mm] : bonds)
        {
            const DVec bond = minimumImage(x[mm] - x[embedded], box);
            atoms.push_back(x[embedded] + (capDistance / bond.norm()) * bond);
        }
        // metatomic-core's model shifts the energy to zero at the cutoff
        const auto lennardJones = [this](double r)
        {
            const double sr6 = std::pow(sigma / r, 6);
            return 4 * epsilon * (sr6 * sr6 - sr6);
        };
        double energy = 0;
        for (size_t i = 0; i < atoms.size(); ++i)
        {
            for (size_t j = i + 1; j < atoms.size(); ++j)
            {
                const double r = minimumImage(atoms[j] - atoms[i], box).norm();
                if (r < cutoff)
                {
                    energy += lennardJones(r) - lennardJones(cutoff);
                }
            }
        }
        return energy;
    }

    //! Energy at the positions of the system
    double energy() const { return energy(positions, box); }

    //! Forces on the atoms, by central differences
    std::vector<DVec> forces() const
    {
        const double      h = 1e-5;
        std::vector<DVec> f(positions.size());
        for (size_t i = 0; i < positions.size(); ++i)
        {
            for (int d = 0; d < DIM; ++d)
            {
                auto plus = positions, minus = positions;
                plus[i][d] += h;
                minus[i][d] -= h;
                f[i][d] = -(energy(plus, box) - energy(minus, box)) / (2 * h);
            }
        }
        return f;
    }

    /*! \brief GROMACS virial, half the derivative of the energy with respect to the strain.
     *
     * Strain e_ab moves component b of every position and box vector by e_ab times component a.
     */
    DMatrix virial() const
    {
        const double h      = 1e-5;
        const auto   strain = [&](int a, int b, double e)
        {
            auto x = positions;
            auto s = box;
            for (auto& v : x)
            {
                v[b] += e * v[a];
            }
            for (auto& v : s)
            {
                v[b] += e * v[a];
            }
            return energy(x, s);
        };
        DMatrix virial;
        for (int a = 0; a < DIM; ++a)
        {
            for (int b = 0; b < DIM; ++b)
            {
                virial[a][b] = 0.5 * (strain(a, b, h) - strain(a, b, -h)) / (2 * h);
            }
        }
        return virial;
    }
};

} // namespace gmx::test

#endif
