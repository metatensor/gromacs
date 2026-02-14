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
 * Scoped timer for Metatomic force provider profiling.
 *
 * RAII timer that prints nested timing information to stderr with MPI rank.
 * Enable with MetatomicTimer::enable(true) before use.
 *
 * \author Metatensor developers <https://github.com/metatensor>
 * \ingroup module_applied_forces
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string>

#include "gromacs/utility/mpicomm.h"

namespace gmx
{

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static std::mutex METATOMIC_TIMER_MUTEX = {};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static int64_t METATOMIC_TIMER_DEPTH = -1;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static uint64_t METATOMIC_TIMER_COUNTER = 0;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static bool METATOMIC_TIMER_ENABLED = false;

/*! \internal \brief RAII scoped timer for Metatomic profiling.
 *
 * Prints hierarchical timing info to stderr. Timers nest automatically.
 * Thread-safe via a global mutex.
 */
class MetatomicTimer
{
public:
    //! Enable or disable all timers globally.
    static void enable(bool toggle)
    {
        auto guard_ = std::lock_guard(METATOMIC_TIMER_MUTEX);
        METATOMIC_TIMER_ENABLED = toggle;
    }

    //! Construct a timer with the given label. Starts timing if enabled.
    MetatomicTimer(std::string name, const MpiComm& mpiComm) :
        enabled_(false), name_(std::move(name)), mpiComm_(mpiComm)
    {
        auto guard_ = std::lock_guard(METATOMIC_TIMER_MUTEX);
        if (METATOMIC_TIMER_ENABLED)
        {
            METATOMIC_TIMER_DEPTH += 1;
            METATOMIC_TIMER_COUNTER += 1;

            this->enabled_          = true;
            this->starting_counter_ = METATOMIC_TIMER_COUNTER;
            this->start_            = std::chrono::high_resolution_clock::now();
            auto indent             = std::string(METATOMIC_TIMER_DEPTH * 3, ' ');

            if (METATOMIC_TIMER_DEPTH == 0)
            {
                std::cerr << "\n";
            }
            std::cerr << "\n" << indent << this->name_ << " ...";
        }
    }

    ~MetatomicTimer()
    {
        auto guard_ = std::lock_guard(METATOMIC_TIMER_MUTEX);

        if (METATOMIC_TIMER_ENABLED && this->enabled_)
        {
            auto stop = std::chrono::high_resolution_clock::now();
            auto elapsed =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start_).count();

            if (METATOMIC_TIMER_COUNTER != starting_counter_)
            {
                auto indent = std::string(METATOMIC_TIMER_DEPTH * 3, ' ');
                std::cerr << "\n" << indent << this->name_;
            }

            std::cerr << " took " << elapsed / 1e6 << "ms (rank " << mpiComm_.rank() << ")"
                      << std::flush;
            METATOMIC_TIMER_DEPTH -= 1;
        }
    }

    // Non-copyable, non-movable
    MetatomicTimer(const MetatomicTimer&)            = delete;
    MetatomicTimer& operator=(const MetatomicTimer&) = delete;
    MetatomicTimer(MetatomicTimer&&)                 = delete;
    MetatomicTimer& operator=(MetatomicTimer&&)      = delete;

private:
    bool                                                 enabled_;
    std::string                                          name_;
    const MpiComm&                                       mpiComm_;
    uint64_t                                             starting_counter_ = 0;
    std::chrono::high_resolution_clock::time_point       start_;
};

} // namespace gmx
