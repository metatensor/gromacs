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
 * RAII timer that writes nested timing information to per-rank files
 * (metatomic_timer_rank_N.log).  Enable with GMX_METATOMIC_TIMER=1.
 *
 * \author Metatensor developers <https://github.com/metatensor>
 * \ingroup module_applied_forces
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>

#include "gromacs/utility/mpicomm.h"

namespace gmx
{

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static std::mutex METATOMIC_TIMER_MUTEX = {};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static thread_local int64_t METATOMIC_TIMER_DEPTH = -1;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static bool METATOMIC_TIMER_ENABLED = false;

/*! \internal \brief RAII scoped timer for Metatomic profiling.
 *
 * Writes hierarchical timing info to a per-rank file
 * (metatomic_timer_rank_N.log).  Timers nest automatically via a
 * global depth counter.  Thread-safe via a global mutex.
 *
 * Enable with GMX_METATOMIC_TIMER=1 environment variable.
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
            this->enabled_ = true;
            this->start_   = std::chrono::high_resolution_clock::now();
        }
    }

    //! Stop the timer early (before scope exit). Safe to call multiple times.
    void stop()
    {
        auto guard_ = std::lock_guard(METATOMIC_TIMER_MUTEX);
        recordAndDisable_();
    }

    ~MetatomicTimer()
    {
        auto guard_ = std::lock_guard(METATOMIC_TIMER_MUTEX);
        recordAndDisable_();
    }

    // Non-copyable, non-movable
    MetatomicTimer(const MetatomicTimer&)            = delete;
    MetatomicTimer& operator=(const MetatomicTimer&) = delete;
    MetatomicTimer(MetatomicTimer&&)                 = delete;
    MetatomicTimer& operator=(MetatomicTimer&&)      = delete;

private:
    //! Record elapsed time and mark as done. Must be called under lock.
    void recordAndDisable_()
    {
        if (METATOMIC_TIMER_ENABLED && this->enabled_)
        {
            auto stop    = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(stop - start_).count();
            auto indent  = std::string(METATOMIC_TIMER_DEPTH * 2, ' ');

            std::string fname = "metatomic_timer_rank_" + std::to_string(mpiComm_.rank()) + ".log";
            FILE*       fp    = std::fopen(fname.c_str(), "a");
            if (fp)
            {
                std::fprintf(fp, "%s%s: %.3f ms\n", indent.c_str(), name_.c_str(), elapsed / 1e3);
                std::fclose(fp);
            }
            // Also print to stdout for immediate visibility
            std::fprintf(stdout, "[MetatomicTimer rank %d] %s%s: %.3f ms\n",
                         mpiComm_.rank(), indent.c_str(), name_.c_str(), elapsed / 1e3);

            this->enabled_ = false;
            METATOMIC_TIMER_DEPTH -= 1;
        }
    }

    bool                                           enabled_;
    std::string                                    name_;
    const MpiComm&                                 mpiComm_;
    std::chrono::high_resolution_clock::time_point start_;
};

} // namespace gmx
