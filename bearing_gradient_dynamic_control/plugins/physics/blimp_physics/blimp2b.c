/****************************************************************************

blimp_physics -- A blimp physics model for Webots.

Copyright (C) 2006 Laboratory of Intelligent Systems, EPFL, Lausanne
Authors:    Alexis Guanella            guanella@ini.phys.ethz.ch
            Antoine Beyeler            antoine.beyeler@epfl.ch
            Jean-Christophe Zufferey   jean-christophe.zufferey@epfl.ch
            Dario Floreano             dario.floreano@epfl.ch
Web: http://lis.epfl.ch

The authors of any publication arising from research using this software are
kindly requested to add the following reference:

        Zufferey, J.C., Guanella, A., Beyeler, A., Floreano, D. (2006) Flying over
        the Reality Gap: From Simulated to Real Indoor Airships. Autonomous Robots,
        Springer US.

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

******************************************************************************/
/*------------------------------------------------------------------------------

Author:		Alexis Guanella (ag)
                        Antoine Beyeler (ab)

------------------------------------------------------------------------------*/

#include "blimp2b.h"
#include <stdint.h>

//------------------------------------------------------------------------------
// Engines: command [-1.0...1.0] to thrust [N] converters

void b2b_compThrustWrench(const dReal *propThrusts, dReal *genForceAB) {
    // Propeller positions (relative to CoM)
    const dReal r[4][3] = {
        {-7.1000000e-05, -8.4940630e-02, -2.5989358e-01},
        {3.5330000e-05,   8.4895020e-02, -2.5989358e-01},
        {0.01771103, 0.14854665, -0.27755072},
        {0.01771103,-0.14852844, -0.27755224}
    };
    // Thrust directions (unit vectors)
    const dReal F_dir[4][3] = {
        {0, 0, 1},         // F0: upward
        {0, 0, 1},         // F1: upward
        {1, 0, 0},  // F2: tilted
        {1, 0, 0}   // F3: tilted
    };

    // Sum forces and torques
    for (int i = 0; i < 4; ++i) {
        // Force
        genForceAB[0] += F_dir[i][0] * propThrusts[i];
        genForceAB[1] += F_dir[i][1] * propThrusts[i];
        genForceAB[2] += F_dir[i][2] * propThrusts[i];
        // Torque: r x F
        genForceAB[3] += r[i][1] * F_dir[i][2] * propThrusts[i] - r[i][2] * F_dir[i][1] * propThrusts[i]; // x
        genForceAB[4] += r[i][2] * F_dir[i][0] * propThrusts[i] - r[i][0] * F_dir[i][2] * propThrusts[i]; // y
        genForceAB[5] += r[i][0] * F_dir[i][1] * propThrusts[i] - r[i][1] * F_dir[i][0] * propThrusts[i]; // z
    }
}
