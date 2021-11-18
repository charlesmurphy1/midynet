// Copyright (C) 2006-2021 Tiago de Paula Peixoto <tiago@skewed.de>
//
// This program is free software; you can redistribute it and/or modify it under
// the terms of the GNU Lesser General Public License as published by the Free
// Software Foundation; either version 3 of the License, or (at your option) any
// later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
// details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

/*
 * Cephes Math Library Release 2.1:  January, 1989
 * Copyright 1985, 1987, 1989 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */

/*
 *      Dilogarithm
 *
 *
 *
 * SYNOPSIS:
 *
 * float x, y, spencef();
 *
 * y = spencef( x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Computes the integral
 *
 *                    x
 *                    -
 *                   | | log t
 * spence(x)  =  -   |   ----- dt
 *                 | |   t - 1
 *                  -
 *                  1
 *
 * for x >= 0.  A rational approximation gives the integral in
 * the interval (0.5, 1.5).  Transformation formulas for 1/x
 * and 1-x are employed outside the basic expansion range.
 *
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE      0,4         30000       4.4e-7      6.3e-8
 *
 *
 */

#include <cmath>
#include <limits>

#include "FastMIDyNet/utility/polylog2_integral.h"


namespace FastMIDyNet{

    double polevl(double x, double coef[], int N)
    {
        double ans;
        int i;
        double *p;

        p = coef;
        ans = *p++;
        i = N;

        do
        ans = ans * x + *p++;
        while (--i);

        return (ans);
    }

    double polylog2Integral(double x)
    {
        double w, y, z;
        int flag;

        if (x < 0.0)
        return std::numeric_limits<double>::quiet_NaN();

        if (x == 1.0)
        return (0.0);

        if (x == 0.0)
        return (PI * PI / 6.0);

        flag = 0;

        if (x > 2.0) {
            x = 1.0 / x;
            flag |= 2;
        }

        if (x > 1.5) {
            w = (1.0 / x) - 1.0;
            flag |= 2;
        }

        else if (x < 0.5) {
            w = -x;
            flag |= 1;
        }

        else
        w = x - 1.0;


        y = -w * polevl(w, A, 7) / polevl(w, B, 7);

        if (flag & 1)
        y = (PI * PI) / 6.0 - std::log(x) * std::log1p(-x) - y;

        if (flag & 2) {
            z = std::log(x);
            y = -0.5 * z * z - y;
        }

        return (y);
    }

}
