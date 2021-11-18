// graph-tool -- a general graph modification and manipulation thingy
//
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

#include "FastMIDyNet/utility/integer_partition.h"
#include "FastMIDyNet/utility/polylog2_integral.h"
#include "FastMIDyNet/utility/functions.h"

using namespace std;

namespace FastMIDyNet
{


double q_rec(int n, int k)
{
   if (n <= 0 || k < 1)
       return 0;
   if (k > n)
       k = n;
   if (k == 1)
       return 1;
   return q_rec(n, k - 1) + q_rec(n - k, k);
}

double log_q_approx_big(size_t n, size_t k)
{
   double C = PI * sqrt(2/3.);
   double S = C * sqrt(n) - log(4 * sqrt(3) * n);
   if (k < n)
   {
       double x = k / sqrt(n) - log(n) / C;
       S -= (2 / C) * exp(-C * x / 2);
   }
   return S;
}

double log_q_approx_small(size_t n, size_t k)
{
   return logBinomialCoefficient(n - 1, k - 1) - logFactorial(k);
}

double get_v(double u, double epsilon=1e-8)
{
   double v = u;
   double delta = 1;
   while (delta > epsilon)
   {
       // polylog2Integral(exp(v)) = -polylog2Integral(exp(-v)) - (v*v)/2
       double n_v = u * sqrt(polylog2Integral(exp(-v)));
       delta = abs(n_v - v);
       v = n_v;
   }
   return v;
}

double log_q_approx(size_t n, size_t k)
{
   if (k < pow(n, 1/4.))
       return log_q_approx_small(n, k);
   double u = k / sqrt(n);
   double v = get_v(u);
   double lf = log(v) - log1p(- exp(-v) * (1 + u * u/2)) / 2 - log(2) * 3 / 2.
       - log(u) - log(PI);
   double g = 2 * v / u - u * log1p(-exp(-v));
   return lf - log(n) + sqrt(n) * g;
}


} // namespace graph_tool
