/*
 * Copyright 2019 SciComp, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file distributions.h
 *
 */

#pragma once

#if defined(__CUDACC__)
#include <sci/cuda/cuda_common.h>
#else
#define __host__
#define __device__
#define __scifinance_exec_check_disable__
#endif

#ifndef SCI_SQRT_2PI
#define SCI_SQRT_2PI 2.50662827463100050241
#endif

namespace sci
{

// Normal standard distribution: Pdf, Cdf and ICdf (Quantile)
template<typename Real>
__host__ __device__ Real normPdf(Real x)
{
    /**
     * Normal probability distribution function
     */
    return exp(-0.5 * x * x) / SCI_SQRT_2PI;
}

template<typename Real>
__host__ __device__ Real normCdf(Real x)
{
    /**
     * Cumulative Normal distribution function
     */
    const Real a[4] = {3.1611237438705e+00,
                       1.1386415415105e+02,
                       3.7748523768530e+02,
                       3.2093775891384e+03};
    const Real b[4] = {2.3601290952344e+01,
                       2.4402463793444e+02,
                       1.2826165260773e+03,
                       2.8442368334391e+03};
    const Real c[8] = {5.6418849698867e-01,
                       8.8831497943883e+00,
                       6.6119190637141e+01,
                       2.9863513819740e+02,
                       8.8195222124176e+02,
                       1.7120476126340e+03,
                       2.0510783778260e+03,
                       1.2303393547979e+03};
    const Real d[8] = {1.5744926110709e+01,
                       1.1769395089131e+02,
                       5.3718110186200e+02,
                       1.6213895745666e+03,
                       3.2907992357334e+03,
                       4.3626190901432e+03,
                       3.4393676741437e+03,
                       1.2303393548037e+03};
    const Real p[5] = {3.0532663496123e-01,
                       3.6034489994980e-01,
                       1.2578172611122e-01,
                       1.6083785148742e-02,
                       6.5874916152983e-04};
    const Real q[5] = {2.5685201922898e+00,
                       1.8729528499234e+00,
                       5.2790510295142e-01,
                       6.0518341312441e-02,
                       2.3352049762686e-03};

    const Real anorm = 1.8577770618460e-01;
    const Real cnorm = 2.1531153547440e-08;
    const Real pnorm = 1.6315387137302e-02;
    const Real rrtpi = 5.6418958354776e-01;
    Real xbreak, result, y, z, xden, xnum, del, ax, xs2;
    int i;

    xbreak = 0.46875;
    xs2 = x * 0.70710678118655;
    ax = fabs(xs2);

    if (ax <= xbreak)
    { // evaluate  erf  for  |x| <= 0.46875
        y = ax;
        z = y * y;
        xnum = anorm * z;
        xden = z;
        for (i = 0; i <= 2; i++)
        {
            xnum = (xnum + a[i]) * z;
            xden = (xden + b[i]) * z;
        }
        result = xs2 * (xnum + a[3]) / (xden + b[3]);
    }
    else if ((ax > xbreak) & (ax <= 4.))
    { // evaluate  erfc  for 0.46875 <= |x| <= 4.0
        y = ax;
        xnum = cnorm * y;
        xden = y;
        for (i = 0; i <= 6; i++)
        {
            xnum = (xnum + c[i]) * y;
            xden = (xden + d[i]) * y;
        }
        result = (xnum + c[7]) / (xden + d[7]);
        z = floor(y * 16.) / 16.;
        del = (y - z) * (y + z);
        result = result * exp(-z * z - del);
    }
    else
    { // evaluate  erfc  for |x| > 4.0
        y = ax;
        z = 1. / (y * y);
        xnum = pnorm * z;
        xden = z;
        for (i = 0; i <= 3; i++)
        {
            xnum = (xnum + p[i]) * z;
            xden = (xden + q[i]) * z;
        }
        result = z * (xnum + p[4]) / (xden + q[4]);
        result = (rrtpi - result) / y;
        z = floor(y * 16.) / 16.;
        del = (y - z) * (y + z);
        result = result * exp(-z * z - del);
    }
    if (ax > xbreak)
    {
        result = 1. - result;
    }
    if (xs2 <= -xbreak)
    {
        result = -result;
    }

    return 0.5 * (1. + result);
}

template<typename Real>
__host__ __device__ Real normICdf(Real ZU)
{
    /**
     * Inverse Cumulative Normal distribution function (quantile)
     * The expansion coefficients are due to Peter J. Acklam
     * http://home.online.no/~pjacklam
     * This is more accurate and approx 70% faster than
     * the Moro and Boyle algorithms
     * The relative error of the approximation has absolute value less
     * than 1.15e-9
     */

    Real plow, phigh, r, q;

    const Real a[6] = {-3.969683028665376e+01,
                       2.209460984245205e+02,
                       -2.759285104469687e+02,
                       1.383577518672690e+02,
                       -3.066479806614716e+01,
                       2.506628277459239e+00};

    const Real b[5] = {-5.447609879822406e+01,
                       1.615858368580409e+02,
                       -1.556989798598866e+02,
                       6.680131188771972e+01,
                       -1.328068155288572e+01};

    const Real c[6] = {-7.784894002430293e-03,
                       -3.223964580411365e-01,
                       -2.400758277161838e+00,
                       -2.549732539343734e+00,
                       4.374664141464968e+00,
                       2.938163982698783e+00};

    const Real d[4] = {7.784695709041462e-03,
                       3.224671290700398e-01,
                       2.445134137142996e+00,
                       3.754408661907416e+00};

    /* break points  */
    plow = 0.02425;
    phigh = 1.0 - plow;

    /* rational approximation for lower region. */
    if (ZU <= plow)
    {
        q = sqrt(-2.0 * log(ZU));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q +
                c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.);
    }

    /* rational approximation for central region. */
    else if (ZU <= phigh)
    {
        q = ZU - 0.5;
        r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r +
                a[5]) *
            q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r +
             1.0);
    }

    /* rational approximation for upper region. */

    else
    {
        q = sqrt(-2.0 * log(1.0 - ZU));
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q +
                 c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
}

// Poisson distribution: ICdf (Quantile)
template<typename Real>
__host__ __device__ int poissonICdf(Real ZU, Real lambda)
{
    Real p, F;
    int N;

    N = 0;
    p = exp(-lambda);
    F = p;
    while (ZU > F)
    {
        N++;
        p = p * lambda / N;
        F = F + p;
    }

    return N;
}

template<typename Real>
__host__ __device__ void
gammaDist(Real x, Real a, Real* gammacdf, Real* gammapdf, Real gammalog)
{
    /* gammapdf(x,a) = x^(a-1) exp(-x) / gamma(a) */
    /* gammacdf(x,a) = integral_0^x gammapdf(t,a) dt = gammainc(x,a)/gamma(a)*/
    /* where gammainc is lower incomplete gamma function */
    /* gammainc(x,a) = integral_0^x t^(a-1) exp(-t) dt */
    int n;
    Real fac, h, hold, nf, na, logx, temp, ag, sum, dsum, p0, p1, q0, q1;

    logx = log(x);
    temp = exp(-x + a * logx - gammalog);
    *gammapdf = temp / x;

    /* cdf:  x<a+1  series expansion */
    if (x < a + 1.)
    {
        ag = a;
        sum = 1. / ag;
        dsum = sum;
        while (dsum >= 1.e-8 * sum)
        {
            ag += 1.;
            dsum *= (x / ag);
            sum += dsum;
        }
        *gammacdf = sum * temp;
    }
    /* cdf:  x>=a+1  continued fraction */
    else
    {
        p0 = 1.;
        p1 = x;
        q0 = 0.;
        q1 = p0;
        fac = 1.;
        n = 1;
        h = q1;
        hold = q0;
        while (fabs(h - hold) >= 1.e-8 * fabs(h))
        {
            na = n - a;
            p0 = (p1 + p0 * na) * fac;
            q0 = (q1 + q0 * na) * fac;
            nf = n * fac;
            p1 = x * p0 + nf * p1;
            q1 = x * q0 + nf * q1;
            fac = 1. / p1;
            hold = h;
            h = q1 * fac;
            n++;
        }
        *gammacdf = 1. - h * temp;
    }
}

// Gamma distribution: Pdf, Cdf and ICdf (Quantile)
template<typename Real>
__host__ __device__ Real gammaPdf(Real x, Real a)
{
    return exp(-x - lgamma(a + 1.e-14)) * pow(x, a - 1);
}
template<typename Real>
__host__ __device__ Real gammaCdf(Real x, Real a)
{
    Real cdf, pdf;
    gammaDist(x, a, &cdf, &pdf, lgamma(a + 1.e-14));
    return cdf;
}

template<typename Real>
__host__ __device__ Real gammaICdf(Real ZU, Real a)
{
    Real r, dr, cdf, pdf, gammalog;

    r = a - 1.;
    if (r < 0.1)
        r = 0.1;
    if (a > 10.)
    {
        r = a * exp(normICdf(ZU) / sqrt(a));
    }
    dr = 1;
    gammalog = lgamma(a + 1.e-14);
    while (fabs(dr) > 1.e-08 * (r > 1. ? r : 1.))
    {
        gammaDist(r, a, &cdf, &pdf, gammalog);
        dr = -(cdf - ZU) / pdf;
        if (dr < -0.5 * r)
            dr = -0.5 * r;
        r = r + dr;
    }
    return r;
}

// Chi-squared distribution: Pdf, Cdf, ICdf
template<typename Real>
__host__ __device__ Real chi2Pdf(Real x, Real r)
{
    return 0.5 * gammaPdf(0.5 * x, 0.5 * r);
}

template<typename Real>
__host__ __device__ Real chi2Cdf(Real x, Real r)
{
    return gammaCdf(0.5 * x, 0.5 * r);
}

template<typename Real>
__host__ __device__ Real chi2ICdf(Real ZU, Real r)
{
    return 2.0 * gammaICdf(ZU, 0.5 * r);
}

//  Leif Andersen's QEM discretization scheme for Heston models
template<typename Real>
__host__ __device__ void QuadraticExponential(
    Real& XS,
    Real& Xv,
    Real U1,
    Real U2,
    Real dt,
    Real kappa,
    Real theta,
    Real sigma,
    Real drift,
    Real rhoSv)
{
    Real M, psi, psi2, b2, a, zv, b, p, beta, Aa, s2, zS;
    Real ekdt, cekdt, sig2k, xv1, xv2, k0, k1, k2, k34, A, k13, rhosig, S, v;
    Real temp;

    S = XS;
    v = Xv;

    ekdt = exp(-kappa * dt);
    cekdt = 1. - ekdt;

    // kappa = 0
    xv1 = sigma * sigma * dt;
    xv2 = 0;
    if (kappa > 0.0)
    {
        sig2k = sigma * sigma / kappa;
        xv1 = sig2k * ekdt * cekdt;
        xv2 = 0.5 * theta * sig2k * cekdt * cekdt;
    }

    rhosig = rhoSv / sigma;

    k0 = -rhosig * kappa * dt * theta;
    k1 = 0.5 * dt * (kappa * rhosig - 0.5) - rhosig;
    k2 = k1 + 2 * rhosig;
    k34 = 0.5 * dt * (1. - rhoSv * rhoSv);
    A = k2 + 0.5 * k34;
    k13 = k1 + 0.5 * k34;

    M = theta * cekdt + v * ekdt;
    s2 = v * xv1 + xv2;

    // psi = 0, psi = infinity
    Xv = M;
    k0 = -A * M - k13 * v;
    if (M > 0.0)
    {
        psi = s2 / (M * M);
        if (psi > 0.0 && psi <= 1.5)
        {
            psi2 = 2. / psi;
            b2 = psi2 - 1. + sqrt(psi2 * (psi2 - 1.));
            if (b2 < 0.)
                b2 = 0.;
            a = M / (1. + b2);
            b = sqrt(b2);
            zv = normICdf(U2);
            Xv = a * (b + zv) * (b + zv);
            Aa = A * a;
            k0 = -Aa * b2 / (1. - 2. * Aa) + 0.5 * log(1. - 2. * Aa) - k13 * v;
        }
        if (psi > 1.5)
        {
            p = (psi - 1.) / (psi + 1.);
            beta = (1. - p) / M;
            /* (U2<=p) is not exactly the same as ((1.0-p)/(1.0-U2)<=1.0 */
            temp = (1.0 - p) / (1.0 - U2);
            Xv = 0.0;
            if (temp > 1.0)
            {
                Xv = log(temp) / beta;
            }
            k0 = -log(p + beta * (1 - p) / (beta - A)) - k13 * v;
        }
    }

    zS = normICdf(U1);
    XS = S *
        exp(dt * drift + k0 + k1 * v + k2 * Xv + sqrt(k34 * (v + Xv)) * zS);
}

// Random variates generators (acceptance-rejection methods)

template<typename Gen, typename Real>
__host__ __device__ Real randGamma(Gen* state, Real alpha)
{
    Real Z, b, Y, W, X, Xp, Yp, a, m, d, f, V;
    const Real e = 2.718281828459;

    /*
     * random draw from standard gamma distribution: gamma(alpha,beta=1)
     * follows G. S. Fishman "Monte Carlo", Sec. 3.14, Springer, 1995
     */

    /* degenerate case, */
    if (fabs(alpha - 1.) <= 1.e-04)
    {
        Z = -log(rngUniformDouble(state));
    }

    /* GS algorithm */
    else if (alpha < 1.)
    {
        b = (alpha + e) / e;
        while (1)
        {
            Y = b * rngUniformDouble(state);
            if (Y <= 1.)
            {
                Z = pow(Y, (1 / alpha));
                W = -log(rngUniformDouble(state));
                if (W >= Z)
                    break;
            }
            else
            {
                Z = -log((b - Y) / alpha);
                W = pow(rngUniformDouble(state), (1 / (alpha - 1)));
                if (W >= Z)
                    break;
            }
        }
    }

    /* GKM1 algorithm */
    else if (alpha < 2.5)
    {
        a = alpha - 1;
        b = (alpha - 1 / (6 * alpha)) / a;
        m = 2 / a;
        d = m + 2;
        while (1)
        {
            Xp = rngUniformDouble(state);
            Yp = rngUniformDouble(state);
            V = b * Yp / Xp;
            if (m * Xp - d + V + 1 / V <= 0.)
                break;
            if (m * log(Xp) - log(V) + V - 1. <= 0.)
                break;
        }
        Z = a * V;
    }

    /* GKM2 algorithm */
    else
    {
        a = alpha - 1;
        b = (alpha - 1 / (6 * alpha)) / a;
        m = 2 / a;
        d = m + 2;
        f = sqrt(alpha);
        while (1)
        {
            while (1)
            {
                X = rngUniformDouble(state);
                Yp = rngUniformDouble(state);
                Xp = Yp + (1. - 1.857764 * X) / f;
                if (Xp > 0. && Xp < 1.)
                    break;
            }
            V = b * Yp / Xp;
            if (m * Xp - d + V + 1 / V <= 0.)
                break;
            if (m * log(Xp) - log(V) + V - 1. <= 0.)
                break;
        }
        Z = a * V;
    }
    return Z;
}

/*****************************************************************************/
/*****************************************************************************/
template<typename Gen, typename Real>
__host__ __device__ Real randBeta(Gen& state, Real alpha, Real beta)
{
    Real ZU;
    Real U, X, Y, Z, maxab, minab, p, t, U1, U2, V, W, R, S, T, d1, d2, d3, d4,
        d5;

    maxab = alpha > beta ? alpha : beta;
    minab = alpha < beta ? alpha : beta;

    /* Algorithm AW */
    if (maxab < 1.)
    {
        t = 1. / (1. + sqrt(beta * (1. - beta) / (alpha * (1. - alpha))));
        p = beta * t / (beta * t + alpha * (1. - t));
        while (1)
        {
            state.getUniform(U);
            state.getUniform(ZU);
            Y = -log(ZU);
            if (U <= p)
            {
                Z = t * pow(U / p, 1. / alpha);
                if (Y >= (1. - beta) * (t - Z) / (1. - t))
                    break;
                if (Y >= (1. - beta) * log((1. - Z) / (1. - t)))
                    break;
            }
            else
            {
                Z = 1. - (1. - t) * pow((1. - U) / (1. - p), 1. / beta);
                if (Y >= (1. - alpha) * (Z / t - 1.))
                    break;
                if (Y >= (1. - alpha) * log(Z / t))
                    break;
            }
        }
        return Z;
    }

    /* BB* Algorithm */
    else if (minab > 1.)
    {
        d1 = minab;
        d2 = maxab;
        d3 = d1 + d2;
        d4 = (d3 - 2) / (2 * d1 * d2 - d3);
        if (d4 < 0.00000000001)
            d4 = 0.00000000001;
        d4 = sqrt(d4);
        d5 = d1 + 1 / d4;
        while (1)
        {
            state.getUniform(U1);
            state.getUniform(U2);
            V = d4 * log(U1 / (1 - U1));
            W = d1 * exp(V);
            Z = U1 * U1 * U2;
            R = d5 * V - 1.38629436;
            S = d1 + R - W;
            if (S + 2.60943791 > 5. * Z)
                break;

            T = log(Z);
            if (S >= T)
                break;

            if (R + d3 * log(d3 / (d2 + W)) >= T)
                break;
        }
        if (d1 == alpha)
        {
            Z = W / (d2 + W);
        }
        else
        {
            Z = d2 / (d2 + W);
        }
        return Z;
    }

    /* Compose from Gamma distributions */
    else
    {
        X = randGamma(state, alpha);
        Y = randGamma(state, beta);
        Z = X / (X + Y);
        return Z;
    }
}

/*****************************************************************************/
/*****************************************************************************/

template<typename Gen, typename Real>
__host__ __device__ Real randChi2(Gen& state, Real r)
{
    return 2.0 * randGamma(state, 0.5 * r);
}

/*****************************************************************************/
/*****************************************************************************/

template<typename Gen, typename Real>
__host__ __device__ Real randNonCenChi2(Gen* state, Real r, Real lambda)
{
    Real X, Y, Z, N;

    if (r > 1.)
    {
        X = 2. * randGamma(state, 0.5 * (r - 1.));
        Y = normICdf(rngUniformDouble(state)) + sqrt(lambda);
        Z = Y * Y + X;
    }
    else
    {
        N = poissonICdf(rngUniformDouble(state), 0.5 * lambda);
        Z = 2.0 * randGamma(state, 0.5 * r + N);
    }
    return Z;
}

/*****************************************************************************/
/*****************************************************************************/
template<typename Gen, typename Real>
__host__ __device__ Real randStudentT(Gen& state, int n)
{
    Real X, Y, Z, X1, X2, V, S, T, U, W, ZU;
    const Real d1 = 0.866025404;
    const Real d2 = 8.591112939;
    const Real d3 = 0.927405715;
    int m;

    /* Algorithm T3T* */
    if (n >= 3)
    {
        while (1)
        {
            while (1)
            {
                state.getUniform(X);
                state.getUniform(ZU);
                Y = 1. - 2. * ZU;
                Z = d1 * Y / X;
                T = Z * Z;
                if (X * (3 + T) <= 3.)
                    break;
            }
            if (n == 3)
                break;

            state.getUniform(U);
            W = U * U;
            if (d3 * d3 - T / d2 >= W)
                break;

            V = 1 + T / 3.;
            S = 2. * log(V * V * d3 / U);
            if (S >= T)
                break;
            m = n + 1;
            if (S >= 1. + m * log(T + n) / m)
                break;
        }
        return Z;
    }

    /* Compose from Gamma distributions */
    else
    {
        state.getUniform(ZU);
        X1 = normICdf(ZU);
        X2 = 0.5 * randGamma(state, Real(n) / 2.0);
        Z = X1 / sqrt(X2);
        return Z;
    }
}

} // namespace sci
