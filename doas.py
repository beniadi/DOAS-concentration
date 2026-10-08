# -*- coding: utf-8 -*-
"""The DOAS retrieval: slant column densities by linear least squares.

    Reference(xs, slit, lo, hi)      sigma_eff = sigma_high-res (*) slit, ready to sample
    optical_depth(I, I0)             ln(I0 / I)
    fit_doas(wl, od, refs, ...)      SCD of every gas, polynomial, residual
    air_number_density(T, P)         molecule/cm3

THE MODEL
---------
In the fit window the measured optical depth is

    ln(I0(l) / I(l)) = sum_i SCD_i * sigma_eff,i(l + d) + P(l) + residual

    SCD_i        slant column density, molecule/cm2 (the unknowns)
    sigma_eff,i  the gas's high-resolution cross section convolved with the
                 slit function
    P(l)         a polynomial (Legendre, order 3-5 typically) that takes up
                 everything broadband: Rayleigh and Mie extinction, lamp and
                 telescope throughput, the broad part of every cross section
    d            an optional wavelength shift (and squeeze) between the
                 spectrum and the cross sections

For a fixed shift the model is LINEAR in SCD_i and the polynomial
coefficients, so one least-squares solve gives them all, with their
covariance.  The shift, when fitted, is the only non-linear parameter: an
outer search over it, each step a linear solve (variable projection), and
the final covariance from the full Jacobian.

A fit with sigma and P is the same fit as one with the differential cross
section sigma' = sigma - (its own polynomial) and P' - the two span the same
space - so the "differential" plots use sigma' and SCD * sigma'.

From the SCD, with the light path L, temperature T and pressure P:

    n_i  = SCD_i / L                      molecule/cm3 (path average)
    x_i  = n_i / N_air,  N_air = P/(k_B T)   mixing ratio (ppb, ppm)
"""

import time

import numpy as np
from numpy.polynomial import legendre as npleg
from scipy.optimize import least_squares

import slit as slit_mod

K_B = 1.380649e-23
UNITS = {"ppb": 1e9, "ppm": 1e6, "ppt": 1e12}

DEFAULTS = {
    "poly_order": 3,          # Legendre polynomial order
    "fit_shift": True, "max_shift": 0.15,   # nm
    "fit_squeeze": False, "max_squeeze": 0.002,
    "oversample": 10,          # convolution grid points per slit FWHM
    "unit": "ppb",
}


def air_number_density(T_K, P_Pa):
    """molecule/cm3."""
    return P_Pa / (K_B * T_K) * 1e-6


def optical_depth(I, I0):
    """ln(I0/I), NaN where either is not positive."""
    I, I0 = np.asarray(I, float), np.asarray(I0, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        od = np.log(I0 / I)
    od[~((I > 0) & (I0 > 0))] = np.nan
    return od


# =============================================================================
# References
# =============================================================================
class Reference:
    """A cross section convolved with the slit on a fine uniform grid
    covering the window plus a margin for the shift."""

    def __init__(self, xs, slit_x, slit_y, lo, hi, oversample=10, margin=1.0):
        self.gas = xs["gas"]
        self.source = xs.get("source", "")
        self.xs_range = (float(xs["wl"][0]), float(xs["wl"][-1]))
        self.grid, self.conv = slit_mod.convolve_uniform(xs["wl"], xs["sigma"], slit_x, slit_y,
                                                         lo - margin, hi + margin, oversample)
        self.covers = self.xs_range[0] <= lo and self.xs_range[1] >= hi

    def __call__(self, wl):
        return np.interp(wl, self.grid, self.conv)


def make_references(xs_list, slit_x, slit_y, lo, hi, oversample=10):
    return [Reference(xs, slit_x, slit_y, lo, hi, oversample) for xs in xs_list]


# =============================================================================
# Fit
# =============================================================================
def _norm(wl, lo, hi):
    return 2.0 * (wl - lo) / (hi - lo) - 1.0


def poly_basis(wl, order, lo, hi):
    return npleg.legvander(_norm(wl, lo, hi), int(order))


def differential(y, wl, order, lo, hi):
    """y minus its own least-squares Legendre polynomial."""
    B = poly_basis(wl, order, lo, hi)
    c, *_ = np.linalg.lstsq(B, y, rcond=None)
    return y - B @ c


def _design(wl, refs, order, lo, hi, shift, squeeze):
    wc = 0.5 * (lo + hi)
    w = wl + shift + squeeze * (wl - wc)
    S = np.column_stack([r(w) for r in refs]) if refs else np.zeros((len(wl), 0))
    return S, poly_basis(wl, order, lo, hi)


def _solve(od, S, B):
    """Linear least squares with the columns scaled to unit norm (the cross
    sections are ~1e-19, the polynomial ~1)."""
    A = np.hstack([S, B])
    scale = np.linalg.norm(A, axis=0); scale[scale == 0] = 1.0
    c, *_ = np.linalg.lstsq(A / scale, od, rcond=None)
    c = c / scale
    return c, od - A @ c, A


def fit_doas(wl, od, refs, region=None, poly_order=3, fit_shift=True, max_shift=0.15,
             fit_squeeze=False, max_squeeze=0.002, T_K=293.15, P_Pa=101325.0, L_cm=1e5,
             unit="ppb", **_ignored):
    """The DOAS fit in `region` (nm).  Returns a dict - see the keys at the end."""
    t0 = time.perf_counter()
    wl = np.asarray(wl, float); od = np.asarray(od, float)
    lo, hi = region if region else (wl[0], wl[-1])
    m = (wl >= lo) & (wl <= hi) & np.isfinite(od)
    x, y = wl[m], od[m]
    G, K = len(refs), int(poly_order) + 1
    if len(x) < G + K + 3:
        raise ValueError("only %d points in the window for %d parameters" % (len(x), G + K))

    def resid(p):
        S, B = _design(x, refs, poly_order, lo, hi, p[0], p[1] if len(p) > 1 else 0.0)
        return _solve(y, S, B)[1]

    nl = []                      # the non-linear parameters: shift, squeeze
    if fit_shift:
        # coarse scan first: the cost has side minima one band spacing away
        grid = np.linspace(-max_shift, max_shift, 61)
        costs = [np.sum(resid([d]) ** 2) for d in grid]
        p0 = [grid[int(np.argmin(costs))]]; lb = [-max_shift]; ub = [max_shift]
        if fit_squeeze:
            p0.append(0.0); lb.append(-max_squeeze); ub.append(max_squeeze)
        sol = least_squares(resid, p0, bounds=(lb, ub), x_scale=[max_shift] + [max_squeeze] * (len(p0) - 1),
                            diff_step=1e-4, xtol=1e-12, ftol=1e-12)
        nl = list(sol.x)
    shift = nl[0] if nl else 0.0
    squeeze = nl[1] if len(nl) > 1 else 0.0

    S, B = _design(x, refs, poly_order, lo, hi, shift, squeeze)
    c, r, A = _solve(y, S, B)
    n, p = len(y), A.shape[1] + len(nl)
    dof = max(n - p, 1)
    s2 = float(r @ r) / dof
    # the Jacobian of the residual in all parameters: the linear columns and,
    # for the shift and squeeze, the numerical derivative of the model
    J = A.copy()
    if nl:
        cols = []
        for j in range(len(nl)):
            h = 1e-4 * (max_shift if j == 0 else max_squeeze)
            q = list(nl); q[j] += h
            S2, _ = _design(x, refs, poly_order, lo, hi, q[0], q[1] if len(q) > 1 else 0.0)
            q[j] -= 2 * h
            S1, _ = _design(x, refs, poly_order, lo, hi, q[0], q[1] if len(q) > 1 else 0.0)
            cols.append(((S2 - S1) @ c[:G]) / (2 * h))
        J = np.hstack([A, np.column_stack(cols)])
    sc = np.linalg.norm(J, axis=0); sc[sc == 0] = 1.0
    cov = np.linalg.pinv((J / sc).T @ (J / sc)) / np.outer(sc, sc) * s2
    err = np.sqrt(np.clip(np.diag(cov), 0, None))
    d = np.sqrt(np.clip(np.diag(cov[:G, :G]), 1e-300, None))
    corr = cov[:G, :G] / np.outer(d, d) if G else np.zeros((0, 0))

    scd, scd_err = c[:G], err[:G]
    poly = B @ c[G:G + K]
    model = y - r
    N_air = air_number_density(T_K, P_Pa)
    f = UNITS.get(unit, 1e9)
    gases = [rf.gas for rf in refs]
    comp, diff_meas, sig_eff, sig_diff = {}, {}, {}, {}
    for i, g in enumerate(gases):
        sd = differential(S[:, i], x, poly_order, lo, hi)
        sig_eff[g], sig_diff[g] = S[:, i], sd
        comp[g] = scd[i] * sd
        diff_meas[g] = comp[g] + r
    # the total differential optical depth: the measured OD minus its own
    # polynomial.  The residual is orthogonal to the polynomial basis (both
    # were in the fit), so diff_total = sum_i SCD_i sigma'_i + residual
    # exactly: subtracting every gas in turn leaves the residual.
    diff_total = differential(y, x, poly_order, lo, hi)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    warnings = []
    if nl and abs(shift) > 0.98 * max_shift:
        warnings.append("shift %.4f nm is at its bound ±%.3g nm" % (shift, max_shift))
    if len(nl) > 1 and abs(squeeze) > 0.98 * max_squeeze:
        warnings.append("squeeze at its bound ±%.3g" % max_squeeze)
    for i, g in enumerate(gases):
        if not refs[i].covers:
            warnings.append("%s cross section (%.1f-%.1f nm) does not cover the window" % ((g,) + refs[i].xs_range))
        if scd[i] < -2 * scd_err[i]:
            warnings.append("%s SCD is negative beyond 2σ" % g)
        for j in range(i + 1, G):
            if abs(corr[i, j]) > 0.9:
                warnings.append("%s and %s are strongly correlated (r = %.2f)" % (g, gases[j], corr[i, j]))
    return {
        "gases": gases, "x": x, "od": y, "fit": model, "residual": r, "poly": poly,
        "poly_coef": c[G:G + K], "components": comp, "diff_meas": diff_meas,
        "diff_total": diff_total, "diff_fit_total": sum(comp.values()) if comp else np.zeros_like(y),
        "sigma_eff": sig_eff, "sigma_diff": sig_diff,
        "scd": dict(zip(gases, scd)), "scd_err": dict(zip(gases, scd_err)),
        "n": dict(zip(gases, scd / L_cm)), "n_err": dict(zip(gases, scd_err / L_cm)),
        "mix": dict(zip(gases, scd / L_cm / N_air * f)), "mix_err": dict(zip(gases, scd_err / L_cm / N_air * f)),
        "unit": unit, "corr": corr, "N_air": N_air, "L_cm": L_cm, "T_K": T_K, "P_Pa": P_Pa,
        "shift": shift, "shift_err": err[A.shape[1]] if nl else None,
        "squeeze": squeeze, "squeeze_err": err[A.shape[1] + 1] if len(nl) > 1 else None,
        "fit_shift": bool(nl), "fit_squeeze": len(nl) > 1,
        "rms": float(np.sqrt(np.mean(r ** 2))), "chi2_red": s2, "r2": 1 - float(r @ r) / ss_tot if ss_tot else float("nan"),
        "peak_diff_od": {g: float(np.ptp(comp[g])) for g in gases},
        "npts": n, "dof": dof, "region": (lo, hi), "poly_order": int(poly_order),
        "warnings": warnings, "elapsed_s": time.perf_counter() - t0,
    }


def cascade_steps(r):
    """[(title, remaining data, model of what remains, gases removed so far)]:
    the total differential OD, then one gas subtracted at a time, until only
    the residual is left (the last step's data is r["residual"] to machine
    precision)."""
    gases = r["gases"]
    data = np.array(r["diff_total"], float)
    steps = [("Total differential OD  =  ln(I₀/I) − polynomial (order %d)" % r["poly_order"],
              data.copy(), np.array(r["diff_fit_total"], float), [])]
    for i, g in enumerate(gases):
        data = data - r["components"][g]
        rest = gases[i + 1:]
        model = sum((r["components"][h] for h in rest), np.zeros_like(data))
        if rest:
            title = "− %s   →   remaining %s" % (g, " + ".join(rest))
        else:
            title = "− %s   →   residual" % g
        steps.append((title, data.copy(), model, gases[:i + 1]))
    return steps
