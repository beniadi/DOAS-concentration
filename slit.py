# -*- coding: utf-8 -*-
"""Slit function (the UV spectrometer's instrument line shape) for UV-DOAS.

    synthetic_slit(kind, ...)      Gaussian, asymmetric Gaussian, super-Gaussian
                                   (flat top), boxcar (*) Gaussian, Voigt
    load_slit(path)                a measured slit (e.g. a Hg line), two columns
    slit_stats(x, y)               FWHM, centroid, asymmetry, area
    kernel(x, y, h)                the slit on a grid of step h, unit sum
    convolve_uniform(...)          a high-resolution spectrum (*) slit, on a uniform grid

WHERE THE CONVOLUTION GOES
--------------------------
In UV-DOAS the absorption is weak (tau << 1), so exp(-tau) (*) S is
exp(-(tau (*) S)) to first order and the slit is applied to the CROSS
SECTION itself: sigma_eff = sigma_high-res (*) S.  The model then stays
linear in the slant column densities (doas.py).  simulate_spectrum.py
instead convolves the INTENSITY, as the spectrometer does, so the examples
test that approximation too.

All offsets are in nm; the slit is taken as the same over the fit window.
"""

import numpy as np
from scipy.signal import fftconvolve
from scipy.special import erf, voigt_profile

_trapz = getattr(np, "trapezoid", None) or np.trapz

KINDS = {
    "gauss": "Gaussian",
    "asym": "Asymmetric Gaussian",
    "supergauss": "Super-Gaussian (flat top)",
    "boxgauss": "Boxcar ⊗ Gaussian",
    "voigt": "Voigt",
}
# the parameters of every kind: (key, label, default, lo, hi, step, decimals, tooltip)
PARAMS = {
    "gauss": (("fwhm", "FWHM (nm)", 0.50, 0.01, 10, 0.01, 3, "Full width at half maximum."),),
    "asym": (("fwhm", "FWHM (nm)", 0.50, 0.01, 10, 0.01, 3, "Full width at half maximum."),
             ("asym", "Asymmetry", 0.15, -0.9, 0.9, 0.01, 3,
              "Right half-width = FWHM/2·(1 + a), left = FWHM/2·(1 − a).\n"
              "Positive a: tail towards longer wavelength.")),
    "supergauss": (("fwhm", "FWHM (nm)", 0.50, 0.01, 10, 0.01, 3, "Full width at half maximum."),
                   ("k", "Exponent k", 3.0, 1.0, 12.0, 0.1, 2,
                    "exp(−ln2·|2x/FWHM|^k): k = 2 is a Gaussian, larger k a flatter top.")),
    "boxgauss": (("box", "Slit width (nm)", 0.40, 0.0, 10, 0.01, 3,
                  "Width of the entrance-slit image (boxcar) on the detector."),
                 ("fwhm", "Blur FWHM (nm)", 0.30, 0.01, 10, 0.01, 3,
                  "Gaussian blur from optics and pixels, convolved with the boxcar.")),
    "voigt": (("fwhm", "Gauss FWHM (nm)", 0.45, 0.01, 10, 0.01, 3, "Gaussian part."),
              ("lorentz", "Lorentz FWHM (nm)", 0.05, 0.0, 5, 0.005, 3,
               "Lorentzian part - the far wings (stray light near the line).")),
}
DEFAULT_KIND = "gauss"


def default_params(kind):
    return {p[0]: p[2] for p in PARAMS[kind]}


# =============================================================================
# Building a slit
# =============================================================================
def _g(x, fwhm):
    return np.exp(-4.0 * np.log(2.0) * (x / fwhm) ** 2)


def _width_hint(kind, p):
    if kind == "boxgauss":
        return float(p.get("box", 0)) + float(p.get("fwhm", 0.3))
    if kind == "voigt":
        return 0.5346 * p.get("lorentz", 0) + np.sqrt(0.2166 * p.get("lorentz", 0) ** 2 + p["fwhm"] ** 2)
    return float(p["fwhm"])


def synthetic_slit(kind="gauss", params=None, step=None, half_width=None):
    """(offset_nm, values) with unit area and its maximum at offset 0
    (the asymmetric one has its peak at 0, so its centroid is off zero)."""
    p = default_params(kind)
    p.update(params or {})
    w = _width_hint(kind, p)
    step = step or w / 60.0
    if half_width is None:
        half_width = (12.0 if kind == "voigt" and p.get("lorentz", 0) > 0 else 3.0) * w
    x = np.arange(-np.ceil(half_width / step), np.ceil(half_width / step) + 1) * step
    if kind == "gauss":
        y = _g(x, p["fwhm"])
    elif kind == "asym":
        a = float(p["asym"])
        y = np.where(x >= 0, _g(x, p["fwhm"] * (1 + a)), _g(x, p["fwhm"] * (1 - a)))
    elif kind == "supergauss":
        y = np.exp(-np.log(2.0) * np.abs(2.0 * x / p["fwhm"]) ** float(p["k"]))
    elif kind == "boxgauss":
        s = p["fwhm"] / (2 * np.sqrt(2 * np.log(2)))
        b = float(p["box"])
        y = _g(x, p["fwhm"]) if b <= 0 else 0.5 * (erf((x + b / 2) / (s * np.sqrt(2))) - erf((x - b / 2) / (s * np.sqrt(2))))
    elif kind == "voigt":
        s = p["fwhm"] / (2 * np.sqrt(2 * np.log(2)))
        y = voigt_profile(x, s, max(p["lorentz"], 0.0) / 2.0)
    else:
        raise ValueError("unknown slit kind %r" % kind)
    y = np.maximum(y, 0.0)
    return x, y / _trapz(y, x)


def describe(kind, params):
    return "%s (%s)" % (KINDS.get(kind, kind), ", ".join("%s %.4g" % (k, v) for k, v in params.items()))


def load_slit(path):
    """A measured slit: two columns (wavelength or offset in nm, value).
    Absolute wavelengths (e.g. a Hg 302.15 nm line) are moved so the peak is
    at 0; a constant background is removed; the area is normalised to 1."""
    from cross_section import read_columns
    data, _ = read_columns(path, max_cols=2)
    x, y = data[:, 0], data[:, 1]
    o = np.argsort(x); x, y = x[o], y[o]
    y = y - min(y[0], y[-1], np.percentile(y, 2))
    if not (x[0] < 0 < x[-1]):
        x = x - _peak(x, y)
    y = np.maximum(y, 0)
    return x, y / _trapz(y, x)


def _peak(x, y):
    i = int(np.argmax(y))
    if 0 < i < len(y) - 1:                         # parabola through the top three
        d = y[i - 1] - 2 * y[i] + y[i + 1]
        if d < 0:
            return x[i] + 0.5 * (y[i - 1] - y[i + 1]) / d * (x[i + 1] - x[i])
    return x[i]


def slit_stats(x, y):
    """{fwhm, left, right, centroid, asymmetry, area}: left/right are the
    half-maximum crossings relative to the peak."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    i = int(np.argmax(y)); h = y[i] / 2.0
    j = i
    while j > 0 and y[j] > h: j -= 1
    xl = np.interp(h, [y[j], y[j + 1]], [x[j], x[j + 1]]) if j < i else x[0]
    k = i
    while k < len(y) - 1 and y[k] > h: k += 1
    xr = np.interp(h, [y[k], y[k - 1]], [x[k], x[k - 1]]) if k > i else x[-1]
    area = _trapz(y, x)
    xp = _peak(x, y)
    return {"fwhm": xr - xl, "left": xp - xl, "right": xr - xp,
            "centroid": _trapz(x * y, x) / area if area else 0.0,
            "asymmetry": ((xr - xp) - (xp - xl)) / (xr - xl) if xr > xl else 0.0, "area": area}


# =============================================================================
# Convolution
# =============================================================================
def kernel(slit_x, slit_y, h):
    """The slit sampled at k*h, k = -M..M, unit sum (area preserving)."""
    M = max(1, int(np.floor(min(-slit_x[0], slit_x[-1]) / h)))
    k = np.arange(-M, M + 1) * h
    v = np.interp(k, slit_x, slit_y, left=0.0, right=0.0)
    s = v.sum()
    if s <= 0:
        raise ValueError("the slit function is empty at step %.3g nm" % h)
    return k, v / s


def grid_step(wl_hr, slit_x, slit_y, oversample=10):
    """A uniform step fine enough for the slit and no coarser than the data."""
    native = float(np.median(np.diff(wl_hr))) if len(wl_hr) > 1 else 0.01
    fwhm = slit_stats(slit_x, slit_y)["fwhm"]
    return min(native, fwhm / max(int(oversample), 2))


def convolve_uniform(wl_hr, y_hr, slit_x, slit_y, lo, hi, oversample=10, step=None):
    """(grid, y (*) slit) on a uniform grid from lo to hi.  Outside the
    tabulated range y is held at its end values; the kernel half-width is
    added on both sides internally so the edges are right."""
    h = step or grid_step(wl_hr, slit_x, slit_y, oversample)
    k, ker = kernel(slit_x, slit_y, h)
    pad = k[-1] + h
    n0 = int(np.floor((lo - pad) / h)); n1 = int(np.ceil((hi + pad) / h))
    g = np.arange(n0, n1 + 1) * h
    y = np.interp(g, wl_hr, y_hr)
    # measured(l) = sum y(l - k) S(k): a line at l0 is recorded as S(l - l0)
    c = fftconvolve(y, ker, mode="same")
    m = (g >= lo - h) & (g <= hi + h)
    return g[m], c[m]
