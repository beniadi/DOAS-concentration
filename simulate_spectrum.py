# -*- coding: utf-8 -*-
"""Synthetic UV-DOAS measurement (active long-path DOAS) with a known answer.

The spectrum is built the way a real one is formed, independently of the
linear model in doas.py:

    tau(l)    = sum_i x_i N_air L sigma_i(l)          gases, HIGH resolution
              + sigma_Rayleigh(l) N_air L               Rayleigh
              + beta_Mie(550) (l/550)^-alpha L          Mie (aerosol)
    I_hr(l)   = lamp(l) * throughput(l) * exp(-tau(l))
    I0_hr(l)  = lamp(l)                                  (short-cut reference)
    I, I0     = (I_hr (*) slit), (I0_hr (*) slit) sampled at the detector
                pixels (+ a wavelength-calibration error), + noise

so the slit acts on the INTENSITY, as in the spectrometer, and the
retrieval's sigma (*) slit is tested as the approximation it is.

    python simulate_spectrum.py              # the two examples in Input/
    python simulate_spectrum.py --gas SO2 20 --gas O3 60 --lo 300 --hi 330 --out Input/my.txt
    python simulate_spectrum.py --gas Input/CrossSections/NO2_real.xs 15 --lo 420 --hi 460 ...

--gas takes a gas name (SO2, O3, NO2, HCHO: the laboratory cross section,
else the synthetic one) or a cross-section file, then the mixing ratio in ppb.  Writes three columns (wavelength nm,
I0, I) with a commented header that records every input.
"""

import argparse
import os

import numpy as np

import cross_section as xsm
import doas
import slit as slit_mod

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(HERE, "Input")

# the worked examples: (file, gases {name: ppb}, window, fit window)
EXAMPLES = {
    "single": {"out": os.path.join(INPUT, "SO2_simulated_15ppb.txt"),
               "gases": {"SO2": 15.0}, "lo": 295.0, "hi": 330.0, "fit": (300.0, 320.0), "noise": 3e-4},
    "multi": {"out": os.path.join(INPUT, "Mix_SO2_O3_NO2_HCHO_simulated.txt"),
              "gases": {"SO2": 8.0, "O3": 60.0, "NO2": 25.0, "HCHO": 10.0},
              "lo": 300.0, "hi": 335.0, "fit": (308.0, 326.0), "noise": 1.5e-4},
}
EXAMPLE_SLIT = ("gauss", {"fwhm": 0.5})
EXAMPLE_PATH = {"T_K": 293.15, "P_Pa": 101325.0, "L_cm": 1.0e5}   # 1 km total light path


def rayleigh_sigma(wl_nm):
    """Rayleigh cross section of air, cm2/molecule (Bates-type power law)."""
    return 4.02e-28 / (wl_nm * 1e-3) ** 4.04


def lamp_spectrum(wl):
    """A smooth Xe-arc-like lamp seen through the spectrometer: Planck at
    6000 K times a broad detector response, peak about 1."""
    lam = wl * 1e-9
    planck = 1.0 / (lam ** 5 * (np.exp(1.4388e-2 / (lam * 6000.0)) - 1.0))
    resp = np.exp(-((wl - 420.0) / 160.0) ** 2) * (1 + 0.04 * np.sin(wl / 7.0))
    y = planck * resp
    return y / y.max()


def resolve_gas(g):
    """A cross-section dict from a file path, or from a gas name: the
    laboratory cross section (Input/CrossSections, from Input/modataXs) if
    there is one, else the synthetic one."""
    if os.path.isfile(g):
        return xsm.load_xs(g)
    p = xsm.default_path(g)
    return xsm.load_xs(p) if os.path.isfile(p) else xsm.synthetic_xs(g)


def simulate(gases, lo, hi, slit_xy, T_K=293.15, P_Pa=101325.0, L_cm=1e5, pixel=0.05,
             noise=3e-4, shift=0.02, visibility_km=20.0, angstrom=1.3, throughput=(0.55, 0.08),
             counts=40000.0, seed=1, fine=0.005):
    """(wl_pixels, I0, I, info).  gases = [(xs dict, ppb), ...].

    noise       relative RMS noise on I and on I0 (co-added spectra)
    shift       wavelength-calibration error, nm: pixel l really sees l + shift
    throughput  (level, slope over the window) of the light that came back,
                relative to the short-cut I0 - turbulence, alignment, retro
    """
    rng = np.random.default_rng(seed)
    sx, sy = slit_xy
    margin = 2.0 + 2 * abs(shift) + sx[-1]
    wl_f = np.arange(lo - margin, hi + margin, fine)
    N_air = doas.air_number_density(T_K, P_Pa)
    tau = np.zeros_like(wl_f)
    info = {"gases": [], "N_air": N_air}
    for xs, ppb in gases:
        scd = ppb * 1e-9 * N_air * L_cm
        tau += scd * np.interp(wl_f, xs["wl"], xs["sigma"], left=0.0, right=0.0)
        info["gases"].append((xs["gas"], xs.get("source", ""), ppb, scd))
    tau_r = rayleigh_sigma(wl_f) * N_air * L_cm
    beta = 3.912 / visibility_km * 1e-5 if visibility_km > 0 else 0.0     # per cm at 550 nm
    tau_m = beta * (wl_f / 550.0) ** (-angstrom) * L_cm
    xc = 0.5 * (lo + hi)
    thr = throughput[0] * (1 + throughput[1] * (wl_f - xc) / (hi - lo))
    I0_hr = counts * lamp_spectrum(wl_f)
    I_hr = I0_hr * thr * np.exp(-(tau + tau_r + tau_m))
    pix = np.round(np.arange(lo, hi + pixel / 2, pixel), 6)
    out = []
    for y in (I0_hr, I_hr):
        g, c = slit_mod.convolve_uniform(wl_f, y, sx, sy, lo - 1.0, hi + 1.0, step=fine)
        v = np.interp(pix + shift, g, c)
        out.append(v * (1 + noise * rng.standard_normal(len(v))))
    info.update(tau_rayleigh=(float(tau_r.min()), float(tau_r.max())),
                tau_mie=(float(tau_m.min()), float(tau_m.max())),
                tau_gas_max=float(tau.max()))
    return pix, out[0], out[1], info


def write(path, wl, I0, I, info, slit_desc, T_K, P_Pa, L_cm, noise, shift, visibility_km, seed):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Simulated UV-DOAS spectrum (simulate_spectrum.py): I0 lamp reference, I after the light path\n")
        for g, src, ppb, scd in info["gases"]:
            f.write("# gas %s %.6g ppb   SCD %.6e molecule/cm2   xs %s\n" % (
                g, ppb, scd, os.path.relpath(src, HERE) if os.path.isabs(src) else src))
        f.write("# T %.2f K   P %.1f Pa   L %.1f cm\n" % (T_K, P_Pa, L_cm))
        f.write("# slit %s\n" % slit_desc)
        f.write("# noise %.3g   shift %.4g nm   visibility %.3g km   seed %d\n" % (noise, shift, visibility_km, seed))
        f.write("# Rayleigh tau %.3f-%.3f   Mie tau %.3f-%.3f   peak gas tau %.4f\n" % (
            info["tau_rayleigh"] + info["tau_mie"] + (info["tau_gas_max"],)))
        f.write("# wavelength_nm\tI0\tI\n")
        for a, b, c in zip(wl, I0, I):
            f.write("%.4f\t%.4f\t%.4f\n" % (a, b, c))


def make_example(name, overwrite=True):
    ex = EXAMPLES[name]
    xsm.import_lab_set()
    sx, sy = slit_mod.synthetic_slit(*EXAMPLE_SLIT)
    gases = [(resolve_gas(g), ppb) for g, ppb in ex["gases"].items()]
    p = EXAMPLE_PATH
    if overwrite or not os.path.isfile(ex["out"]):
        wl, I0, I, info = simulate(gases, ex["lo"], ex["hi"], (sx, sy), p["T_K"], p["P_Pa"], p["L_cm"],
                                   noise=ex["noise"])
        write(ex["out"], wl, I0, I, info, slit_mod.describe(*EXAMPLE_SLIT), p["T_K"], p["P_Pa"], p["L_cm"],
              ex["noise"], 0.02, 20.0, 1)
    return ex["out"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gas", nargs=2, action="append", metavar=("NAME_OR_FILE", "PPB"))
    ap.add_argument("--lo", type=float, default=300.0); ap.add_argument("--hi", type=float, default=340.0)
    ap.add_argument("--T", type=float, default=293.15); ap.add_argument("--P", type=float, default=101325.0)
    ap.add_argument("--L-cm", type=float, default=1e5, help="total light path, cm")
    ap.add_argument("--pixel", type=float, default=0.05)
    ap.add_argument("--slit", default="gauss", choices=list(slit_mod.KINDS))
    ap.add_argument("--fwhm", type=float, default=0.5)
    ap.add_argument("--noise", type=float, default=3e-4)
    ap.add_argument("--shift", type=float, default=0.02)
    ap.add_argument("--visibility", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not a.gas:
        for k in EXAMPLES:
            print("wrote", make_example(k))
        return
    params = dict(slit_mod.default_params(a.slit)); params["fwhm"] = a.fwhm
    sx, sy = slit_mod.synthetic_slit(a.slit, params)
    gases = [(resolve_gas(g), float(v)) for g, v in a.gas]
    wl, I0, I, info = simulate(gases, a.lo, a.hi, (sx, sy), a.T, a.P, a.L_cm, a.pixel, a.noise, a.shift,
                               a.visibility, seed=a.seed)
    out = a.out or os.path.join(INPUT, "Sim_%s.txt" % "_".join(x["gas"] for x, _ in gases))
    write(out, wl, I0, I, info, slit_mod.describe(a.slit, params), a.T, a.P, a.L_cm, a.noise, a.shift,
          a.visibility, a.seed)
    print("wrote", out)


if __name__ == "__main__":
    main()
