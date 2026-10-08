# -*- coding: utf-8 -*-
"""Absorption cross sections for UV-DOAS: reading, writing and synthesising them.

    load_xs(path)              a cross-section file -> {"gas", "wl", "sigma", "source", "T_K"}
    save_xs(path, xs, notes)   two columns (wavelength nm, sigma cm2/molecule) with a header
    synthetic_xs(gas)          a synthetic high-resolution cross section (SO2, O3, NO2, HCHO)
    write_synthetic_set(dir)   all synthetic gases as files in Input/CrossSections

WHAT A CROSS SECTION IS HERE
----------------------------
In the UV the gas absorption is a continuous band spectrum, not a list of
lines, so the reference is a tabulated sigma(lambda) - e.g. a file from the
MPI-Mainz UV/VIS Spectral Atlas, HITRAN-UV, or a laboratory measurement - at
a resolution well above the spectrometer's.  load_xs reads any such file with
two numeric columns:

    wavelength (nm)      sigma (cm2/molecule)
    or wavenumber (cm-1) sigma - recognised by values above 1000 and converted

Comment lines (#, ;, !, %) and text headers are skipped; the rows are sorted
by wavelength.

THE SYNTHETIC CROSS SECTIONS
----------------------------
synthetic_xs builds cross sections that look like the real ones - a
vibronic band progression on a smooth continuum, each band with narrow
"rotational" sub-structure finer than a typical UV spectrometer's slit, and
magnitudes of the right order:

    SO2   280-330 nm   bands ~2.5 nm apart, peaks ~1e-18 near 290-300 nm
    O3    300-360 nm   Huggins bands on a continuum falling ~x3.5 per 10 nm
    NO2   290-500 nm   irregular dense bands, 2e-19 (320 nm) to 6e-19 (420 nm)
    HCHO  290-360 nm   isolated bands ~12 nm apart, peaks ~5e-20

They are NOT the real spectra (the band positions are only roughly right):
they exist so the retrieval can be tested against a known answer, and so the
program runs without any download.  For real measurements load real cross
sections (Cross Section -> Add Cross-section File).
"""

import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
XS_DIR = os.path.join(HERE, "Input", "CrossSections")

SYNTHETIC_GASES = ("SO2", "O3", "NO2", "HCHO")
# the default high-resolution grid of the synthetic cross sections
SYN_LO, SYN_HI, SYN_STEP = 270.0, 520.0, 0.01


# =============================================================================
# Files
# =============================================================================
_NUM = re.compile(r"[,;\t ]+")


def _lines(path):
    """The text lines of a file, or of the first member of a .zip."""
    if path.lower().endswith(".zip"):
        import zipfile
        with zipfile.ZipFile(path) as z:
            name = [n for n in z.namelist() if not n.endswith("/")][0]
            return z.read(name).decode("latin-1").splitlines()
    with open(path, "r", encoding="latin-1") as fh:
        return fh.read().splitlines()


def read_columns(path, max_cols=None):
    """(rows as an (N, k) array, comment/header lines).  Lines that do not
    parse as numbers are header.  Old OMA/MPI files carry lone numbers in
    their header ("    1" = number of tracks): one-column rows are dropped
    when most rows have more.  Rows with more columns than the shortest
    data row (e.g. extra temperatures over part of the range) are cut."""
    rows, header = [], []
    for ln in _lines(path):
        s = ln.strip()
        if not s:
            continue
        if s[0] in "#;!%*":
            header.append(s.lstrip("#;!%* ").rstrip()); continue
        parts = [p for p in _NUM.split(s) if p]
        try:
            vals = [float(p.replace("D", "E").replace("d", "e")) for p in parts]
        except ValueError:
            header.append(s); continue
        if max_cols:
            vals = vals[:max_cols]
        rows.append(vals)
    if rows and sum(len(r) >= 2 for r in rows) > len(rows) / 2:
        rows = [r for r in rows if len(r) >= 2]
    if not rows:
        raise ValueError("no numeric rows in %s" % path)
    k = min(len(r) for r in rows)
    return np.array([r[:k] for r in rows], dtype=float), header


def vac_to_air(wl_nm):
    """Vacuum -> air wavelength (Edlén 1966 / Birch & Downs 1994 dispersion
    of standard air), nm.  At 310 nm air is 0.090 nm shorter."""
    s2 = (1e3 / np.asarray(wl_nm, float)) ** 2
    n = 1 + 1e-8 * (8342.54 + 2406147.0 / (130.0 - s2) + 15998.0 / (38.9 - s2))
    return wl_nm / n


def _sigma_column(data, xcol):
    """The first column after the wavelength whose values look like a cross
    section (cm2/molecule, |median| between 1e-26 and 1e-15)."""
    for c in range(xcol + 1, data.shape[1]):
        m = np.nanmedian(np.abs(data[:, c]))
        if 1e-26 < m < 1e-15:
            return c
    return xcol + 1


def gas_from_name(path, header=()):
    """'gas SO2' in the header, else the file name up to its first '_' or '.'."""
    for h in header:
        m = re.match(r"^\s*gas\s*[:=]?\s*(\S+)", h)
        if m:
            return m.group(1)
    return re.split(r"[_.\s-]", os.path.basename(path))[0] or "gas"


def load_xs(path, to_air=None, scale=None):
    """A cross section from a two- or more-column file.

    The wavelength is column 1 (nm, or cm-1 if above 1000); the cross section
    is the first later column with values like cm2/molecule (so the Bremen
    "nm, cm-1, sigma, error" files work).  to_air: convert vacuum wavelengths
    to air - None does it when the header says "vacuum".  scale multiplies
    the cross section (files tabulated in units of 1e-20 cm2 ...)."""
    data, header = read_columns(path)
    if data.shape[1] < 2:
        raise ValueError("%s: need two columns (wavelength, cross section)" % os.path.basename(path))
    x, s = data[:, 0], data[:, _sigma_column(data, 0)]
    if scale:
        s = s * scale
    ok = np.isfinite(x) & np.isfinite(s)
    x, s = x[ok], s[ok]
    if np.nanmedian(x) > 1000.0:                     # wavenumber cm-1 -> nm
        x = 1e7 / x
    # a file that states its wavelength scale ("wavelength scale: air ...") is
    # taken as it says; otherwise a header that mentions "vacuum" means vacuum
    stated = any(re.match(r"\s*wavelength scale", h, re.I) for h in header)
    vacuum = not stated and any(re.search(r"\bvacuum\b", h, re.I) for h in header)
    if to_air is None:
        to_air = vacuum
    if to_air:
        x = vac_to_air(x)
    o = np.argsort(x)
    x, s = x[o], s[o]
    keep = np.concatenate(([True], np.diff(x) > 0))  # no repeated wavelengths
    T = None
    for h in header:
        m = re.search(r"\bT\s*[:=]?\s*([\d.]+)\s*K", h)
        if m:
            T = float(m.group(1)); break
    return {"gas": gas_from_name(path, header), "wl": x[keep], "sigma": s[keep],
            "source": path, "T_K": T, "synthetic": any("synthetic" in h.lower() for h in header),
            "converted_to_air": bool(to_air), "header": header}


def save_xs(path, xs, notes=()):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# gas %s\n" % xs["gas"])
        for n in notes:
            f.write("# %s\n" % n)
        f.write("# wavelength_nm\tsigma_cm2_per_molecule\n")
        for a, b in zip(xs["wl"], xs["sigma"]):
            f.write("%.5f\t%.6e\n" % (a, b))


def xs_range(xs):
    return float(xs["wl"][0]), float(xs["wl"][-1])


# =============================================================================
# Laboratory cross sections (Input/modataXs)
# =============================================================================
# The measured cross sections chosen from the laboratory collection in
# Input/modataXs, written as clean two-column files (air wavelength, nm;
# cm2/molecule) to Input/CrossSections by import_lab_set().  The wavelength
# scale of each was checked by cross-correlating it with other data sets of
# the same gas (README: "Laboratory cross sections").
LAB_DIR = os.path.join(HERE, "Input", "modataXs")
LAB_SET = (
    {"gas": "SO2", "file": "so2vandaele", "out": "SO2_Vandaele_295K.xs", "T": 295, "to_air": False,
     "range": (260.0, 327.7),
     "notes": ("Vandaele, Simon, Guilmot, Carleer, Colin, J. Geophys. Res. 99, 25599 (1994)",
               "Fourier-transform spectrometer, room temperature (~295 K), step 0.0064 nm",
               "wavelength scale air: matches Brassington 1981 and the MPI compilation to <0.01 nm")},
    {"gas": "O3", "file": "o3_293h.dat", "out": "O3_Bremen_293K.xs", "T": 293, "to_air": True,
     "range": (240.0, 550.0),
     "notes": ("Voigt, Orphal, Bogumil, Burrows, J. Photochem. Photobiol. A 143, 1 (2001); IUP Bremen",
               "Bruker IFS 120HR FTS, 5 cm-1 resolution, 293 K, 1000 mbar",
               "wavelength scale air: converted from the file's vacuum wavelengths (Edlen)")},
    {"gas": "NO2", "file": os.path.join("bremen", "no2_293h.zip"), "out": "NO2_Bremen_293K.xs", "T": 293,
     "to_air": True, "range": (250.0, 550.0),
     "notes": ("Voigt, Orphal, Burrows, J. Photochem. Photobiol. A 149, 1 (2002); IUP Bremen",
               "Bruker IFS 120HR FTS, 1 cm-1 (<23000 cm-1: 0.5 cm-1) resolution, 293 K, 1000 mbar",
               "wavelength scale air: converted from vacuum; then matches Merienne 1995 (Reims) to 0.02 nm")},
    {"gas": "HCHO", "file": "hchocal.dat", "out": "HCHO_Cantrell_296K.xs", "T": 296, "to_air": False,
     "range": (300.3, 385.8),
     "notes": ("Cantrell, Davidson, McDaniel, Shetter, Calvert, J. Phys. Chem. 94, 3902 (1990)",
               "Bomem DA3.01 FTS, 296 K, step 0.0056 nm",
               "wavelength scale air (as given); vs Meller 2000 / Bass it sits +0.03..+0.07 nm - check")},
)


def lab_path(gas, folder=None):
    for e in LAB_SET:
        if e["gas"] == gas.upper():
            return os.path.join(folder or XS_DIR, e["out"])
    return None


def import_lab_set(folder=None, overwrite=False):
    """The LAB_SET cross sections from Input/modataXs as clean files in
    Input/CrossSections; returns the paths written or already there."""
    folder = folder or XS_DIR
    os.makedirs(folder, exist_ok=True)
    out = []
    for e in LAB_SET:
        p = os.path.join(folder, e["out"])
        src = os.path.join(LAB_DIR, e["file"])
        if not os.path.isfile(p) or overwrite:
            if not os.path.isfile(src):
                continue
            xs = load_xs(src, to_air=e["to_air"])
            m = (xs["wl"] >= e["range"][0]) & (xs["wl"] <= e["range"][1])
            xs["wl"], xs["sigma"], xs["gas"] = xs["wl"][m], xs["sigma"][m], e["gas"]
            save_xs(p, xs, notes=e["notes"] + ("T %g K" % e["T"],
                                               "source Input/modataXs/%s" % e["file"].replace(os.sep, "/")))
        out.append(p)
    return out


def default_path(gas):
    """The laboratory cross section of gas if there is one, else the synthetic one."""
    p = lab_path(gas)
    if p and not os.path.isfile(p):
        import_lab_set()
    return p if p and os.path.isfile(p) else synthetic_path(gas)


# =============================================================================
# Synthetic cross sections
# =============================================================================
def _lorentz(x, c, hw):
    return 1.0 / (1.0 + ((x - c) / hw) ** 2)


def _band(wl, centre, width, rng, n_sub=4, sub_width=0.08):
    """One vibronic band: a broad Lorentz plus a few narrow sub-bands (the
    unresolved rotational structure), peak about 1."""
    y = 0.55 * _lorentz(wl, centre, width)
    offs = rng.uniform(-0.9, 0.6, n_sub) * width
    amps = rng.uniform(0.15, 0.45, n_sub)
    for o, a in zip(offs, amps):
        y += a * _lorentz(wl, centre + o, sub_width * rng.uniform(0.7, 1.5))
    return y


def _progression(wl, nu0, dnu, n, rng, width_nm, amp_fn, n_sub=4, sub_width=0.08, jitter=0.0):
    y = np.zeros_like(wl)
    for k in range(n):
        nu = nu0 + k * dnu + (rng.normal(0, jitter) if jitter else 0.0)
        c = 1e7 / nu
        if c < wl[0] - 5 or c > wl[-1] + 5:
            continue
        y += amp_fn(c) * _band(wl, c, width_nm, rng, n_sub, sub_width)
    return y


def synthetic_xs(gas, lo=SYN_LO, hi=SYN_HI, step=SYN_STEP):
    """A synthetic high-resolution cross section of `gas` (cm2/molecule)."""
    g = gas.upper()
    wl = np.round(np.arange(lo, hi + step / 2, step), 6)
    rng = np.random.default_rng({"SO2": 11, "O3": 22, "NO2": 33, "HCHO": 44}.get(g, 99))
    if g == "SO2":
        nu = 1e7 / wl
        env = 1.3e-18 * np.exp(-((nu - 34800.0) / 2300.0) ** 2)
        cont = 0.40 * env
        bands = _progression(wl, 30650.0, 235.0, 26, rng, 0.45,
                             lambda c: 1.0 * 1.3e-18 * np.exp(-((1e7 / c - 34800.0) / 2300.0) ** 2),
                             n_sub=5, sub_width=0.07, jitter=12.0)
        sigma = cont + 0.75 * bands
    elif g == "O3":
        cont = 1.1e-19 * np.exp(-(wl - 310.0) / 8.0)
        cont = np.minimum(cont, 1.15e-17 * np.exp(-((wl - 255.0) / 22.0) ** 2) + 1e-19 * np.exp(-(wl - 310.0) / 8.0))
        chap = 4.5e-21 * np.exp(-((wl - 600.0) / 90.0) ** 2)          # tail of the Chappuis band
        bands = _progression(wl, 28900.0, 300.0, 22, rng, 0.9,
                             lambda c: 0.45 * 1.1e-19 * np.exp(-(c - 310.0) / 8.0),
                             n_sub=3, sub_width=0.25, jitter=20.0)
        sigma = cont + bands + chap
    elif g == "NO2":
        env = 6.2e-19 * np.exp(-((wl - 425.0) / 95.0) ** 2) + 1.0e-20
        y = np.zeros_like(wl)
        c = 288.0
        while c < 515.0:
            w = rng.uniform(0.35, 1.1)
            y += rng.uniform(0.15, 0.45) * _band(wl, c, w, rng, n_sub=3, sub_width=0.06)
            c += rng.uniform(1.2, 3.2)
        sigma = env * (0.75 + 0.55 * y / (1.0 + 0.6 * y))
    elif g == "HCHO":
        cont = 3.0e-21 * np.exp(-((wl - 300.0) / 40.0) ** 2)
        y = np.zeros_like(wl)
        for c, a in ((303.6, 0.55), (308.9, 0.25), (316.2, 0.85), (321.0, 0.30), (326.1, 1.00),
                     (330.9, 0.35), (339.0, 0.90), (343.2, 0.30), (352.9, 0.55), (357.5, 0.15)):
            y += a * _band(wl, c, 0.45, rng, n_sub=4, sub_width=0.06)
        sigma = cont + 5.5e-20 * y
    else:
        raise ValueError("no synthetic cross section for %s (have %s)" % (gas, ", ".join(SYNTHETIC_GASES)))
    return {"gas": g, "wl": wl, "sigma": np.maximum(sigma, 0.0), "source": "synthetic", "T_K": 293.0,
            "synthetic": True}


def synthetic_path(gas, folder=None):
    return os.path.join(folder or XS_DIR, "%s_synthetic_293K.xs" % gas.upper())


def write_synthetic_set(folder=None, gases=SYNTHETIC_GASES, overwrite=False):
    """The synthetic cross sections as files; returns their paths."""
    folder = folder or XS_DIR
    os.makedirs(folder, exist_ok=True)
    out = []
    for g in gases:
        p = synthetic_path(g, folder)
        if overwrite or not os.path.isfile(p):
            save_xs(p, synthetic_xs(g), notes=(
                "SYNTHETIC cross section (cross_section.synthetic_xs) - for testing, not real data",
                "T 293 K   step %.3g nm   %.0f-%.0f nm" % (SYN_STEP, SYN_LO, SYN_HI)))
        out.append(p)
    return out


_CATALOG = {}


def catalog(folder=None):
    """{path: row} of the database that build_xs_database.py wrote to the
    gas subfolders of Input/CrossSections (catalog.csv), {} without one.
    A row has gas, T_K, lo_nm, hi_nm, kind, unit, reference, source, quality ..."""
    import csv
    folder = folder or XS_DIR
    p = os.path.join(folder, "catalog.csv")
    if not os.path.isfile(p):
        return {}
    key = (p, os.path.getmtime(p))
    if key not in _CATALOG:
        with open(p, newline="", encoding="utf-8") as f:
            _CATALOG[key] = {os.path.join(folder, *r["file"].split("/")): r for r in csv.DictReader(f)}
    return _CATALOG[key]


def file_temperature(path):
    """T (K) of a cross-section file from the catalog or its name ('_293K'), else None."""
    r = catalog(os.path.dirname(os.path.dirname(path))).get(path) or catalog().get(path)
    if r and r.get("T_K"):
        return float(r["T_K"])
    m = re.search(r"_(\d+(?:\.\d+)?)K", os.path.basename(path))
    return float(m.group(1)) if m else None


def scan_folder(folder=None):
    """{path: (gas, lo, hi)} for the cross-section files in folder - the files
    in the folder itself (read) and the database in its gas subfolders (from
    catalog.csv, without opening the files)."""
    folder = folder or XS_DIR
    out = {}
    if not os.path.isdir(folder):
        return out
    for p, r in catalog(folder).items():
        out[p] = (r["gas"], float(r["lo_nm"]), float(r["hi_nm"]))
    for n in sorted(os.listdir(folder)):
        p = os.path.join(folder, n)
        if not os.path.isfile(p) or os.path.splitext(n)[1].lower() not in (".xs", ".txt", ".dat", ".csv", ".xsc"):
            continue
        try:
            xs = load_xs(p)
            out[p] = (xs["gas"], xs["wl"][0], xs["wl"][-1])
        except Exception:
            pass
    return out


if __name__ == "__main__":
    for p in import_lab_set(overwrite=True) + write_synthetic_set(overwrite=True):
        xs = load_xs(p)
        print("%-5s %s  %d points  max %.2e" % (xs["gas"], p, len(xs["wl"]), xs["sigma"].max()))
