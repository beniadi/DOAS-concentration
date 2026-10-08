# -*- coding: utf-8 -*-
"""DOAS GUI - trace-gas concentration from a UV spectrum by DOAS fitting.

The look (stylesheet, buttons, group boxes, tables, plots) comes from
gui_common.py, so this window is one family with the FTIR Concentration GUI.
The method is UV-DOAS: absorption cross sections convolved with the
spectrometer's slit function are fitted, together with a polynomial for the
broadband extinction, to the optical depth ln(I0/I) by LINEAR least squares;
the slant column density of every gas comes out of one solve (doas.fit_doas).

    python DOAS_GUI.py
    python DOAS_GUI.py spectrum.txt      # open a spectrum at start-up
    python DOAS_GUI.py --no-example      # start empty
    python DOAS_GUI.py --selftest        # both worked examples, no screen
    python DOAS_GUI.py --selftest --save --montecarlo
                                         # ... plus plots/reports in Results/ and 50 noise seeds

At start-up the window opens the single-gas example - Input/SO2_simulated_15ppb.txt
(simulate_spectrum.py, 15 ppb SO2 over a 1 km light path) with the synthetic
SO2 cross section and a Gaussian slit of 0.5 nm FWHM - and retrieves it, so
there is always a known answer to compare against.  Help -> Load Multi-gas
Example does the same for SO2 + O3 + NO2 + HCHO.

WHAT IT DOES
------------
1. Opens a spectrum: three columns (wavelength nm, I0, I), or two columns
   (wavelength, I) with the lamp/reference I0 from a second file, or two
   columns of optical depth ln(I0/I).
2. Takes the cross sections of the gases to fit (Cross Section -> Select
   Gases): files in Input/CrossSections or anywhere else - real ones (e.g.
   MPI-Mainz UV/VIS Spectral Atlas) or the synthetic set.
3. Takes the slit function: measured (Slit -> Load Slit File) or synthetic
   (Gaussian, asymmetric Gaussian, super-Gaussian, boxcar x Gaussian, Voigt).
4. With the light path, temperature and pressure, fits
       ln(I0/I) = sum_i SCD_i * (sigma_i (*) slit)(lambda + shift) + P(lambda)
   in the fit window and reports every gas's SCD, path-averaged number
   density and mixing ratio with its uncertainty, the shift, the polynomial,
   the residual and the correlation between gases.
5. Simulate -> Synthetic Spectrum makes a new test spectrum with any gases,
   path, slit, noise, aerosol and calibration shift.
"""

import os, sys, re
from datetime import datetime

import numpy as np

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QFileDialog, QGridLayout, QMenuBar, QAction, QMessageBox, QSpinBox, QComboBox,
    QCheckBox, QTabWidget, QTextEdit, QTableWidget, QDialog, QDialogButtonBox, QFormLayout,
    QDoubleSpinBox, QLineEdit, QHeaderView)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

import matplotlib
from matplotlib.figure import Figure

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import cross_section as xsm                                 # noqa: E402
import slit as slit_mod                                     # noqa: E402
import doas                                                 # noqa: E402
import simulate_spectrum as sim                             # noqa: E402
from gui_common import (STYLESHEET, JobWorker, style_table, table_cell, fmt,     # noqa: E402
                        config_dir, results_dir, load_json, save_json, INK_MUTED,
                        PubCanvas, pub_axes, pub_legend, pub_figure, pub_savefig,
                        PUB_FONT, PUB_FIT_LW)

APP_TITLE = "UV-DOAS Concentration 1.0"
WIN_W, WIN_H = 1440, 940
PRESSURE_UNITS = {"atm": 101325.0, "hPa": 100.0, "mbar": 100.0, "Torr": 133.322368, "kPa": 1000.0, "Pa": 1.0}
LENGTH_UNITS = {"cm": 1.0, "m": 100.0, "km": 1e5}
VIEWS = (("od", "Optical depth"), ("cascade", "Differential cascade"), ("gas", "Differential per gas"),
         ("int", "Intensity"))
STACKED = ("gas", "cascade")          # views of one panel per gas, sized by the number of gases


def settings_path(): return os.path.join(config_dir(), "doas_settings.json")


DEFAULTS = {
    "last_open_dir": None, "last_save_dir": None,
    # the light path
    "T_K": 293.15, "P_value": 1013.25, "P_unit": "hPa", "L_value": 1.0, "L_unit": "km",
    # the fit (doas.DEFAULTS)
    "poly_order": 3, "fit_shift": True, "max_shift": 0.15, "fit_squeeze": False, "max_squeeze": 0.002,
    "oversample": 10, "unit": "ppb",
    # the slit
    "slit_kind": "gauss", "slit_params": {"fwhm": 0.5},
    # the gases last fitted
    "xs_paths": [],
    # synthetic spectrum dialog
    "sim_lo": 300.0, "sim_hi": 345.0, "sim_pixel": 0.05, "sim_noise": 3e-4, "sim_shift": 0.02,
    "sim_visibility": 20.0, "sim_seed": 1, "sim_ppb": {"SO2": 10.0, "O3": 50.0, "NO2": 20.0, "HCHO": 8.0},
    # start-up
    "example_on_start": True,
}


def load_settings():
    s = load_json(settings_path(), DEFAULTS)
    if not s["last_open_dir"]: s["last_open_dir"] = os.path.join(HERE, "Input")
    if not s["last_save_dir"]: s["last_save_dir"] = results_dir()
    return s


def fit_options(s):
    return {"poly_order": int(s["poly_order"]), "fit_shift": bool(s["fit_shift"]),
            "max_shift": float(s["max_shift"]), "fit_squeeze": bool(s["fit_squeeze"]),
            "max_squeeze": float(s["max_squeeze"]), "unit": s["unit"]}


_GAS_RE = re.compile(r"^gas\s+(\S+)\s+([-+\d.eE]+)\s*ppb")
_PATH_RE = re.compile(r"T\s+([-+\d.eE]+)\s*K\s+P\s+([-+\d.eE]+)\s*Pa\s+L\s+([-+\d.eE]+)\s*cm")


def read_sim_header(header):
    """({gas: ppb}, (T_K, P_Pa, L_cm) or None) from simulate_spectrum.py's header lines."""
    gases, cond = {}, None
    for h in header:
        m = _GAS_RE.match(h)
        if m:
            gases[m.group(1)] = float(m.group(2))
        m = _PATH_RE.search(h)
        if m:
            cond = tuple(float(g) for g in m.groups())
    return gases, cond


def retrieve(wl, od, xs_list, slit_xy, region, T_K, P_Pa, L_cm, s):
    refs = doas.make_references(xs_list, slit_xy[0], slit_xy[1], region[0], region[1], int(s["oversample"]))
    return doas.fit_doas(wl, od, refs, region, T_K=T_K, P_Pa=P_Pa, L_cm=L_cm, **fit_options(s))


class Job:
    """What JobWorker needs to name a job in its error messages."""
    def __init__(self, name):
        self.name = name


# =============================================================================
# Plot
# =============================================================================
_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def chem(name):
    """'SO2' -> 'SO₂' (Unicode subscripts, so it works in every font and format)."""
    return "".join(c.translate(_SUB) if c.isdigit() and i > 0 else c for i, c in enumerate(name))


def _empty(fig, text):
    ax = fig.add_subplot(111); pub_axes(ax, grid=False)
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=INK_MUTED,
            fontsize=PUB_FONT + 1)
    ax.set_xticks([]); ax.set_yticks([])


def build_main_figure(fig, view, data, region, r, show, keep_xlim=None):
    """data: {"wl", "od", "I", "I0"} (any may be None).  view: od | cascade | gas | int.
    Drawn in the gui_common publication style (Camera_calibration_GUI)."""
    fig.clear()
    wl = data.get("wl")
    if wl is None:
        _empty(fig, "File → Open Spectrum"); return
    if view == "gas":
        _per_gas_figure(fig, r, show); return
    if view == "cascade":
        _cascade_figure(fig, r, show); return
    with_res = bool(show.get("residual")) and r is not None and view == "od"
    if with_res:
        gs = fig.add_gridspec(2, 1, height_ratios=(3, 1))
        ax = fig.add_subplot(gs[0]); axr = fig.add_subplot(gs[1], sharex=ax)
    else:
        ax = fig.add_subplot(111); axr = None
    if region:
        ax.axvspan(region[0], region[1], color="C0", alpha=0.10, lw=0, label="Fit window")
    if view == "int":
        if data.get("I0") is not None:
            ax.plot(wl, data["I0"], "-", color="C0", label="I₀ (reference)")
        if data.get("I") is not None:
            ax.plot(wl, data["I"], "-", color="C1", label="I (measured)")
        if data.get("I") is None and data.get("I0") is None:
            ax.text(0.5, 0.5, "The file holds optical depth only", transform=ax.transAxes, ha="center",
                    color=INK_MUTED, fontsize=PUB_FONT + 1)
        ylabel = "Intensity [counts]"
    else:
        if data.get("od") is not None and show.get("data", True):
            ax.plot(wl, data["od"], "-", color="C0", label="Measured ln(I₀/I)")
        if r is not None:
            if show.get("fit", True):
                ax.plot(r["x"], r["fit"], "-", color="C1", lw=PUB_FIT_LW, label="DOAS model")
            if show.get("poly"):
                ax.plot(r["x"], r["poly"], "--", color="C2", lw=PUB_FIT_LW,
                        label="Polynomial (order %d)" % r["poly_order"])
        ylabel = "Optical depth [-]"
    pub_axes(ax, None if axr is not None else "Wavelength [nm]", ylabel)
    pub_legend(ax, loc="upper right" if view == "int" else "upper left")
    if axr is not None:
        axr.plot(r["x"], r["residual"], "-", color="C0", label="Residual   RMS %.3g" % r["rms"])
        pub_axes(axr, "Wavelength [nm]", "Residual [-]")
        pub_legend(axr)
        ax.tick_params(labelbottom=False)
    if keep_xlim:
        ax.set_xlim(*keep_xlim)
    elif region:
        pad = 0.05 * (region[1] - region[0])
        ax.set_xlim(max(wl[0], region[0] - pad), min(wl[-1], region[1] + pad))
    else:
        ax.set_xlim(wl[0], wl[-1])
    _autoscale_y(ax)
    if axr is not None:
        _autoscale_y(axr)


def _per_gas_figure(fig, r, show):
    """One panel per gas: measured differential OD (gas + residual) and the
    fitted SCD x sigma', as in QDOAS / WinDOAS; the residual at the bottom."""
    if r is None:
        _empty(fig, "Retrieve first: this view shows each gas's fitted differential absorption"); return
    gases = r["gases"]
    axes = fig.subplots(len(gases) + 1, 1, sharex=True, squeeze=False)[:, 0]
    for ax, g in zip(axes, gases):
        if show.get("data", True):
            ax.plot(r["x"], r["diff_meas"][g], "-", color="C0", label="%s measured (SCD·σ′ + residual)" % chem(g))
        if show.get("fit", True):
            ax.plot(r["x"], r["components"][g], "-", color="C1", lw=PUB_FIT_LW,
                    label="%s fit: SCD = %s molec/cm², %s %s" % (
                        chem(g), fmt(r["scd"][g], r["scd_err"][g], 4), fmt(r["mix"][g], r["mix_err"][g], 4),
                        r["unit"]))
        pub_axes(ax, None, "ΔOD [-]")
        pub_legend(ax)
        ax.tick_params(labelbottom=False)
    ax = axes[-1]
    ax.plot(r["x"], r["residual"], "-", color="C0", label="Residual   RMS %.3g" % r["rms"])
    pub_axes(ax, "Wavelength [nm]", "ΔOD [-]")
    pub_legend(ax)
    ax.set_xlim(r["x"][0], r["x"][-1])
    for ax in axes:
        _autoscale_y(ax)


cascade_steps = doas.cascade_steps


def _cascade_figure(fig, r, show):
    """The total differential OD with every gas subtracted in turn, one panel
    per step, down to the residual.  All panels share the y scale, so what
    each gas contributes - and how small the residual is - shows directly."""
    if r is None:
        _empty(fig, "Retrieve first: this view starts from the total differential optical depth\n"
                    "and subtracts one gas at a time down to the residual"); return
    steps = cascade_steps(r)
    gases = r["gases"]
    axes = fig.subplots(len(steps), 1, sharex=True, sharey=True, squeeze=False)[:, 0]
    for k, (ax, (_t, data, model, removed)) in enumerate(zip(axes, steps)):
        last = k == len(steps) - 1
        rms = float(np.sqrt(np.mean(data ** 2)))
        rest = gases[len(removed):]
        if k == 0:
            name = "Total differential OD = ln(I₀/I) − polynomial"
        elif last:
            name = "After − %s: residual" % chem(removed[-1])
        else:
            name = "After − %s" % chem(removed[-1])
        if show.get("data", True):
            ax.plot(r["x"], data, "-", color="C0", label="%s   RMS %.3g" % (name, rms))
        if show.get("fit", True) and not last:
            ax.plot(r["x"], model, "-", color="C1", lw=PUB_FIT_LW,
                    label="Fit: %s" % " + ".join(chem(g) for g in rest))
        pub_axes(ax, None if not last else "Wavelength [nm]", "ΔOD [-]")
        pub_legend(ax)
        if not last:
            ax.tick_params(labelbottom=False)
    axes[-1].set_xlim(r["x"][0], r["x"][-1])
    # one y scale for all panels, with room at the top for the largest legend
    a = max(float(np.max(np.abs(s[1]))) for s in steps)
    top = max(_legend_fraction(ax) for ax in axes)
    axes[0].set_ylim(-1.1 * a, -1.1 * a + 2.2 * a / max(1.0 - top, 0.35))


def export_figure(view, r, data, region, show, px_w=1000):
    """The view as a figure for a file: px_w pixels wide at 120 dpi (the
    Camera_calibration_GUI size), 130 px per panel for the stacked views."""
    n = (len(r["gases"]) + 1) if (r is not None and view in STACKED) else 0
    fig = pub_figure(px_w, max(650, 130 * n + 60) if n else 650)
    build_main_figure(fig, view, data, region, r, show)
    return fig


def mouse_navigation(pc, on_reset):
    """Zoom and pan a PlotCanvas without a toolbar: scroll zooms the x axis
    about the cursor, a left-button drag pans it, a double-click calls
    on_reset (redraw at the full range).  Axes with their own y scale follow
    the visible data; shared-y stacks keep their common scale."""
    c, st = pc.canvas, {}

    def rescale():
        for ax in pc.fig.axes:
            if ax.get_yscale() == "linear" and len(ax.get_shared_y_axes().get_siblings(ax)) == 1:
                _autoscale_y(ax)
        c.draw_idle()

    def scroll(ev):
        ax = ev.inaxes
        if ax is None or ev.xdata is None:
            return
        f = 0.8 if ev.button == "up" else 1.25
        lo, hi = ax.get_xlim()
        ax.set_xlim(ev.xdata - (ev.xdata - lo) * f, ev.xdata + (hi - ev.xdata) * f)
        rescale()

    def press(ev):
        if ev.inaxes is None:
            return
        if ev.dblclick:
            st.clear(); on_reset(); return
        if ev.button == 1:
            st.update(ax=ev.inaxes, px=ev.x, lim=ev.inaxes.get_xlim())

    def motion(ev):
        if not st or ev.x is None:
            return
        lo, hi = st["lim"]
        d = (ev.x - st["px"]) * (hi - lo) / max(st["ax"].bbox.width, 1.0)
        st["ax"].set_xlim(lo - d, hi - d)
        rescale()

    c.mpl_connect("scroll_event", scroll)
    c.mpl_connect("button_press_event", press)
    c.mpl_connect("motion_notify_event", motion)
    c.mpl_connect("button_release_event", lambda _e: st.clear())
    pc.setToolTip("Scroll: zoom  ·  drag: pan  ·  double-click: full range")


def _autoscale_y(ax):
    """y limits from what is visible in the current x range."""
    lo, hi = ax.get_xlim()
    ys = []
    for ln in ax.get_lines():
        xd, yd = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
        if len(xd) < 2:
            continue
        m = (xd >= lo) & (xd <= hi)
        if m.any():
            ys.append(yd[m])
    if ys:
        y = np.concatenate(ys); y = y[np.isfinite(y)]
        if len(y):
            a, b = float(y.min()), float(y.max())
            pad = 0.05 * (b - a) if b > a else 0.05
            top = _legend_fraction(ax)          # keep the data clear of the legend at the top
            span = (b - a + 2 * pad) / max(1.0 - top, 0.35)
            ax.set_ylim(a - pad, a - pad + span)


def _legend_fraction(ax):
    """Fraction of the axes height the legend takes, plus a small gap (0 without one)."""
    leg = ax.get_legend()
    if leg is None:
        return 0.0
    try:
        h = leg.get_window_extent(ax.figure.canvas.get_renderer()).height / max(ax.bbox.height, 1.0)
    except Exception:
        h = 0.06 + 0.11 * len(leg.get_texts())
    return min(h + 0.04, 0.6)


# =============================================================================
# Dialogs
# =============================================================================
def _form_page(title):
    w = QGroupBox(title)
    f = QFormLayout(w); f.setContentsMargins(16, 16, 16, 14); f.setSpacing(10)
    f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    return w, f


def _dspin(lo, hi, step, decimals, value, tip=None):
    s = QDoubleSpinBox(); s.setRange(lo, hi); s.setSingleStep(step); s.setDecimals(decimals); s.setValue(value)
    if tip: s.setToolTip(tip)
    return s


class SettingsDialog(QDialog):
    """The retrieval settings.  Edits a copy and hands it back on OK, so
    Cancel really cancels."""

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Retrieval Settings")
        self.setMinimumWidth(560)
        self._s = dict(settings)
        self._w = {}
        v = QVBoxLayout(self); v.setContentsMargins(14, 14, 14, 14); v.setSpacing(10)

        g, f = _form_page("DOAS fit")
        f.addRow("Polynomial order", self._ispin("poly_order", 0, 10,
                 "Legendre polynomial for the broadband part: Rayleigh, Mie, lamp and\n"
                 "throughput changes, the broad part of the cross sections. 3-5 is usual;\n"
                 "too high an order starts to take up the gas bands too."))
        row = QHBoxLayout(); row.addWidget(self._check("fit_shift", "Fit"))
        row.addWidget(QLabel("max |shift|")); row.addWidget(self._dspin("max_shift", 0.001, 2, 0.01, 4)); row.addStretch(1)
        f.addRow("Wavelength shift (nm)", row)
        row = QHBoxLayout(); row.addWidget(self._check("fit_squeeze", "Fit"))
        row.addWidget(QLabel("max |squeeze|")); row.addWidget(self._dspin("max_squeeze", 1e-5, 0.05, 0.0005, 5))
        row.addStretch(1)
        f.addRow("Squeeze (stretch)", row)
        f.addRow("Convolution grid (points per FWHM)", self._ispin("oversample", 2, 100,
                 "Step of the uniform grid on which σ ⊛ slit is computed."))
        f.addRow("Concentration unit", self._combo("unit", (("ppb", "ppb"), ("ppm", "ppm"), ("ppt", "ppt"))))
        v.addWidget(g)
        g, f = _form_page("Start-up")
        f.addRow(self._check("example_on_start", "Load and retrieve the worked example at start-up"))
        v.addWidget(g)
        v.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        bb.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self._restore)
        v.addWidget(bb)

    def _dspin(self, key, lo, hi, step, decimals, tip=None):
        s = _dspin(lo, hi, step, decimals, float(self._s[key]), tip)
        self._w[key] = (s, float); return s

    def _ispin(self, key, lo, hi, tip=None):
        s = QSpinBox(); s.setRange(lo, hi); s.setValue(int(self._s[key]))
        if tip: s.setToolTip(tip)
        self._w[key] = (s, int); return s

    def _check(self, key, text, tip=None):
        c = QCheckBox(text); c.setChecked(bool(self._s[key]))
        if tip: c.setToolTip(tip)
        self._w[key] = (c, bool); return c

    def _combo(self, key, items, tip=None):
        c = QComboBox()
        for val, lab in items: c.addItem(lab, val)
        c.setCurrentIndex(max(0, c.findData(self._s[key])))
        if tip: c.setToolTip(tip)
        self._w[key] = (c, str); return c

    def _restore(self):
        for k, (wd, cast) in self._w.items():
            val = DEFAULTS[k]
            if isinstance(wd, QComboBox): wd.setCurrentIndex(max(0, wd.findData(val)))
            elif isinstance(wd, QCheckBox): wd.setChecked(bool(val))
            else: wd.setValue(cast(val))

    def values(self):
        s = dict(self._s)
        for k, (wd, cast) in self._w.items():
            if isinstance(wd, QComboBox): s[k] = wd.currentData()
            elif isinstance(wd, QCheckBox): s[k] = wd.isChecked()
            else: s[k] = cast(wd.value())
        return s


class SlitDialog(QDialog):
    """A synthetic slit function: shape, parameters and a live preview."""

    def __init__(self, kind, params, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Synthetic Slit Function")
        self.resize(640, 560)
        self._params = dict(params)
        v = QVBoxLayout(self); v.setContentsMargins(14, 14, 14, 14); v.setSpacing(10)
        g, self._form = _form_page("Shape")
        self.combo = QComboBox()
        for k, lab in slit_mod.KINDS.items(): self.combo.addItem(lab, k)
        self.combo.setCurrentIndex(max(0, self.combo.findData(kind)))
        self.combo.currentIndexChanged.connect(self._rebuild)
        self._form.addRow("Kind", self.combo)
        v.addWidget(g)
        self.plot = PubCanvas(min_height=240)
        v.addWidget(self.plot, 1)
        self.lbl = QLabel("—"); self.lbl.setProperty("muted", True); v.addWidget(self.lbl)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject); v.addWidget(bb)
        self.setStyleSheet(STYLESHEET)
        self._spins = {}
        self._rebuild()

    def _rebuild(self):
        while self._form.rowCount() > 1:
            self._form.removeRow(1)
        kind = self.combo.currentData()
        self._spins = {}
        for key, lab, dflt, lo, hi, step, dec, tip in slit_mod.PARAMS[kind]:
            sp = _dspin(lo, hi, step, dec, float(self._params.get(key, dflt)), tip)
            sp.valueChanged.connect(self._preview)
            self._form.addRow(lab, sp); self._spins[key] = sp
        self._preview()

    def values(self):
        return self.combo.currentData(), {k: sp.value() for k, sp in self._spins.items()}

    def _preview(self):
        kind, p = self.values()
        self._params.update(p)
        fig = self.plot.fig; fig.clear(); ax = fig.add_subplot(111)
        try:
            x, y = slit_mod.synthetic_slit(kind, p)
        except Exception as e:
            pub_axes(ax); self.lbl.setText("%s" % e); self.plot.draw(); return
        st = slit_mod.slit_stats(x, y)
        ax.plot(x, y / y.max(), "-", color="C0", label="%s, FWHM %.3f nm" % (slit_mod.KINDS[kind], st["fwhm"]))
        ax.axhline(0.5, color="C7", lw=PUB_FIT_LW, ls="--", label="Half maximum")
        ax.set_xlim(-2.5 * st["fwhm"], 2.5 * st["fwhm"])
        pub_axes(ax, "Offset [nm]", "Slit (peak = 1) [-]"); pub_legend(ax)
        self.plot.draw()
        self.lbl.setText("FWHM %.4f nm   ·   left %.4f / right %.4f nm   ·   centroid %+.4f nm   ·   asymmetry %+.3f"
                         % (st["fwhm"], st["left"], st["right"], st["centroid"], st["asymmetry"]))


class GasDialog(QDialog):
    """The gases to fit: every cross-section file found - the recommended set
    and the whole database in Input/CrossSections - ticked or not."""

    COLS = ["Fit", "Gas", "T (K)", "Range (nm)", "Kind", "Quality", "File"]

    def __init__(self, files, chosen, region, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Gases")
        self.resize(1100, 620)
        self.setStyleSheet(STYLESHEET)
        self._files = dict(files)            # path -> (gas, lo, hi)
        self._chosen = set(chosen)
        self._region = region
        v = QVBoxLayout(self); v.setContentsMargins(14, 14, 14, 14); v.setSpacing(8)
        hint = QLabel("Every ticked file is fitted with its own slant column density. Leave out a gas that "
                      "absorbs in the window and the others take up its structure; add one that is not there "
                      "and its SCD should come out near zero. Rows in grey do not cover the fit window. "
                      "The database files (Input/CrossSections/&lt;gas&gt;/, built by build_xs_database.py) show "
                      "their reference, source and any quality remark in the tooltip.")
        hint.setProperty("muted", True); hint.setWordWrap(True); v.addWidget(hint)
        row = QHBoxLayout()
        cap = QLabel("Search"); cap.setProperty("muted", True); row.addWidget(cap)
        self.ed_filter = QLineEdit()
        self.ed_filter.setPlaceholderText("gas, temperature, author, file ...  e.g.  NO2 293  or  Bremen")
        self.ed_filter.textChanged.connect(self._apply_filter); row.addWidget(self.ed_filter, 1)
        self.chk_cover = QCheckBox("Only files covering the fit window")
        self.chk_cover.toggled.connect(self._apply_filter); row.addWidget(self.chk_cover)
        self.chk_ticked = QCheckBox("Ticked only")
        self.chk_ticked.toggled.connect(self._apply_filter); row.addWidget(self.chk_ticked)
        v.addLayout(row)
        self.tbl = QTableWidget(0, len(self.COLS))
        self.tbl.setHorizontalHeaderLabels(self.COLS)
        style_table(self.tbl)
        h = self.tbl.horizontalHeader()
        for c in range(len(self.COLS) - 1):
            h.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(len(self.COLS) - 1, QHeaderView.Stretch)
        v.addWidget(self.tbl, 1)
        self.lbl_count = QLabel(""); self.lbl_count.setProperty("muted", True); v.addWidget(self.lbl_count)
        row = QHBoxLayout()
        b = QPushButton("Add File…"); b.setObjectName("secondary"); b.clicked.connect(self._add); row.addWidget(b)
        b = QPushButton("Laboratory Set"); b.setObjectName("secondary")
        b.setToolTip("Import SO2, O3, NO2, HCHO measured cross sections from Input/modataXs.")
        b.clicked.connect(self._lab); row.addWidget(b)
        b = QPushButton("Synthetic Set"); b.setObjectName("secondary")
        b.setToolTip("Write the synthetic SO2, O3, NO2 and HCHO cross sections to Input/CrossSections.")
        b.clicked.connect(self._synthetic); row.addWidget(b)
        row.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject); row.addWidget(bb)
        v.addLayout(row)
        self._populate()

    @staticmethod
    def _info(p):
        """(T text, kind, quality, tooltip, search text) of a file."""
        r = xsm.catalog().get(p)
        T = xsm.file_temperature(p)
        Tt = "%g" % T if T is not None else "?"
        if r:
            tip = "%s\nreference: %s\nsource: Input/modataXs/%s%s%s" % (
                p, r["reference"], r["source"], "\nnote: " + r["note"] if r["note"] else "",
                "\nquality: " + r["quality"] if r["quality"] else "")
            return Tt, r["kind"], r["quality"], tip, " ".join((r["reference"], r["source"], r["note"]))
        top = os.path.dirname(p) == xsm.XS_DIR
        kind = ("synthetic" if "synthetic" in os.path.basename(p).lower() else "recommended") if top else ""
        return Tt, kind, "", p, kind

    def _populate(self):
        self.tbl.setRowCount(0)
        self._search = []
        order = sorted(self._files.items(), key=lambda kv: (kv[1][0].lower(), os.path.dirname(kv[0]) != xsm.XS_DIR,
                                                            xsm.file_temperature(kv[0]) or 1e9, kv[0]))
        for p, (gas, lo, hi) in order:
            r = self.tbl.rowCount(); self.tbl.insertRow(r)
            cb = QCheckBox(); cb.setChecked(p in self._chosen)
            cb.toggled.connect(lambda on, p=p: (self._chosen.add(p) if on else self._chosen.discard(p),
                                                self._apply_filter()))
            w = QWidget(); hl = QHBoxLayout(w); hl.setContentsMargins(0, 0, 0, 0); hl.setAlignment(Qt.AlignCenter)
            hl.addWidget(cb); self.tbl.setCellWidget(r, 0, w)
            Tt, kind, quality, tip, extra = self._info(p)
            rel = os.path.relpath(p, xsm.XS_DIR) if p.startswith(xsm.XS_DIR) else p
            covers = not self._region or (lo <= self._region[0] and hi >= self._region[1])
            cells = (table_cell(gas), table_cell(Tt), table_cell("%.1f – %.1f" % (lo, hi)), table_cell(kind),
                     table_cell(quality, Qt.AlignLeft | Qt.AlignVCenter),
                     table_cell(rel, Qt.AlignLeft | Qt.AlignVCenter))
            for c, it in enumerate(cells, start=1):
                if not covers:
                    it.setForeground(Qt.gray)
                it.setToolTip(tip); self.tbl.setItem(r, c, it)
            self._search.append((p, covers, " ".join((gas, Tt, rel, kind, quality, extra)).lower()))
        self._apply_filter()

    def _apply_filter(self, *_):
        words = self.ed_filter.text().lower().split()
        shown = 0
        for r, (p, covers, text) in enumerate(self._search):
            ok = all(w in text for w in words)
            ok = ok and (covers or not self.chk_cover.isChecked())
            ok = ok and (p in self._chosen or not self.chk_ticked.isChecked())
            self.tbl.setRowHidden(r, not ok)
            shown += ok
        self.lbl_count.setText("%d of %d files shown  ·  %d ticked" % (shown, len(self._search), len(self._chosen)))

    def _add(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Add cross-section files", xsm.XS_DIR,
                                                "Cross sections (*.xs *.txt *.dat *.csv *.xsc);;All files (*)")
        for p in paths:
            try:
                xs = xsm.load_xs(p)
            except Exception as e:
                QMessageBox.warning(self, "Add", "%s: %s" % (os.path.basename(p), e)); continue
            self._files[p] = (xs["gas"], xs["wl"][0], xs["wl"][-1]); self._chosen.add(p)
        self._populate()

    def _lab(self):
        for p in xsm.import_lab_set():
            xs = xsm.load_xs(p)
            self._files[p] = (xs["gas"], xs["wl"][0], xs["wl"][-1])
        self._populate()

    def _synthetic(self):
        for p in xsm.write_synthetic_set():
            xs = xsm.load_xs(p)
            self._files[p] = (xs["gas"], xs["wl"][0], xs["wl"][-1])
        self._populate()

    def values(self):
        return [p for p in self._files if p in self._chosen], self._files


class SimulateDialog(QDialog):
    """A synthetic measurement with the current slit and light path."""

    def __init__(self, files, settings, slit_desc, path_desc, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Synthetic Spectrum")
        self.resize(720, 640)
        self.setStyleSheet(STYLESHEET)
        s = settings
        self._files = files
        v = QVBoxLayout(self); v.setContentsMargins(14, 14, 14, 14); v.setSpacing(8)
        hint = QLabel("I = lamp · throughput · exp(−Σ SCDᵢσᵢ − τ_Rayleigh − τ_Mie), convolved with the slit at "
                      "high resolution and sampled at the pixels, with noise and a wavelength-calibration "
                      "shift. The true mixing ratios go in the file's header, so the retrieval has an answer "
                      "to compare against.<br><b>Slit:</b> %s<br><b>Light path:</b> %s" % (slit_desc, path_desc))
        hint.setProperty("muted", True); hint.setWordWrap(True); v.addWidget(hint)
        row = QHBoxLayout()
        cap = QLabel("Search"); cap.setProperty("muted", True); row.addWidget(cap)
        self.ed_filter = QLineEdit(); self.ed_filter.setPlaceholderText("gas, temperature, file ...  (ticked rows stay)")
        self.ed_filter.textChanged.connect(self._apply_filter); row.addWidget(self.ed_filter, 1)
        v.addLayout(row)
        self.tbl = QTableWidget(0, 5)
        self.tbl.setHorizontalHeaderLabels(["Use", "Gas", "T (K)", "ppb", "File"])
        style_table(self.tbl, editable=True)
        ppb = s.get("sim_ppb", {})
        self._rows = []
        for p, (gas, lo, hi) in sorted(files.items(), key=lambda kv: (kv[1][0].lower(),
                                                                      os.path.dirname(kv[0]) != xsm.XS_DIR, kv[0])):
            r = self.tbl.rowCount(); self.tbl.insertRow(r)
            # one file per gas ticked: the laboratory one where there is one
            cb = QCheckBox(); cb.setChecked(gas in ppb and os.path.normcase(p) == os.path.normcase(xsm.default_path(gas)))
            w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0); h.setAlignment(Qt.AlignCenter)
            h.addWidget(cb); self.tbl.setCellWidget(r, 0, w)
            self.tbl.setItem(r, 1, table_cell(gas))
            T = xsm.file_temperature(p)
            self.tbl.setItem(r, 2, table_cell("%g" % T if T is not None else "?"))
            sp = _dspin(0, 1e7, 1, 3, float(ppb.get(gas, 10.0))); self.tbl.setCellWidget(r, 3, sp)
            rel = os.path.relpath(p, xsm.XS_DIR) if p.startswith(xsm.XS_DIR) else p
            self.tbl.setItem(r, 4, table_cell(rel, Qt.AlignLeft | Qt.AlignVCenter))
            self._rows.append((p, gas, cb, sp))
            cb.toggled.connect(self._apply_filter)
        v.addWidget(self.tbl, 1)
        self._apply_filter()
        g, f = _form_page("Spectrometer and atmosphere")
        row = QHBoxLayout()
        self.sp_lo = _dspin(150, 1000, 1, 2, float(s["sim_lo"])); self.sp_hi = _dspin(150, 1000, 1, 2, float(s["sim_hi"]))
        row.addWidget(self.sp_lo); row.addWidget(QLabel("to")); row.addWidget(self.sp_hi); row.addStretch(1)
        f.addRow("Wavelength range (nm)", row)
        self.sp_pix = _dspin(0.001, 2, 0.005, 4, float(s["sim_pixel"]), "Detector pixel spacing.")
        f.addRow("Pixel step (nm)", self.sp_pix)
        self.sp_noise = _dspin(0, 0.1, 1e-4, 6, float(s["sim_noise"]),
                               "Relative RMS noise on I and on I₀ (after co-adding).")
        f.addRow("Noise (relative)", self.sp_noise)
        self.sp_shift = _dspin(-1, 1, 0.005, 4, float(s["sim_shift"]),
                               "Wavelength-calibration error: pixel λ really sees λ + shift.")
        f.addRow("Calibration shift (nm)", self.sp_shift)
        self.sp_vis = _dspin(0, 1000, 1, 1, float(s["sim_visibility"]),
                             "Aerosol (Mie) extinction from the visibility, Ångström α = 1.3. 0 = none.")
        f.addRow("Visibility (km)", self.sp_vis)
        self.sp_seed = QSpinBox(); self.sp_seed.setRange(0, 10 ** 6); self.sp_seed.setValue(int(s["sim_seed"]))
        f.addRow("Noise seed", self.sp_seed)
        self.ed_out = QLineEdit(os.path.join(HERE, "Input", "Sim_spectrum.txt"))
        row = QHBoxLayout(); row.addWidget(self.ed_out, 1)
        b = QPushButton("…"); b.setObjectName("secondary"); b.clicked.connect(self._browse); row.addWidget(b)
        f.addRow("Save as", row)
        v.addWidget(g)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Simulate")
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject); v.addWidget(bb)
        for _p, _g, cb, _sp in self._rows:
            cb.toggled.connect(self._suggest_name)
        self._suggest_name()

    def _apply_filter(self, *_):
        words = self.ed_filter.text().lower().split()
        for r, (p, gas, cb, _sp) in enumerate(self._rows):
            text = " ".join((gas, self.tbl.item(r, 2).text(), p)).lower()
            self.tbl.setRowHidden(r, not (cb.isChecked() or all(w in text for w in words)))

    def _suggest_name(self, *_):
        gases = [g for _p, g, cb, _sp in self._rows if cb.isChecked()]
        if gases:
            self.ed_out.setText(os.path.join(HERE, "Input", "Sim_%s.txt" % "_".join(gases)))

    def _browse(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save synthetic spectrum", self.ed_out.text(), "Text (*.txt)")
        if p:
            self.ed_out.setText(p)

    def values(self):
        return {"gases": [(p, g, sp.value()) for p, g, cb, sp in self._rows if cb.isChecked()],
                "lo": self.sp_lo.value(), "hi": self.sp_hi.value(), "pixel": self.sp_pix.value(),
                "noise": self.sp_noise.value(), "shift": self.sp_shift.value(),
                "visibility": self.sp_vis.value(), "seed": self.sp_seed.value(), "out": self.ed_out.text()}


# =============================================================================
# Main window
# =============================================================================
class DoasWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(WIN_W, WIN_H)
        self.setStyleSheet(STYLESHEET)
        self._settings = load_settings()
        self._wl = None                 # pixel wavelengths, nm
        self._I = self._I0 = None       # intensities (None for an optical-depth file)
        self._od_file = None            # optical depth read from the file
        self._spec_path = ""
        self._i0_src = ""
        self._xs = []                   # cross-section dicts of the gases to fit
        self._xs_files = {}             # every known cross-section file: path -> (gas, lo, hi)
        self._slit = None               # (offset_nm, values)
        self._slit_src = ""
        self._ref_gases = {}            # {gas: ppb} from a simulated file's header
        self._result = None
        self._worker = None
        self._busy = False
        self._build_menu_bar(); self._build_central()
        self._init_slit(); self._init_gases()
        self._redraw()
        self._update_status()
        self.statusBar().showMessage("Ready. Open a spectrum, select the gases and the slit, then Retrieve.")

    # ---------------------------------------------------------------- menu --
    def _build_menu_bar(self):
        mb = QMenuBar(self); self.setMenuBar(mb)

        fm = mb.addMenu("File")
        a = QAction("Open Spectrum…", self); a.setShortcut("Ctrl+O")
        a.setToolTip("Three columns (λ, I₀, I), or two (λ, I) or (λ, ln(I₀/I)).")
        a.triggered.connect(self.open_spectrum); fm.addAction(a)
        a = QAction("Open Reference I₀…", self); a.setShortcut("Ctrl+Shift+O")
        a.setToolTip("Two columns (λ, I₀): the lamp / reference spectrum for a two-column I file.")
        a.triggered.connect(self.open_i0); fm.addAction(a)
        fm.addSeparator()
        a = QAction("Save Result…", self); a.setShortcut("Ctrl+S")
        a.triggered.connect(self.save_result); fm.addAction(a)
        a = QAction("Export Curves…", self)
        a.setToolTip("Optical depth, model, polynomial, residual and each gas's differential\n"
                     "absorption in the fit window, as text columns.")
        a.triggered.connect(self.export_curves); fm.addAction(a)
        a = QAction("Save Plot…", self); a.triggered.connect(self.save_plot); fm.addAction(a)
        fm.addSeparator()
        a = QAction("Exit", self); a.triggered.connect(self.close); fm.addAction(a)

        xm = mb.addMenu("Cross Section")
        a = QAction("Select Gases…", self); a.setShortcut("Ctrl+G")
        a.triggered.connect(self.select_gases); xm.addAction(a)
        a = QAction("Add Cross-section File…", self)
        a.setToolTip("Two columns: wavelength (nm) or wavenumber (cm⁻¹), σ (cm²/molecule).\n"
                     "E.g. a file from the MPI-Mainz UV/VIS Spectral Atlas.")
        a.triggered.connect(self.add_xs_file); xm.addAction(a)
        xm.addSeparator()
        a = QAction("Import Laboratory Cross Sections", self)
        a.setToolTip("SO2 (Vandaele), O3 and NO2 (IUP Bremen FTS), HCHO (Cantrell) from Input/modataXs,\n"
                     "on the air wavelength scale, into Input/CrossSections.")
        a.triggered.connect(self.import_lab_xs); xm.addAction(a)
        a = QAction("Write Synthetic Cross Sections", self)
        a.setToolTip("SO2, O3, NO2, HCHO synthetic cross sections into Input/CrossSections.")
        a.triggered.connect(self.write_synthetic_xs); xm.addAction(a)

        sm = mb.addMenu("Slit")
        a = QAction("Load Slit File…", self); a.setShortcut("Ctrl+I")
        a.setToolTip("Two columns: wavelength or offset (nm), value - e.g. a recorded Hg line.")
        a.triggered.connect(self.load_slit); sm.addAction(a)
        a = QAction("Synthetic Slit Function…", self); a.setShortcut("Ctrl+Shift+I")
        a.triggered.connect(self.synthetic_slit); sm.addAction(a)

        rm = mb.addMenu("Retrieval")
        a = QAction("Retrieve Concentration", self); a.setShortcut("Ctrl+R")
        a.triggered.connect(self.run_retrieval); rm.addAction(a)
        rm.addSeparator()
        a = QAction("Retrieval Settings…", self); a.triggered.connect(self.show_settings); rm.addAction(a)

        simm = mb.addMenu("Simulate")
        a = QAction("Synthetic Spectrum…", self); a.setShortcut("Ctrl+M")
        a.setToolTip("A test spectrum with gases you choose, the current slit and light path.")
        a.triggered.connect(self.simulate_spectrum); simm.addAction(a)

        hp = mb.addMenu("Help")
        a = QAction("Load Example (single gas)", self)
        a.setToolTip("SO2_simulated_15ppb.txt, 1 km path, Gaussian slit 0.5 nm - true answer 15 ppb SO2.")
        a.triggered.connect(lambda: self.load_example("single")); hp.addAction(a)
        a = QAction("Load Multi-gas Example", self)
        a.setToolTip("SO2 8, O3 60, NO2 25, HCHO 10 ppb fitted together (laboratory cross sections).")
        a.triggered.connect(lambda: self.load_example("multi")); hp.addAction(a)
        hp.addSeparator()
        a = QAction("About", self); a.triggered.connect(self.show_about); hp.addAction(a)
        a = QAction("Method && Caveats", self); a.triggered.connect(self.show_method); hp.addAction(a)

    # ------------------------------------------------------------- central --
    def _build_central(self):
        c = QWidget(); self.setCentralWidget(c)
        main = QHBoxLayout(c); main.setContentsMargins(10, 10, 10, 10); main.setSpacing(10)
        left = QVBoxLayout(); left.setSpacing(10)
        right = QVBoxLayout(); right.setSpacing(10)
        main.addLayout(left, 58); main.addLayout(right, 42)
        self._build_plot_box(left)
        self._build_action_row(left)
        self._build_inputs_box(right)
        self._build_conditions_box(right)
        self._build_result_tabs(right)

    # -- left column ---------------------------------------------------------
    def _build_plot_box(self, parent_layout):
        box = QGroupBox("Spectrum")
        v = QVBoxLayout(box); v.setContentsMargins(12, 12, 12, 10); v.setSpacing(6)
        self.plot = PubCanvas(min_height=300)
        mouse_navigation(self.plot, lambda: self._redraw())
        v.addWidget(self.plot, 1)
        row = QHBoxLayout(); row.setSpacing(10)
        cap = QLabel("View"); cap.setProperty("muted", True); row.addWidget(cap)
        self.combo_view = QComboBox()
        for k, lab in VIEWS: self.combo_view.addItem(lab, k)
        self.combo_view.currentIndexChanged.connect(lambda _i: self._redraw())
        row.addWidget(self.combo_view)
        row.addSpacing(8)
        self.chk_show = {}
        for key, text, on in (("data", "Data", True), ("fit", "Model", True), ("poly", "Polynomial", True),
                              ("residual", "Residual", True)):
            cb = QCheckBox(text); cb.setChecked(on); cb.toggled.connect(lambda _c: self._redraw(True))
            row.addWidget(cb); self.chk_show[key] = cb
        row.addStretch(1)
        v.addLayout(row)
        parent_layout.addWidget(box, 1)

    def _build_action_row(self, parent_layout):
        row = QHBoxLayout(); row.setSpacing(8)
        cap = QLabel("Fit window (nm)"); cap.setProperty("muted", True); row.addWidget(cap)
        self.spin_lo = QDoubleSpinBox(); self.spin_hi = QDoubleSpinBox()
        for sp in (self.spin_lo, self.spin_hi):
            sp.setDecimals(3); sp.setRange(0, 1e5); sp.setSingleStep(0.5); sp.setMinimumWidth(110)
            sp.editingFinished.connect(lambda: self._redraw())
        row.addWidget(self.spin_lo); row.addWidget(QLabel("to")); row.addWidget(self.spin_hi)
        b = QPushButton("Use view"); b.setObjectName("secondary")
        b.setToolTip("Set the fit window to the x range shown in the plot (zoom: mouse wheel, pan: drag).")
        b.clicked.connect(self._region_from_view); row.addWidget(b)
        b = QPushButton("Full"); b.setObjectName("secondary")
        b.clicked.connect(self._region_full); row.addWidget(b)
        row.addSpacing(12)
        self.btn_run = QPushButton("Retrieve Concentration"); self.btn_run.setMinimumHeight(38)
        self.btn_run.setToolTip("Linear DOAS fit of σ ⊛ slit + polynomial to ln(I₀/I) in the window.")
        self.btn_run.clicked.connect(self.run_retrieval)
        row.addWidget(self.btn_run, 1)
        parent_layout.addLayout(row)
        self.lbl_status = QLabel("Open a spectrum first."); self.lbl_status.setProperty("muted", True)
        self.lbl_status.setWordWrap(True)
        parent_layout.addWidget(self.lbl_status)

    # -- right column --------------------------------------------------------
    def _build_inputs_box(self, parent_layout):
        box = QGroupBox("Inputs")
        g = QGridLayout(box); g.setContentsMargins(12, 10, 12, 10)
        g.setHorizontalSpacing(8); g.setVerticalSpacing(6)

        def file_row(r, caption, slot, extra=None, open_text="Open…"):
            cap = QLabel(caption); cap.setProperty("cap", True); g.addWidget(cap, r, 0)
            lbl = QLabel("none"); lbl.setProperty("muted", True); lbl.setWordWrap(True)
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            g.addWidget(lbl, r, 1)
            b = QPushButton(open_text); b.setObjectName("secondary"); b.clicked.connect(slot)
            g.addWidget(b, r, 2)
            if extra:
                b2 = QPushButton(extra[0]); b2.setObjectName("secondary"); b2.clicked.connect(extra[1])
                g.addWidget(b2, r, 3)
            return lbl

        self.lbl_spec = file_row(0, "Spectrum", self.open_spectrum)
        self.lbl_i0 = file_row(1, "Reference I₀", self.open_i0)
        self.lbl_gases = file_row(2, "Gases", self.select_gases, open_text="Select…")
        self.lbl_slit = file_row(3, "Slit", self.load_slit, extra=("Synthetic…", self.synthetic_slit))
        cap = QLabel("Data is"); cap.setProperty("muted", True); g.addWidget(cap, 4, 0)
        self.combo_kind = QComboBox()
        self.combo_kind.addItem("Intensity I  (with reference I₀)", "I")
        self.combo_kind.addItem("Optical depth  ln(I₀/I)", "OD")
        self.combo_kind.currentIndexChanged.connect(self._on_kind_changed)
        g.addWidget(self.combo_kind, 4, 1, 1, 2)
        g.setColumnStretch(1, 1)
        parent_layout.addWidget(box)

    def _build_conditions_box(self, parent_layout):
        box = QGroupBox("Light path")
        g = QGridLayout(box); g.setContentsMargins(12, 10, 12, 10)
        g.setHorizontalSpacing(8); g.setVerticalSpacing(6)
        s = self._settings
        self.spin_T = _dspin(1, 1000, 0.5, 2, float(s["T_K"]))
        self.spin_P = _dspin(0, 1e9, 1, 4, float(s["P_value"]))
        self.combo_P = QComboBox()
        for u in PRESSURE_UNITS: self.combo_P.addItem(u, u)
        self.combo_P.setCurrentIndex(max(0, self.combo_P.findData(s["P_unit"])))
        self._p_unit = self.combo_P.currentData()
        self.combo_P.currentIndexChanged.connect(self._on_p_unit_changed)
        self.spin_L = _dspin(1e-6, 1e9, 0.1, 4, float(s["L_value"]))
        self.combo_L = QComboBox()
        for u in LENGTH_UNITS: self.combo_L.addItem(u, u)
        self.combo_L.setCurrentIndex(max(0, self.combo_L.findData(s["L_unit"])))
        self._l_unit = self.combo_L.currentData()
        self.combo_L.currentIndexChanged.connect(self._on_l_unit_changed)
        for r, (name, w, unit, tip) in enumerate((
                ("Temperature", self.spin_T, QLabel("K"), "Air temperature along the light path."),
                ("Pressure", self.spin_P, self.combo_P, "Air pressure along the light path."),
                ("Light path", self.spin_L, self.combo_L,
                 "TOTAL light path: for a long-path DOAS with a retro-reflector,\n"
                 "twice the telescope-reflector distance."))):
            cap = QLabel(name); cap.setProperty("muted", True); cap.setToolTip(tip)
            g.addWidget(cap, r, 0); g.addWidget(w, r, 1); g.addWidget(unit, r, 2)
        g.setColumnStretch(1, 1)
        parent_layout.addWidget(box)

    def _build_result_tabs(self, parent_layout):
        tabs = QTabWidget(); self._tabs = tabs
        parent_layout.addWidget(tabs, 1)
        self._build_result_tab(tabs)
        self._build_slit_tab(tabs)
        self._build_xs_tab(tabs)
        self._build_log_tab(tabs)

    def _summary_grid(self, box, rows):
        sg = QGridLayout(box); sg.setContentsMargins(12, 10, 12, 10)
        sg.setHorizontalSpacing(8); sg.setVerticalSpacing(5)
        mono = QFont("Consolas"); mono.setPointSize(10)
        out = {}
        for r, (name, tip) in enumerate(rows):
            cap = QLabel(name); cap.setProperty("muted", True); cap.setToolTip(tip)
            val = QLabel("—"); val.setFont(mono); val.setProperty("cap", True); val.setToolTip(tip)
            val.setWordWrap(True); val.setTextInteractionFlags(Qt.TextSelectableByMouse)
            sg.addWidget(cap, r, 0); sg.addWidget(val, r, 1)
            out[name] = val
        sg.setColumnStretch(1, 1)
        return out

    def _build_result_tab(self, tabs):
        tab = QWidget(); tabs.addTab(tab, "Result")
        v = QVBoxLayout(tab); v.setContentsMargins(8, 10, 8, 8); v.setSpacing(8)
        box = QGroupBox("Concentration")
        bv = QVBoxLayout(box); bv.setContentsMargins(12, 12, 12, 10); bv.setSpacing(6)
        self.grid_conc = QGridLayout(); self.grid_conc.setHorizontalSpacing(14); self.grid_conc.setVerticalSpacing(4)
        bv.addLayout(self.grid_conc)
        self.lbl_conc_sub = QLabel("path average · ± one standard error of the fit")
        self.lbl_conc_sub.setProperty("muted", True); self.lbl_conc_sub.setAlignment(Qt.AlignCenter)
        bv.addWidget(self.lbl_conc_sub)
        row = QHBoxLayout(); row.addStretch(1)
        self.btn_details = QPushButton("Details…"); self.btn_details.setObjectName("secondary")
        self.btn_details.setToolTip("Slant columns, fit parameters, correlations and warnings.")
        self.btn_details.clicked.connect(self._show_details); row.addWidget(self.btn_details)
        row.addStretch(1); bv.addLayout(row)
        v.addWidget(box)
        self.tbl_gas = QTableWidget(0, 6)
        self.tbl_gas.setHorizontalHeaderLabels(["Gas", "SCD (cm⁻²)", "± SCD", "Conc.", "± Conc.", "ΔOD'"])
        style_table(self.tbl_gas)
        self.tbl_gas.setMinimumHeight(120)
        v.addWidget(self.tbl_gas, 1)
        self._set_gas_rows([])

        self._dlg_details = QDialog(self)
        self._dlg_details.setWindowTitle("Result Details")
        self._dlg_details.setMinimumWidth(620)
        self._dlg_details.setStyleSheet(STYLESHEET)
        dv = QVBoxLayout(self._dlg_details); dv.setContentsMargins(14, 14, 14, 14); dv.setSpacing(10)
        box = QGroupBox("Details")
        self.lbl_res = self._summary_grid(box, [
            ("Slant columns", "SCD of every gas, molecule/cm² - what the DOAS fit gives directly."),
            ("Number density", "SCD / L, molecule/cm³ (path average)."),
            ("Reference", "The true mixing ratios written in a simulated file's header, and the\n"
                          "deviation of the retrieval from them in units of its own error."),
            ("Detection limit", "2 × the fit error, as a mixing ratio."),
            ("Correlation", "Correlation coefficients between the gases' SCDs.\n"
                            "|r| near 1: the window does not separate the two gases."),
            ("Shift / squeeze", "Wavelength shift of the cross sections against the spectrum (nm), squeeze."),
            ("Polynomial", "Legendre coefficients in the normalised wavelength (−1..1 over the window)."),
            ("RMS / χ²", "Residual RMS (optical depth), reduced χ² (= residual variance), R²."),
            ("Air", "N_air = P / k_B T, molecule/cm³, and the light path."),
            ("Points / time", "Pixels in the window, degrees of freedom, solve time."),
            ("Warnings", "Parameters on a bound, cross sections that do not cover the window, ..."),
        ])
        dv.addWidget(box, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self._dlg_details.close); dv.addWidget(bb)

    def _show_details(self):
        d = self._dlg_details
        d.show(); d.raise_(); d.activateWindow()

    def _set_gas_rows(self, rows):
        """rows: (gas, 'value ± u unit', reference note or '').  Empty: a dash."""
        g = self.grid_conc
        while g.count():
            w = g.takeAt(0).widget()
            if w is not None:
                w.hide(); w.setParent(None); w.deleteLater()
        if not rows:
            rows = [("", "—", "")]
        big = len(rows) == 1
        css = "font-size: %dpx; font-weight: 800; color: #1e3a8a;" % (28 if big else 20)
        for i, (gas, val, note) in enumerate(rows):
            for c, (text, style, align) in enumerate((
                    (gas, css, Qt.AlignRight), (val, css, Qt.AlignLeft), (note, None, Qt.AlignLeft))):
                lab = QLabel(text); lab.setAlignment(align | Qt.AlignVCenter)
                lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
                if style: lab.setStyleSheet(style)
                else: lab.setProperty("muted", True)
                g.addWidget(lab, i, c)
        g.setColumnStretch(0, 1); g.setColumnStretch(2, 1)

    def _build_slit_tab(self, tabs):
        tab = QWidget(); tabs.addTab(tab, "Slit")
        v = QVBoxLayout(tab); v.setContentsMargins(8, 10, 8, 8); v.setSpacing(8)
        self.slit_plot = PubCanvas(min_height=220)
        mouse_navigation(self.slit_plot, lambda: self._draw_slit())
        v.addWidget(self.slit_plot, 1)
        self.lbl_slit_stats = QLabel("—"); self.lbl_slit_stats.setProperty("muted", True)
        self.lbl_slit_stats.setWordWrap(True)
        v.addWidget(self.lbl_slit_stats)

    def _build_xs_tab(self, tabs):
        tab = QWidget(); tabs.addTab(tab, "Cross Sections")
        v = QVBoxLayout(tab); v.setContentsMargins(8, 10, 8, 8); v.setSpacing(8)
        row = QHBoxLayout()
        self.combo_xs_view = QComboBox()
        self.combo_xs_view.addItem("σ high-res and σ ⊛ slit", "conv")
        self.combo_xs_view.addItem("Differential σ' (after the polynomial)", "diff")
        self.combo_xs_view.currentIndexChanged.connect(lambda _i: self._draw_xs())
        row.addWidget(self.combo_xs_view); row.addStretch(1)
        v.addLayout(row)
        self.xs_plot = PubCanvas(min_height=220)
        mouse_navigation(self.xs_plot, lambda: self._draw_xs())
        v.addWidget(self.xs_plot, 1)

    def _build_log_tab(self, tabs):
        tab = QWidget(); tabs.addTab(tab, "Log")
        lv = QVBoxLayout(tab); lv.setContentsMargins(8, 10, 8, 8); lv.setSpacing(8)
        self.txt_log = QTextEdit(); self.txt_log.setReadOnly(True)
        self.txt_log.setLineWrapMode(QTextEdit.NoWrap)
        self.txt_log.setFontFamily("Consolas"); self.txt_log.setFontPointSize(9)
        lv.addWidget(self.txt_log, 1)
        row = QHBoxLayout(); row.addStretch(1)
        b = QPushButton("Clear"); b.setObjectName("secondary"); b.setMaximumWidth(100)
        b.clicked.connect(self.txt_log.clear); row.addWidget(b)
        lv.addLayout(row)

    # ------------------------------------------------------------- helpers --
    def log(self, msg):
        self.txt_log.append("[%s] %s" % (datetime.now().strftime("%H:%M:%S"), msg))

    def region(self):
        lo, hi = self.spin_lo.value(), self.spin_hi.value()
        return (lo, hi) if hi > lo else None

    def pressure_pa(self):
        return self.spin_P.value() * PRESSURE_UNITS[self.combo_P.currentData()]

    def path_cm(self):
        return self.spin_L.value() * LENGTH_UNITS[self.combo_L.currentData()]

    def optical_depth(self):
        """ln(I0/I) on the spectrum's pixels, or None while I0 is missing."""
        if self._wl is None:
            return None
        if self.combo_kind.currentData() == "OD":
            return self._od_file if self._od_file is not None else self._I
        if self._I is None or self._I0 is None:
            return None
        return doas.optical_depth(self._I, self._I0)

    def _plot_data(self):
        od_mode = self.combo_kind.currentData() == "OD"
        return {"wl": self._wl, "od": self.optical_depth(),
                "I": None if od_mode else self._I, "I0": None if od_mode else self._I0}

    def _set_busy(self, busy, text=""):
        self._busy = busy
        self.btn_run.setEnabled(not busy)
        if text:
            self.lbl_status.setText(text)
        QApplication.setOverrideCursor(Qt.WaitCursor) if busy else QApplication.restoreOverrideCursor()

    def _remember_dir(self, key, path):
        self._settings[key] = os.path.dirname(path)
        self._save_settings()

    def _save_settings(self):
        s = self._settings
        s["T_K"], s["P_value"], s["P_unit"] = self.spin_T.value(), self.spin_P.value(), self.combo_P.currentData()
        s["L_value"], s["L_unit"] = self.spin_L.value(), self.combo_L.currentData()
        s["xs_paths"] = [x["source"] for x in self._xs]
        try: save_json(settings_path(), s)
        except Exception as e: self.log("could not save settings: %s" % e)

    def _start(self, name, job, on_done, busy_text):
        if self._busy:
            return
        self._set_busy(True, busy_text)
        self._worker = JobWorker([(Job(name), job)], self)
        self._worker.done.connect(on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(lambda: self._set_busy(False))
        self._worker.start()

    def _on_failed(self, msg):
        self.log("FAILED %s" % msg)
        self.lbl_status.setText("Failed: %s" % msg)
        QMessageBox.warning(self, "Failed", msg)

    def _path_desc(self):
        return "L %g %s, T %g K, P %g %s" % (self.spin_L.value(), self.combo_L.currentData(), self.spin_T.value(),
                                             self.spin_P.value(), self.combo_P.currentData())

    # -- settings ------------------------------------------------------------
    def show_settings(self):
        dlg = SettingsDialog(self._settings, self)
        dlg.setStyleSheet(STYLESHEET)
        if dlg.exec_() != QDialog.Accepted:
            return
        before = dict(self._settings)
        self._settings = dlg.values()
        self._save_settings()
        changed = sorted(k for k in self._settings if self._settings[k] != before.get(k))
        if changed:
            self.log("settings changed: %s" % ", ".join(changed))
        self._draw_xs()

    def _on_p_unit_changed(self, _i):
        """Keep the physical pressure, change only the unit it is shown in."""
        new = self.combo_P.currentData()
        pa = self.spin_P.value() * PRESSURE_UNITS[self._p_unit]
        self._p_unit = new
        self.spin_P.setValue(pa / PRESSURE_UNITS[new])

    def _on_l_unit_changed(self, _i):
        new = self.combo_L.currentData()
        cm = self.spin_L.value() * LENGTH_UNITS[self._l_unit]
        self._l_unit = new
        self.spin_L.setValue(cm / LENGTH_UNITS[new])

    def _set_conditions(self, T_K, P_Pa, L_cm):
        self.spin_T.setValue(T_K)
        self.spin_P.setValue(P_Pa / PRESSURE_UNITS[self.combo_P.currentData()])
        unit = "km" if L_cm >= 1e5 else ("m" if L_cm >= 100 else "cm")
        self.combo_L.blockSignals(True); self.combo_L.setCurrentIndex(self.combo_L.findData(unit))
        self.combo_L.blockSignals(False); self._l_unit = unit
        self.spin_L.setValue(L_cm / LENGTH_UNITS[unit])

    # -- region --------------------------------------------------------------
    def _set_region(self, lo, hi):
        self.spin_lo.setValue(lo); self.spin_hi.setValue(hi)

    def _region_from_view(self):
        if self._wl is None or not self.plot.fig.axes:
            return
        lo, hi = self.plot.fig.axes[0].get_xlim()
        self._set_region(max(lo, self._wl[0]), min(hi, self._wl[-1]))
        self._redraw(True)

    def _region_full(self):
        if self._wl is None:
            return
        self._set_region(self._wl[0], self._wl[-1])
        self._redraw()

    # -- spectrum ------------------------------------------------------------
    def open_spectrum(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open spectrum", self._settings["last_open_dir"] or HERE,
                                              "Spectra (*.txt *.dat *.csv *.std *.spe);;All files (*)")
        if path:
            self._remember_dir("last_open_dir", path)
            self.load_spectrum_file(path)

    def load_spectrum_file(self, path):
        try:
            data, header = xsm.read_columns(path)
        except Exception as e:
            QMessageBox.critical(self, "Open failed", "%s: %s" % (type(e).__name__, e)); return False
        if data.shape[1] < 2:
            QMessageBox.critical(self, "Open failed", "Need at least two columns."); return False
        o = np.argsort(data[:, 0]); data = data[o]
        self._wl = data[:, 0]
        self._spec_path = path
        self._result = None
        self._od_file = None
        if data.shape[1] >= 3:
            self._I0, self._I = data[:, 1], data[:, 2]
            self._i0_src = "column 2 of the spectrum file"
            kind = "I"
        else:
            self._I = data[:, 1]
            # two columns are intensity unless the header says optical depth
            hdr = " ".join(header).lower()
            kind = "OD" if ("optical depth" in hdr or "ln(i0/i)" in hdr or re.search(r"od", hdr)) else "I"
            if kind == "OD":
                self._od_file = data[:, 1]
            if self._I0 is not None and len(self._I0) != len(self._wl):
                self._I0 = None
            if self._i0_src.startswith("column"):
                self._I0 = None; self._i0_src = ""
        self.combo_kind.blockSignals(True)
        self.combo_kind.setCurrentIndex(self.combo_kind.findData(kind))
        self.combo_kind.blockSignals(False)
        name = os.path.basename(path)
        self.lbl_spec.setText("%s  ·  %d pixels, %.2f–%.2f nm" % (name, len(self._wl), self._wl[0], self._wl[-1]))
        self.lbl_spec.setToolTip(path)
        self._show_i0()
        self._ref_gases, cond = read_sim_header(header)
        if cond:
            self._set_conditions(*cond)
            self.log("%s: simulated with %s; T %g K, P %g Pa, L %g cm - light path taken from it"
                     % (name, ", ".join("%s %g ppb" % kv for kv in self._ref_gases.items()), *cond))
        r = self.region()
        if r is None or r[1] < self._wl[0] or r[0] > self._wl[-1]:
            self._set_region(self._wl[0], self._wl[-1])
        self.log("opened %s (%d pixels, %s)" % (path, len(self._wl), "I₀ and I" if data.shape[1] >= 3 else kind))
        self._show_result(None)
        self._redraw()
        self._update_status()
        return True

    def open_i0(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open reference I₀", self._settings["last_open_dir"] or HERE,
                                              "Spectra (*.txt *.dat *.csv *.std *.spe);;All files (*)")
        if not path:
            return
        self._remember_dir("last_open_dir", path)
        try:
            data, _ = xsm.read_columns(path, max_cols=2)
        except Exception as e:
            QMessageBox.critical(self, "Open failed", "%s: %s" % (type(e).__name__, e)); return
        o = np.argsort(data[:, 0]); x, y = data[o, 0], data[o, 1]
        if self._wl is None:
            QMessageBox.information(self, "Reference I₀", "Open the spectrum first."); return
        self._I0 = np.interp(self._wl, x, y, left=np.nan, right=np.nan)
        self._i0_src = path
        self.combo_kind.setCurrentIndex(self.combo_kind.findData("I"))
        self.log("reference I₀ %s (%d points, interpolated to the spectrum's pixels)" % (path, len(x)))
        self._show_i0(); self._result = None; self._show_result(None); self._redraw(); self._update_status()

    def _show_i0(self):
        if self.combo_kind.currentData() == "OD":
            self.lbl_i0.setText("not needed (optical depth)")
        elif self._I0 is None:
            self.lbl_i0.setText("none - open the lamp / reference spectrum")
        else:
            self.lbl_i0.setText(os.path.basename(self._i0_src) if os.path.isfile(self._i0_src) else self._i0_src)
        self.lbl_i0.setToolTip(self._i0_src)

    def _on_kind_changed(self, _i):
        if self.combo_kind.currentData() == "OD" and self._od_file is None and self._I is not None:
            self._od_file = self._I
        self._result = None
        self._show_i0(); self._show_result(None); self._redraw(True); self._update_status()

    # -- cross sections ------------------------------------------------------
    def _init_gases(self):
        try:
            xsm.import_lab_set()
        except Exception as e:
            self.log("laboratory cross sections: %s" % e)
        self._xs_files = xsm.scan_folder()
        paths = [p for p in self._settings.get("xs_paths") or [] if os.path.isfile(p)]
        self._set_gases(paths, quiet=True)

    def _set_gases(self, paths, quiet=False):
        xs = []
        for p in paths:
            try:
                x = xsm.load_xs(p); xs.append(x)
                self._xs_files[p] = (x["gas"], x["wl"][0], x["wl"][-1])
            except Exception as e:
                self.log("cross section %s: %s" % (p, e))
        self._xs = xs
        self.lbl_gases.setText("  ·  ".join("%s%s" % (x["gas"], " (synthetic)" if x.get("synthetic") else "")
                                            for x in xs) or "none")
        self.lbl_gases.setToolTip("\n".join(x["source"] for x in xs))
        if not quiet:
            self.log("gases: %s" % ", ".join("%s (%s)" % (x["gas"], x["source"]) for x in xs))
        self._result = None
        self._show_result(None)
        self._draw_xs(); self._update_status()

    def select_gases(self):
        self._xs_files.update(xsm.scan_folder())
        dlg = GasDialog(self._xs_files, [x["source"] for x in self._xs], self.region(), self)
        if dlg.exec_() != QDialog.Accepted:
            return
        paths, files = dlg.values()
        self._xs_files.update(files)
        self._set_gases(paths)
        self._save_settings(); self._redraw()

    def add_xs_file(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Add cross-section files", xsm.XS_DIR,
                                                "Cross sections (*.xs *.txt *.dat *.csv *.xsc);;All files (*)")
        if not paths:
            return
        cur = [x["source"] for x in self._xs]
        self._set_gases(cur + [p for p in paths if p not in cur])
        self._save_settings(); self._redraw()

    def import_lab_xs(self):
        paths = xsm.import_lab_set(overwrite=True)
        self._xs_files.update(xsm.scan_folder())
        self.log("laboratory cross sections in %s: %s" % (xsm.XS_DIR, ", ".join(os.path.basename(p) for p in paths)))
        QMessageBox.information(self, "Laboratory cross sections", "Imported:\n%s\n\nCross Section → Select Gases "
                                "to use them." % "\n".join(os.path.basename(p) for p in paths))

    def write_synthetic_xs(self):
        paths = xsm.write_synthetic_set()
        self._xs_files.update(xsm.scan_folder())
        self.log("synthetic cross sections in %s: %s" % (xsm.XS_DIR, ", ".join(os.path.basename(p) for p in paths)))
        QMessageBox.information(self, "Synthetic cross sections",
                                "Written to %s.\nCross Section → Select Gases to use them." % xsm.XS_DIR)

    def _draw_xs(self):
        fig = self.xs_plot.fig; fig.clear()
        reg = self.region()
        if not self._xs or self._slit is None or reg is None:
            _empty(fig, "Select gases and a slit"); self.xs_plot.draw(); return
        ax = fig.add_subplot(111)
        lo, hi = reg
        diff = self.combo_xs_view.currentData() == "diff"
        try:
            for k, xs in enumerate(self._xs):
                col = "C%d" % (k % 10)
                ref = doas.Reference(xs, self._slit[0], self._slit[1], lo, hi, int(self._settings["oversample"]))
                g = np.linspace(lo, hi, 2000)
                c = ref(g)
                if diff:
                    ax.plot(g, doas.differential(c, g, int(self._settings["poly_order"]), lo, hi), "-", color=col,
                            label="%s σ′" % chem(xs["gas"]))
                else:
                    m = (xs["wl"] >= lo) & (xs["wl"] <= hi)
                    ax.plot(xs["wl"][m], xs["sigma"][m], "-", color=col, lw=0.5, alpha=0.35)
                    ax.plot(g, c, "-", color=col, label="%s ⊛ slit" % chem(xs["gas"]))
            ax.set_yscale("linear" if diff else "log")
        except Exception as e:
            ax.text(0.5, 0.5, str(e), transform=ax.transAxes, ha="center", color=INK_MUTED)
        pub_axes(ax, "Wavelength [nm]", "σ [cm²/molecule]"); pub_legend(ax, loc="best")
        self.xs_plot.draw()

    # -- slit ----------------------------------------------------------------
    def _init_slit(self):
        kind = self._settings.get("slit_kind", "gauss")
        if kind not in slit_mod.KINDS:
            kind = "gauss"
        self._set_synthetic_slit(kind, dict(self._settings.get("slit_params") or {}), quiet=True)

    def _set_synthetic_slit(self, kind, params, quiet=False):
        p = slit_mod.default_params(kind); p.update({k: v for k, v in params.items() if k in p})
        x, y = slit_mod.synthetic_slit(kind, p)
        self._settings["slit_kind"], self._settings["slit_params"] = kind, p
        self.set_slit(x, y, "synthetic " + slit_mod.describe(kind, p), quiet)

    def synthetic_slit(self):
        dlg = SlitDialog(self._settings.get("slit_kind", "gauss"), self._settings.get("slit_params") or {}, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        self._set_synthetic_slit(*dlg.values())
        self._save_settings()

    def load_slit(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load slit function", self._settings["last_open_dir"] or HERE,
                                              "Slit (*.txt *.dat *.csv *.slf);;All files (*)")
        if not path:
            return
        self._remember_dir("last_open_dir", path)
        try:
            x, y = slit_mod.load_slit(path)
        except Exception as e:
            QMessageBox.critical(self, "Load failed", "%s: %s" % (type(e).__name__, e)); return
        self.set_slit(x, y, path)

    def set_slit(self, x, y, src, quiet=False):
        self._slit = (np.asarray(x, float), np.asarray(y, float))
        self._slit_src = src
        st = slit_mod.slit_stats(*self._slit)
        self.lbl_slit.setText("%s  ·  FWHM %.3f nm" % (os.path.basename(src) if os.path.isfile(src) else src, st["fwhm"]))
        self.lbl_slit.setToolTip(src)
        self.lbl_slit_stats.setText("FWHM %.4f nm  ·  left %.4f / right %.4f nm  ·  centroid %+.4f nm  ·  "
                                    "asymmetry %+.3f  ·  %d points, %.3f to %.3f nm"
                                    % (st["fwhm"], st["left"], st["right"], st["centroid"], st["asymmetry"],
                                       len(x), x[0], x[-1]))
        if not quiet:
            self.log("slit %s, FWHM %.4f nm" % (src, st["fwhm"]))
        self._result = None; self._show_result(None)
        self._draw_slit(); self._draw_xs(); self._update_status()

    def _draw_slit(self):
        fig = self.slit_plot.fig; fig.clear()
        ax = fig.add_subplot(111)
        if self._slit is not None:
            x, y = self._slit
            f = slit_mod.slit_stats(x, y)["fwhm"]
            ax.plot(x, y / y.max(), "-", color="C0", label="Slit, FWHM %.3f nm" % f)
            ax.axhline(0.5, color="C7", lw=PUB_FIT_LW, ls="--", label="Half maximum")
            ax.set_xlim(max(x[0], -3 * f), min(x[-1], 3 * f))
        pub_axes(ax, "Offset [nm]", "Slit (peak = 1) [-]"); pub_legend(ax)
        self.slit_plot.draw()

    # -- simulation ----------------------------------------------------------
    def simulate_spectrum(self):
        if self._slit is None:
            QMessageBox.information(self, "Simulate", "Set a slit function first."); return
        self._xs_files.update(xsm.scan_folder())
        if not self._xs_files:
            xsm.write_synthetic_set()
            self._xs_files.update(xsm.scan_folder())
        dlg = SimulateDialog(self._xs_files, self._settings, self._slit_src, self._path_desc(), self)
        if dlg.exec_() != QDialog.Accepted:
            return
        v = dlg.values()
        if not v["gases"] or v["hi"] <= v["lo"]:
            QMessageBox.information(self, "Simulate", "Tick at least one gas and give a wavelength range."); return
        s = self._settings
        s.update(sim_lo=v["lo"], sim_hi=v["hi"], sim_pixel=v["pixel"], sim_noise=v["noise"], sim_shift=v["shift"],
                 sim_visibility=v["visibility"], sim_seed=v["seed"], sim_ppb={g: c for _p, g, c in v["gases"]})
        self._save_settings()
        T, P, L, slit_xy, slit_src = self.spin_T.value(), self.pressure_pa(), self.path_cm(), self._slit, self._slit_src

        def job():
            gases = [(xsm.load_xs(p), c) for p, _g, c in v["gases"]]
            wl, I0, I, info = sim.simulate(gases, v["lo"], v["hi"], slit_xy, T, P, L, v["pixel"], v["noise"],
                                           v["shift"], v["visibility"], seed=v["seed"])
            sim.write(v["out"], wl, I0, I, info, slit_src, T, P, L, v["noise"], v["shift"], v["visibility"],
                      v["seed"])
            return {"path": v["out"], "gases": [(x["gas"], c) for x, c in gases], "paths": [p for p, _g, _c in v["gases"]]}

        self._start("simulation", job, self._on_simulated, "Simulating…")

    def _on_simulated(self, _job, r):
        self.log("simulated %s: %s" % (r["path"], ", ".join("%s %g ppb" % gc for gc in r["gases"])))
        if self.load_spectrum_file(r["path"]):
            self._set_gases(r["paths"])
            self._region_full()
            self.statusBar().showMessage("Simulated %s - Retrieve to analyse it" % os.path.basename(r["path"]))

    # -- retrieval -----------------------------------------------------------
    def _missing(self):
        miss = []
        if self._wl is None: miss.append("a spectrum")
        elif self.optical_depth() is None: miss.append("the reference I₀")
        if not self._xs: miss.append("cross sections (Select Gases)")
        if self._slit is None: miss.append("a slit function")
        return miss

    def _update_status(self):
        miss = self._missing()
        self.lbl_status.setText("Still needed: %s." % ", ".join(miss) if miss else
                                "Ready: set the fit window and the light path, then Retrieve Concentration.")

    def run_retrieval(self):
        miss = self._missing()
        if miss:
            QMessageBox.information(self, "Retrieve", "Still needed: %s." % ", ".join(miss)); return
        reg = self.region()
        if reg is None:
            QMessageBox.information(self, "Retrieve", "The fit window is empty."); return
        wl, od = self._wl.copy(), self.optical_depth().copy()
        npts = int(((wl >= reg[0]) & (wl <= reg[1])).sum())
        if npts < len(self._xs) + int(self._settings["poly_order"]) + 5:
            QMessageBox.information(self, "Retrieve", "Too few pixels (%d) in the fit window." % npts); return
        T_K, P_Pa, L_cm = self.spin_T.value(), self.pressure_pa(), self.path_cm()
        xs_list, slit_xy, s = list(self._xs), self._slit, dict(self._settings)
        self._save_settings()
        self.log("retrieve %s, %.2f–%.2f nm (%d pixels), %s, gases %s, polynomial %d%s"
                 % (os.path.basename(self._spec_path), reg[0], reg[1], npts, self._path_desc(),
                    " + ".join(x["gas"] for x in xs_list), int(s["poly_order"]), ", shift fitted" if s["fit_shift"] else ""))
        self._start("retrieval", lambda: retrieve(wl, od, xs_list, slit_xy, reg, T_K, P_Pa, L_cm, s),
                    self._on_retrieved, "Fitting %d gas%s…" % (len(xs_list), "es" if len(xs_list) > 1 else ""))

    def _on_retrieved(self, _job, r):
        r["spectrum"], r["slit_src"] = self._spec_path, self._slit_src
        r["xs_paths"] = {x["gas"]: x["source"] for x in self._xs}
        self._result = r
        self._show_result(r)
        if len(r["gases"]) > 1 and self.combo_view.currentData() == "int":
            self.combo_view.setCurrentIndex(0)
        self._redraw(True)
        u = r["unit"]
        txt = "   ".join("%s %s %s" % (g, fmt(r["mix"][g], r["mix_err"][g], 5), u) for g in r["gases"])
        self.log("→ %s   shift %+.4f nm   rms %.3g   (%.3f s)" % (txt, r["shift"], r["rms"], r["elapsed_s"]))
        for w in r["warnings"]:
            self.log("   warning: %s" % w)
        self.lbl_status.setText("Done in %.3f s: %s." % (r["elapsed_s"], txt))
        self.statusBar().showMessage("Retrieved %s" % txt)

    def _ref_ppb(self, gas):
        return self._ref_gases.get(gas)

    def _show_result(self, r):
        L = self.lbl_res
        if r is None:
            self._set_gas_rows([])
            self.tbl_gas.setRowCount(0)
            for w in L.values(): w.setText("—")
            return
        u, f = r["unit"], doas.UNITS[r["unit"]]
        rows = []
        for g in r["gases"]:
            ref = self._ref_ppb(g)
            note = ""
            if ref:
                refu = ref * 1e-9 * f
                note = "true %g %s  (%+.2f %%, %+.1fσ)" % (refu, u, 100 * (r["mix"][g] - refu) / refu,
                                                         (r["mix"][g] - refu) / r["mix_err"][g] if r["mix_err"][g] else 0)
            rows.append((g, "%s %s" % (fmt(r["mix"][g], r["mix_err"][g], 5), u), note))
        self._set_gas_rows(rows)
        t = self.tbl_gas
        t.setRowCount(0)
        for g in r["gases"]:
            i = t.rowCount(); t.insertRow(i)
            for c, txt in enumerate((g, "%.4e" % r["scd"][g], "%.2e" % r["scd_err"][g], "%.5g %s" % (r["mix"][g], u),
                                     "%.2g" % r["mix_err"][g], "%.2e" % r["peak_diff_od"][g])):
                t.setItem(i, c, table_cell(txt))
        L["Slant columns"].setText("\n".join("%-5s %s molec/cm²" % (g, fmt(r["scd"][g], r["scd_err"][g], 6))
                                             for g in r["gases"]))
        L["Number density"].setText("\n".join("%-5s %s molec/cm³" % (g, fmt(r["n"][g], r["n_err"][g], 5))
                                              for g in r["gases"]))
        refs = []
        for g in r["gases"]:
            ref = self._ref_ppb(g)
            if ref:
                refu = ref * 1e-9 * f
                refs.append("%-5s true %g %s → %+.3f %% (%+.2f σ)" % (
                    g, refu, u, 100 * (r["mix"][g] - refu) / refu, (r["mix"][g] - refu) / r["mix_err"][g]))
        L["Reference"].setText("\n".join(refs) or "— (not a simulated spectrum)")
        L["Detection limit"].setText("   ".join("%s %.3g %s" % (g, 2 * r["mix_err"][g], u) for g in r["gases"]))
        G = len(r["gases"])
        L["Correlation"].setText("\n".join("%s–%s  r = %+.3f" % (r["gases"][i], r["gases"][j], r["corr"][i, j])
                                           for i in range(G) for j in range(i + 1, G)) or "— (one gas)")
        sh = ("%+.5f ± %.2g nm" % (r["shift"], r["shift_err"])) if r["fit_shift"] else "0 (fixed)"
        sq = ("   squeeze %+.3g ± %.2g" % (r["squeeze"], r["squeeze_err"])) if r["fit_squeeze"] else ""
        L["Shift / squeeze"].setText(sh + sq)
        L["Polynomial"].setText("order %d:  %s" % (r["poly_order"], "  ".join("%.4g" % c for c in r["poly_coef"])))
        L["RMS / χ²"].setText("%.4g   /   %.4g   ·   R² %.6f" % (r["rms"], r["chi2_red"], r["r2"]))
        L["Air"].setText("N_air %.5g molec/cm³   ·   L %.6g cm" % (r["N_air"], r["L_cm"]))
        L["Points / time"].setText("%d pixels, %d dof, %.3f s" % (r["npts"], r["dof"], r["elapsed_s"]))
        L["Warnings"].setText("\n".join(r["warnings"]) or "none")

    def _redraw(self, keep_view=False):
        xlim = None
        if keep_view and self.plot.fig.axes and self._wl is not None and self.combo_view.currentData() not in STACKED:
            xlim = self.plot.fig.axes[0].get_xlim()
        show = {k: cb.isChecked() for k, cb in self.chk_show.items()}
        build_main_figure(self.plot.fig, self.combo_view.currentData(), self._plot_data(), self.region(),
                          self._result, show, keep_xlim=xlim)
        self.plot.draw()
        if not keep_view:
            self._draw_xs()                 # the fit window may have changed

    # -- examples ------------------------------------------------------------
    def load_example(self, which="single"):
        ex = sim.EXAMPLES[which]
        try:
            path = sim.make_example(which, overwrite=False)
        except Exception as e:
            QMessageBox.warning(self, "Example", "Could not make the example: %s" % e); return
        kind, params = sim.EXAMPLE_SLIT
        self._set_synthetic_slit(kind, params, quiet=True)
        if not self.load_spectrum_file(path):
            return
        self._set_gases([xsm.default_path(g) for g in ex["gases"]])
        self._set_region(*ex["fit"])
        self.combo_view.setCurrentIndex(self.combo_view.findData("gas" if which == "multi" else "od"))
        self.log("example %s: %s, fit window %.1f–%.1f nm" % (which, ", ".join(
            "%s %g ppb" % kv for kv in ex["gases"].items()), *ex["fit"]))
        self._redraw()
        self.run_retrieval()

    # -- saving --------------------------------------------------------------
    def save_result(self):
        r = self._result
        if r is None:
            QMessageBox.information(self, "Save", "Retrieve a concentration first."); return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path, _ = QFileDialog.getSaveFileName(self, "Save result", os.path.join(
            self._settings["last_save_dir"] or results_dir(), "doas_result_%s.txt" % stamp),
            "Text (*.txt);;All files (*)")
        if not path:
            return
        self._remember_dir("last_save_dir", path)
        try:
            write_report(path, r, self._ref_gases)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", "%s: %s" % (type(e).__name__, e)); return
        self.log("saved %s" % path)
        self.statusBar().showMessage("Saved %s" % os.path.basename(path))

    def export_curves(self):
        r = self._result
        if r is None:
            QMessageBox.information(self, "Export", "Retrieve a concentration first."); return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path, _ = QFileDialog.getSaveFileName(self, "Export curves", os.path.join(
            self._settings["last_save_dir"] or results_dir(), "doas_curves_%s.txt" % stamp),
            "Text (*.txt);;All files (*)")
        if not path:
            return
        self._remember_dir("last_save_dir", path)
        try:
            write_curves(path, r)
        except Exception as e:
            QMessageBox.critical(self, "Export failed", "%s: %s" % (type(e).__name__, e)); return
        self.log("exported %s" % path)

    def save_plot(self):
        if self._wl is None:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path, _ = QFileDialog.getSaveFileName(self, "Save plot", os.path.join(
            self._settings["last_save_dir"] or results_dir(), "doas_plot_%s.png" % stamp),
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
        if not path:
            return
        self._remember_dir("last_save_dir", path)
        try:
            # the current view, drawn again at a fixed size in the same publication style
            view = self.combo_view.currentData()
            show = {k: cb.isChecked() for k, cb in self.chk_show.items()}
            fig = export_figure(view, self._result, self._plot_data(), self.region(), show)
            pub_savefig(fig, path, dpi=600 if path.lower().endswith(".png") else 300)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", "%s: %s" % (type(e).__name__, e)); return
        self.log("saved %s" % path)

    # -- help ----------------------------------------------------------------
    def show_about(self):
        mb = QMessageBox(self)
        mb.setWindowTitle("About")
        mb.setTextFormat(Qt.RichText)
        mb.setTextInteractionFlags(Qt.TextBrowserInteraction)
        mb.setText((
            "<b>%s</b><br><br>"
            "Trace-gas slant columns and path-averaged concentrations from a UV spectrum by "
            "Differential Optical Absorption Spectroscopy: cross sections convolved with the slit "
            "function, a broadband polynomial, and a linear least-squares fit.<br><br>"
            "Developed inspired by:<br>"
            "Trisna, B. A., et al.: Measurement report: Radiative efficiencies of "
            "(CF<sub>3</sub>)<sub>2</sub>CFCN, CF<sub>3</sub>OCFCF<sub>2</sub>, and "
            "CF<sub>3</sub>OCF<sub>2</sub>CF<sub>3</sub>, Atmos. Chem. Phys., 23, 4489, 2023, "
            "<a href=\"https://acp.copernicus.org/articles/23/4489/2023/\">"
            "https://acp.copernicus.org/articles/23/4489/2023/</a><br><br>"
            "<b>If you use this code for a publication, you must cite this paper.</b><br><br>"
            "Modules: doas · cross_section · slit · simulate_spectrum.<br>"
            "Method: Platt &amp; Stutz, <i>Differential Optical Absorption Spectroscopy</i>, Springer, 2008.<br>"
        ) % APP_TITLE)
        mb.exec_()

    def show_method(self):
        QMessageBox.information(self, "Method && Caveats", (
            "<b>Model</b><br>"
            "ln(I₀/I) = Σ<sub>i</sub> SCD<sub>i</sub>·σ<sub>eff,i</sub>(λ + shift) + P(λ), with "
            "σ<sub>eff</sub> = σ<sub>high-res</sub> ⊛ slit and P a Legendre polynomial. For a fixed shift "
            "the model is linear: one least-squares solve gives every SCD and its error. The shift is "
            "found by an outer search, each step a linear solve.<br><br>"
            "<b>Broadband and differential</b><br>"
            "Rayleigh and Mie extinction, lamp and throughput changes and the broad part of the cross "
            "sections are all smooth in λ and go into P(λ). Only the narrow, differential structure "
            "σ' identifies the gases - the 'Differential per gas' view shows each one's.<br><br>"
            "<b>Concentration</b><br>"
            "n = SCD / L and x = n / N<sub>air</sub>, N<sub>air</sub> = P / k<sub>B</sub>T: path "
            "average over the TOTAL light path (twice the distance for a retro-reflector).<br><br>"
            "<b>Approximation</b><br>"
            "Convolving σ instead of the intensity holds for τ ≪ 1. For strong absorbers or a "
            "structured I₀ (sunlight: Fraunhofer lines) the I₀-effect correction is needed; it is not "
            "applied here. The cross sections should be at the right temperature and in the same "
            "wavelength scale (air/vacuum) as the spectrometer.<br><br>"
            "<b>Uncertainty</b><br>"
            "± is one standard error from the fit residual. The residuals of a real spectrum are "
            "correlated over the slit width, so the true error is larger (Stutz &amp; Platt 1996: "
            "up to ~3×); the cross sections' own uncertainty and L, T, P add to it."))

    def closeEvent(self, e):
        try:
            if self._worker is not None and self._worker.isRunning():
                self._worker.wait(10000)
        except RuntimeError:
            pass
        self._save_settings()
        super().closeEvent(e)


# =============================================================================
# Reports
# =============================================================================
def write_report(path, r, ref_gases=None):
    ref_gases = ref_gases or {}
    u, f = r["unit"], doas.UNITS[r["unit"]]
    rows = [("date", datetime.now().isoformat(timespec="seconds")),
            ("spectrum", r.get("spectrum", "")), ("slit", r.get("slit_src", "")),
            ("window_nm", "%.4f %.4f" % r["region"]),
            ("T_K", "%.6g" % r["T_K"]), ("P_Pa", "%.8g" % r["P_Pa"]), ("L_cm", "%.6g" % r["L_cm"]),
            ("N_air_cm-3", "%.6g" % r["N_air"]), ("poly_order", "%d" % r["poly_order"]),
            ("shift_nm", ("%.6g ± %.2g" % (r["shift"], r["shift_err"])) if r["fit_shift"] else "0 (fixed)"),
            ("squeeze", ("%.6g ± %.2g" % (r["squeeze"], r["squeeze_err"])) if r["fit_squeeze"] else "0 (fixed)"),
            ("poly_coef", " ".join("%.8g" % c for c in r["poly_coef"])),
            ("rms", "%.6g" % r["rms"]), ("chi2_red", "%.6g" % r["chi2_red"]), ("r2", "%.8f" % r["r2"]),
            ("pixels", "%d" % r["npts"]), ("warnings", "; ".join(r["warnings"]) or "none")]
    for g in r["gases"]:
        rows += [("xs_%s" % g, r.get("xs_paths", {}).get(g, "")),
                 ("scd_%s_cm-2" % g, "%.8g ± %.4g" % (r["scd"][g], r["scd_err"][g])),
                 ("n_%s_cm-3" % g, "%.8g ± %.4g" % (r["n"][g], r["n_err"][g])),
                 ("%s_%s" % (u, g), "%.8g ± %.4g" % (r["mix"][g], r["mix_err"][g])),
                 ("diff_od_%s" % g, "%.4g" % r["peak_diff_od"][g])]
        if g in ref_gases:
            rows.append(("reference_%s_%s" % (u, g), "%g" % (ref_gases[g] * 1e-9 * f)))
    G = r["gases"]
    for i in range(len(G)):
        for j in range(i + 1, len(G)):
            rows.append(("corr_%s_%s" % (G[i], G[j]), "%+.4f" % r["corr"][i, j]))
    with open(path, "w", encoding="utf-8") as fh:
        for k, v in rows:
            fh.write("%-24s %s\n" % (k, v))


def write_curves(path, r):
    cols = [r["x"], r["od"], r["fit"], r["poly"], r["residual"]]
    names = ["wavelength_nm", "od_measured", "od_model", "polynomial", "residual"]
    for g in r["gases"]:
        cols += [r["diff_meas"][g], r["components"][g]]
        names += ["diff_od_measured_%s" % g, "diff_od_fitted_%s" % g]
    # the cascade: total differential OD, then what remains after each gas is subtracted
    steps = cascade_steps(r)
    cols += [steps[0][1], steps[0][2]]
    names += ["diff_od_total", "diff_od_total_fitted"]
    for _title, data, _model, removed in steps[1:]:
        cols.append(data)
        names.append("diff_od_minus_%s" % "_".join(removed))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# %s   %s\n" % (os.path.basename(r.get("spectrum", "")), "  ".join(
            "%s %s %s" % (g, fmt(r["mix"][g], r["mix_err"][g]), r["unit"]) for g in r["gases"])))
        fh.write("# " + "\t".join(names) + "\n")
        for row in zip(*cols):
            fh.write("\t".join("%.6g" % v for v in row) + "\n")


# =============================================================================
# Headless self-test
# =============================================================================
def self_test(save_dir=None):
    """Both worked examples, retrieved without the window; returns the results."""
    out = {}
    kind, params = sim.EXAMPLE_SLIT
    sx, sy = slit_mod.synthetic_slit(kind, params)
    s = dict(DEFAULTS)
    for which, ex in sim.EXAMPLES.items():
        path = sim.make_example(which, overwrite=False)
        data, header = xsm.read_columns(path)
        true, (T, P, L) = read_sim_header(header)
        wl, I0, I = data[:, 0], data[:, 1], data[:, 2]
        xs = [xsm.load_xs(xsm.default_path(g)) for g in ex["gases"]]
        r = retrieve(wl, doas.optical_depth(I, I0), xs, (sx, sy), ex["fit"], T, P, L, s)
        r["spectrum"], r["slit_src"] = path, slit_mod.describe(kind, params)
        print("%s example (%s, %.0f-%.0f nm, shift %+.4f nm, rms %.2e, %.3f s):" % (
            which, os.path.basename(path), ex["fit"][0], ex["fit"][1], r["shift"], r["rms"], r["elapsed_s"]))
        for g in r["gases"]:
            print("   %-5s true %7.3f ppb   retrieved %s ppb   (%+.2f %%, %+.2f σ)   SCD %.4e" % (
                g, true[g], fmt(r["mix"][g], r["mix_err"][g], 5), 100 * (r["mix"][g] - true[g]) / true[g],
                (r["mix"][g] - true[g]) / r["mix_err"][g], r["scd"][g]))
        steps = cascade_steps(r)
        print("   cascade: total differential OD RMS %.3g -> %s -> residual; |last step - residual| max %.1e" % (
            np.sqrt(np.mean(steps[0][1] ** 2)), " -> ".join("%.3g" % np.sqrt(np.mean(d ** 2)) for _t, d, _m, _g in steps[1:-1])
            or "-", np.max(np.abs(steps[-1][1] - r["residual"]))))
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            base = os.path.join(save_dir, "example_%s" % which)
            write_report(base + "_result.txt", r, true)
            write_curves(base + "_curves.txt", r)
            data = {"wl": wl, "od": doas.optical_depth(I, I0), "I": I, "I0": I0}
            for view in ("od", "cascade", "gas"):
                fig = export_figure(view, r, data, ex["fit"], {"data": True, "fit": True, "poly": True,
                                                               "residual": True})
                for ext in ("png", "pdf"):
                    pub_savefig(fig, base + "_%s.%s" % (view, ext), dpi=300)
        out[which] = r
    return out


def monte_carlo(which="multi", n=50):
    """The example simulated again with n noise seeds: the mean deviation
    (bias) and the spread of (retrieved - true) / error, which is 1 when the
    fit's error bars are right."""
    ex = sim.EXAMPLES[which]
    kind, params = sim.EXAMPLE_SLIT
    sx, sy = slit_mod.synthetic_slit(kind, params)
    p = sim.EXAMPLE_PATH
    gases = [(sim.resolve_gas(g), ppb) for g, ppb in ex["gases"].items()]
    xs = [x for x, _ in gases]
    refs = doas.make_references(xs, sx, sy, *ex["fit"])
    mix, pull = [], []
    for seed in range(n):
        wl, I0, I, _ = sim.simulate(gases, ex["lo"], ex["hi"], (sx, sy), p["T_K"], p["P_Pa"], p["L_cm"],
                                    noise=ex["noise"], seed=seed)
        r = doas.fit_doas(wl, doas.optical_depth(I, I0), refs, ex["fit"], T_K=p["T_K"], P_Pa=p["P_Pa"],
                          L_cm=p["L_cm"], **fit_options(DEFAULTS))
        mix.append([r["mix"][x["gas"]] for x in xs])
        pull.append([(r["mix"][x["gas"]] - c) / r["mix_err"][x["gas"]] for x, c in gases])
    mix, pull = np.array(mix), np.array(pull)
    print("Monte Carlo, %s example, %d noise seeds:" % (which, n))
    for k, (x, c) in enumerate(gases):
        print("   %-5s true %7.3f ppb   mean %7.3f ppb (bias %+.2f %% ± %.2f %%)   scatter %.3f ppb   "
              "pull std %.2f" % (x["gas"], c, mix[:, k].mean(), 100 * (mix[:, k].mean() - c) / c,
                                 100 * mix[:, k].std() / np.sqrt(n) / c, mix[:, k].std(), pull[:, k].std()))
    return mix, pull


def main():
    if "--selftest" in sys.argv:
        self_test(results_dir() if "--save" in sys.argv else None)
        if "--montecarlo" in sys.argv:
            for which in sim.EXAMPLES:
                monte_carlo(which)
        return
    app = QApplication(sys.argv)
    w = DoasWindow(); w.show()
    files = [p for p in sys.argv[1:] if os.path.isfile(p)]
    if files:
        w.load_spectrum_file(files[0])
    elif "--no-example" not in sys.argv and w._settings.get("example_on_start", True):
        QTimer.singleShot(0, lambda: w.load_example("single"))
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
