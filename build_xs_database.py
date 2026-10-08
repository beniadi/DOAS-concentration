# -*- coding: utf-8 -*-
"""Build the cross-section database in Input/CrossSections from the
laboratory collection in Input/modataXs.

    python build_xs_database.py            # (re)builds Input/CrossSections/<gas>/*.xs + catalog

Every file of the collection is listed in FILES below with what it holds:
the gas, temperature, reference, unit scale, wavelength axis and scale
(air/vacuum), whether it is an absolute or a differential cross section.
The metadata come from the files' own headers and from the collection's
index (modata.txt); where a header is missing that is said in the note.
A file is written to the database only if, after the stated conversions,
its values are those of a cross section (cm2/molecule, or cm5/molecule2 for
the O4 collision pair) on a wavelength axis; everything else - lamp lines,
solar and lunar spectra, filter curves, relative spectra, data in
plotter-millimetres or extinction units, files without a wavelength axis -
is listed in excluded.csv with the reason.

Output, in Input/CrossSections:

    <GAS>/<GAS>_<T>K_<source file>.xs   two columns: air wavelength (nm), sigma
    catalog.csv, catalog.json           one row per data set
    excluded.csv                        every file not used, and why

Conversions applied (each written into the file's header): unit scale
factors (x1e-18 ...), wavenumber -> wavelength, vacuum -> air wavelength
(Edlen) where the source is in vacuum, ascending order, repeated wavelengths
averaged, "no data" markers (0 or 9999 in multi-temperature tables) removed.
Exact duplicates (the same data in several files or archives) are kept once.
"""

import csv
import hashlib
import io
import json
import os
import re
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cross_section as xsm                                # noqa: E402

SRC = os.path.join(HERE, "Input", "modataXs")
OUT = os.path.join(HERE, "Input", "CrossSections")

# references used more than once
R = {
    "trost": "B. Trost, Institut fuer Umweltphysik Heidelberg, 1994 (aromatic hydrocarbons)",
    "hd_arom": "Institut fuer Umweltphysik Heidelberg, ACTON spectrometer 1996-2001 (aromatic hydrocarbons; "
               "cf. Etzkorn et al., Atmos. Environ. 33, 525, 1999)",
    "mpi": "MPI Mainz",
    "wahner": "A. Wahner et al. (BrO), data received 9 Mar 1990",
    "bremen_o3": "S. Voigt, J. Orphal, K. Bogumil, J.P. Burrows, J. Photochem. Photobiol. A 143, 1 (2001); IUP Bremen FTS",
    "bremen_no2": "S. Voigt, J. Orphal, J.P. Burrows, J. Photochem. Photobiol. A 149, 1 (2002); IUP Bremen FTS",
    "gome": "J.P. Burrows et al., JQSRT 61, 509 (1999) / A. Richter, IUP Bremen: GOME flight-model spectra",
    "frost_no2": "G.J. Frost et al., NO2 at 200+/-20 K (JGR, submitted 1996; N2O4 removed)",
    "frost_oclo": "G.J. Frost, L.M. Goss, V. Vaida, J. Geophys. Res. 101, 3879 (1996)",
    "kromminga": "H. Kromminga, J. Orphal, S. Voigt, J.P. Burrows, IUP Bremen FTS (1999)",
    "harder": "J.W. Harder, J.W. Brault, P.V. Johnston, G.H. Mount, J. Geophys. Res. 102, 3861 (1997)",
    "wahner_oclo": "A. Wahner et al., J. Phys. Chem. 91, 2734 (1987)",
    "gillotay": "D. Gillotay, P.C. Simon, L. Dierickx, Quadrennial Ozone Symposium 1988",
    "simon": "P.C. Simon et al., 1988",
    "mcgee": "T.J. McGee, J. Burris Jr., JQSRT 37, 165 (1987)",
    "freeman": "K. Yoshino, D.E. Freeman et al. (MPI file 1990)",
    "cl2_ab": "R.G. Aickin, N.S. Bayliss, Trans. Faraday Soc. 33, 1333 (1937)",
    "cl2_gb": "G.E. Gibson, N.S. Bayliss, Phys. Rev. 44, 188 (1933)",
    "cl2_bb": "J.B. Burkholder, E.J. Bair, J. Phys. Chem. 87, 1859 (1983)",
    "seery": "D.J. Seery, D. Britton, J. Phys. Chem. 68, 2263 (1964)",
    "stutz": "J. Stutz, E.S. Kim, U. Platt et al., J. Geophys. Res. 105, 14585 (2000)",
    "cantrell": "C.A. Cantrell, J.A. Davidson, A.H. McDaniel, R.E. Shetter, J.G. Calvert, J. Phys. Chem. 94, 3902 (1990)",
    "meller": "R. Meller, G.K. Moortgat, J. Geophys. Res. 105, 7089 (2000)",
    "sander_no3": "S.P. Sander, J. Phys. Chem. 90, 4135 (1986)",
    "brion": "J. Brion, J. Malicet et al., GSMA Reims (O3 temperature series)",
    "harwood": "M.H. Harwood, R.L. Jones, J. Geophys. Res. 99, 22955 (1994)",
    "merienne": "M.F. Merienne, A. Jenouvrier, B. Coquart, J. Atmos. Chem. 20, 281 (1995); 21, 251 (1995)",
}

# what is not a cross section, by file-name pattern (checked first)
NOT_XS = [
    (r"^lines|\.ref$|^lines\(nist\)", "emission-line list / lamp reference, not a cross section"),
    (r"^fil(rg|ug)", "glass filter transmission curve"),
    (r"^albedo", "surface albedo (soil), not a cross section"),
    (r"^nadir", "LOWTRAN-7 radiance, not a cross section"),
    (r"^moon", "lunar reflectance spectrum"),
    (r"^(solarfl|solar_fl|solkur|kurutz|sol_info)", "solar spectrum, not a cross section"),
    (r"^hglamp|aromate/hg\d|hg\d\.xls", "Hg lamp spectrum (wavelength calibration)"),
    (r"\.(doc|rtf|pdf|opj|grf|exe|ini|log|com|dir|tsk|ftn|slo|1st)$", "document / program / plot file"),
    (r"^modata\.|^no2_xxxx|^readme|^no2hlp|stutz\.txt|esa-wv-readme|^te$|^35dmba|omaexi|^copy",
     "text / index / readme file"),
    (r"aromate/std/", "OMA spectrum without wavelength axis (the same compounds are in aromate/xls with wavelengths)"),
]

S = lambda **k: k            # noqa: E731  (one data set)
TROST_NOTE = ("the absolute level differs from the later Heidelberg set (aromate/xls) by an offset; "
              "the narrow structure agrees for the xylenes but not for benzene and benzaldehyde")

# ---------------------------------------------------------------------------
# Every file of the collection (path relative to Input/modataXs, lower case).
#   gas, T, ref       what it is
#   scale             multiply sigma by this (files in units of 1e-18 cm2 ...)
#   x                 "nm" or "cm-1" (default: detected)
#   vac               True: vacuum wavelengths -> converted to air
#   col               sigma column (default: first column that looks like a cross section)
#   temps             [(column, T), ...]: one data set per temperature column
#   missing           values that mean "no data"
#   kind              "absolute" (default) or "differential" (broadband removed / relative scale)
#   unit              "cm2/molecule" (default) or "cm5/molecule2" (O4)
#   reader            special file layouts
#   parts             several files of one data set, joined
#   exclude           the reason it is not in the database
# ---------------------------------------------------------------------------
FILES = {
    # ---- SO2 ----------------------------------------------------------------
    "so2vandaele": S(gas="SO2", T=295, ref="A.C. Vandaele et al., J. Geophys. Res. 99, 25599 (1994), FTS"),
    "so2.dat": S(gas="SO2", T=None, ref="MPI compilation 1990 (Brassington 1981 + McGee 1987)"),
    "so2210.dat": S(gas="SO2", T=210, ref=R["mcgee"]),
    "so2295.dat": S(gas="SO2", T=295, ref=R["mcgee"]),
    "so2bra.dat": S(gas="SO2", T=None, ref="D.J. Brassington, Appl. Opt. 20, 3774 (1981), dye laser",
                    note="room temperature (not stated)"),
    "so2fre.005": S(gas="SO2", T=213, ref=R["freeman"], note="0.005 nm data set"),
    "so2fre.015": S(gas="SO2", T=213, ref=R["freeman"], note="0.015 nm data set"),
    "so2fre.200": S(gas="SO2", T=213, ref=R["freeman"], note="0.2 nm data set"),
    "so2tho.dat": S(gas="SO2", T=None, ref="R.T. Thompson Jr., J. Appl. Phys. 46, 3040 (1975)"),
    "so2woo.dat": S(gas="SO2", T=293, ref="P.T. Woods, B.W. Jolliffe, B.R. Marx, Opt. Commun. (1980), SO2 in N2"),
    "so2woo.org": S(gas="SO2", T=293, ref="P.T. Woods et al., Opt. Commun. (1980), original table", scale=1e-19),
    "so2war.dat": S(exclude="values are plotter millimetres (Y-unit mm) without a conversion factor"),
    "so2_brmg.std": S(exclude="OMA spectrum without wavelength axis"),
    # ---- O3 -----------------------------------------------------------------
    "o3.dat": S(gas="O3", T=None, ref="MPI compilation (WMO 1975, Bass, Johnston)"),
    "o3.org": S(gas="O3", T=None, ref="MPI compilation (WMO 1975), original 1 nm table"),
    "o3nasa.dat": S(gas="O3", T=None, ref="WMO Report 1975 (NASA)"),
    "o3bas.dat": S(gas="O3", T=250, ref="A.M. Bass, R.J. Paur et al. (computer program)"),
    "o3john.dat": S(gas="O3", T=None, ref="P.V. Johnston, unpublished (visible)"),
    "o3graham.dat": S(gas="O3", T=None, ref="Graham & Johnston (received from P.V. Johnston 1996)", scale=1e-22),
    "o3and1.dat": S(gas="O3", T=298, ref="S. Anderson, Augsburg College (1994, Chappuis/Wulf bands)", scale=1e-23),
    "o3c220.dat": S(gas="O3", T=220, ref="M. Cacciani et al., J. Geophys. Res. 94, 8485 (1989)", scale=1e-20),
    "o3c293.dat": S(gas="O3", T=293, ref="M. Cacciani et al., J. Geophys. Res. 94, 8485 (1989)", scale=1e-20),
    "o3dat.std": S(gas="O3", T=None, ref="unknown (index: 'O3 ?', two columns without information)",
                   note="identity uncertain"),
    "o3brion.dat": S(gas="O3", ref=R["brion"], temps=[(1, 218), (2, 228), (3, 243), (4, 273), (5, 295)],
                     missing=(9999,)),
    "o3mol.dat": S(exclude="single column without wavelengths"),
    **{"o3_%s.dat" % t: S(gas="O3", T=int(t), ref=R["gome"], vac=True) for t in ("202", "221", "241", "273", "293")},
    **{"o3_%s.air" % t: S(gas="O3", T=int(t), ref=R["gome"] + " (preliminary, air wavelengths)")
       for t in ("202", "221", "241", "273", "293")},
    **{"o3_%s%s.dat" % (t, p): S(gas="O3", T=int(t), ref=R["bremen_o3"], vac=True, col=2,
                                 note="%s mbar total pressure" % ("1000" if p == "h" else "100"))
       for t in ("203", "223", "246", "280", "293") for p in "hl"},
    # ---- NO2 ----------------------------------------------------------------
    "no2mpi.dat": S(gas="NO2", T=298, ref="K. Schneider, G.S. Tyndall, J.P. Burrows, G.K. Moortgat, MPI Mainz (1987)"),
    "no2mpi.int": S(gas="NO2", T=298, ref="K. Schneider et al., MPI Mainz (1987), reduced resolution"),
    "no2vanda.294": S(gas="NO2", T=294, ref="A.C. Vandaele et al., FTS, visible (preprint 1996)"),
    "no2reims.dat": S(gas="NO2", ref=R["merienne"], reader="varcols", temps=[(1, 293), (2, 240), (3, 220)]),
    "no2jones.dat": S(gas="NO2", ref=R["harwood"], missing=(0.0,),
                      temps=[(1, 213), (2, 225), (3, 243), (4, 253), (5, 263), (6, 273), (7, 298)]),
    "no2hal.dat": S(gas="NO2", T=298.2, ref="T.C. Hall, F.E. Blacet, J. Chem. Phys. 20, 1745 (1952)"),
    "no2har.dat": S(gas="NO2", T=296, ref="A.B. Harker, W. Ho, J.J. Ratto, Chem. Phys. Lett. 50, 394 (1977)"),
    "no2nak.dat": S(gas="NO2", T=None, ref="T. Nakayama, M.Y. Kitamura, K. Watanabe, J. Chem. Phys. 30, 1180 (1959)"),
    "no2dave.txt": S(gas="NO2", T=None, ref="unknown (no header; the file name suggests Davidson et al.)"),
    "no2c_97.dat": S(gas="NO2", T=None, x="cm-1", vac=True,
                     ref="unknown (no header; FTS, vacuum wavenumbers 15000-42000 cm-1)",
                     note="'c' in the name suggests the cold set of a 1997 cold/room pair"),
    "no2r_97.dat": S(gas="NO2", T=None, x="cm-1", vac=True,
                     ref="unknown (no header; FTS, vacuum wavenumbers 15000-42000 cm-1)",
                     note="'r' in the name suggests the room-temperature set of a 1997 cold/room pair"),
    "no2esa223h_uv.dat": S(gas="NO2", T=223, ref=R["bremen_no2"] + " (UV part, as given)",
                           note="no header; the wavelength scale (air/vacuum) is not stated"),
    "no2_05a.dat": S(gas="NO2", T=200, ref=R["frost_no2"] + ", 0.5 cm-1", scale=1e-18,
                     parts=["no2_05a.dat", "no2_05b.dat"]),
    "no2_05b.dat": S(part_of="no2_05a.dat"),
    "no2_20.dat": S(gas="NO2", T=200, ref=R["frost_no2"] + ", 2 cm-1", scale=1e-18),
    **{"no2_%s.air" % t: S(gas="NO2", T=int(t), ref=R["gome"] + " (air wavelengths)") for t in ("221", "241", "273", "293")},
    **{"no2noaa.%s" % t: S(gas="NO2", T=int(t), ref=R["harder"], reader="noaa", scale=1e-19,
                           note="wavelength scale as given (not stated)") for t in ("217", "230", "238", "294")},
    **{"bremen/no2_%s%s.zip" % (t, p): S(gas="NO2", T=int(t), ref=R["bremen_no2"], vac=True, col=2,
                                         note="%s mbar total pressure" % ("1000" if p == "h" else "100"))
       for t in ("223", "246", "260", "280", "293") for p in "hl"},
    **{"no2_%s%s.zip" % (t, p): S(gas="NO2", T=int(t), ref=R["bremen_no2"] + " (copy in the main folder)", vac=True, col=2,
                                  note="%s mbar total pressure" % ("1000" if p == "h" else "100"))
       for t in ("223", "246", "260", "280", "293") for p in "hl"},
    "no2_280h.dat": S(gas="NO2", T=280, ref=R["bremen_no2"] + " (copy in the main folder)", vac=True, col=2,
                      note="1000 mbar total pressure"),
    "no2bas.org": S(exclude="extinction coefficients (Bass 1976), not cross sections in cm2"),
    "no2235.dat": S(exclude="wavelength column only (the data are printed in the paper)"),
    "no2298.dat": S(exclude="wavelength column only (the data are printed in the paper)"),
    "no2cmp.txt": S(exclude="comparison table, not a spectrum"),
    "gerd/no2hd.dat": S(gas="NO2", T=295, kind="differential",
                        ref="G. Hoenninger, Heidelberg flow-cell NO2 reference matching the IO spectrum (diploma thesis)",
                        note="differential cross section"),
    # ---- HCHO ---------------------------------------------------------------
    "hchocal.dat": S(gas="HCHO", T=296.16, ref=R["cantrell"]),
    "hchocal1.dat": S(gas="HCHO", T=296.16, ref=R["cantrell"] + " (ascending copy)"),
    "hchocal.org": S(exclude="temperature-fit coefficients vs wavenumber (evaluated at 296 K in hchocal.dat)"),
    "l": S(exclude="copy of hchocal.org (temperature-fit coefficients)"),
    "hchomoor.dat": S(gas="HCHO", ref=R["meller"], blocks=[(0, 298, "attachment hchomel.dat"),
                                                            (1, 223, "attachment hcho223x.dat")],
                      block_skip={2: "attachment 3 is the temperature gradient d(sigma)/dT, not a cross section"}),
    "hcho2000.dat": S(gas="HCHO", T=298, ref=R["meller"] + " (224-232 nm part only)"),
    "hchomoo2.dat": S(gas="HCHO", T=298, ref=R["meller"] + " (224-232 nm part only)"),
    "hchompi.dat": S(gas="HCHO", T=297, ref="R. Meller, MPI Mainz (averaged, 0.1 nm)"),
    "hchompi.010": S(gas="HCHO", T=297, ref="R. Meller, MPI Mainz (averaged, 0.1 nm)"),
    "hchomoo.dat": S(gas="HCHO", T=284.7, ref="MPI Mainz, 10 cm cell, 0.5 nm resolution (1980)"),
    "hchobas.dat": S(gas="HCHO", T=295, ref="A.M. Bass et al. (1980)"),
    "hchosny.dat": S(exclude="absorption coefficient in cm-1 atm-1 (vacuum UV, 105-185 nm), base not stated"),
    # ---- HONO ---------------------------------------------------------------
    "stutz_hono.zip": S(gas="HONO", T=296, ref=R["stutz"], member="hono_stutz.dat"),
    "oclowah3.dat": S(gas="OClO", T=204, ref=R["wahner_oclo"] + " (low-temperature file in refmainz.zip)"),
    "hcho2000.xls": S(exclude="Excel copy of the Meller & Moortgat e-mail (data as text in one column; see hchomoor.dat)"),
    "no2mpi.xls": S(gas="NO2", T=298, ref="K. Schneider et al., MPI Mainz (1987), Excel copy (extract)", reader="xls",
                    xls_cols=(0, 1)),
    "hono_hd.dat": S(gas="HONO", T=296, ref=R["stutz"] + " (Heidelberg copy without header)"),
    "no2hono.zip": S(gas="HONO", T=296, ref=R["stutz"] + " (copy in no2hono.zip)", member="hono_hd.dat"),
    "honocnrs.dat": S(gas="HONO", T=None, ref="unknown (CNRS; no further header)"),
    "xx.dat": S(gas="HONO", T=None, ref="unknown (copy of honocnrs.dat?)"),
    "hno2jo.dat": S(exclude="unit uncertain: values 0.2-13 'cm**2' fit no power of ten against the other HONO data"),
    # ---- NO3 ----------------------------------------------------------------
    "no3jpl.dat": S(gas="NO3", T=298, ref="JPL Report 1987 (mean of Ravishankara et al. and others)"),
    "no3s230.dat": S(gas="NO3", T=230, ref=R["sander_no3"]),
    "no3s298.dat": S(gas="NO3", T=298, ref=R["sander_no3"]),
    "no3gj298.dat": S(gas="NO3", T=298, ref="Graham & Johnston (index: 1 nm, 400-704 nm)"),
    "no3san.dat": S(exclude="plotter millimetres (Y-unit mm); the same Sander data are in no3s298.dat"),
    # ---- halogen oxides -------------------------------------------------------
    **{"brow%s.%s" % (t, e): S(gas="BrO", T=int(t), ref=R["wahner"],
                               note="original digitisation" if e == "org" else "")
       for t in ("228", "243", "298") for e in ("dat", "org")},
    "bro.dat": S(gas="BrO", T=298, ref=R["wahner"] + " (copy)"),
    "br_243_4.txt": S(gas="BrO", T=243, ref=R["wahner"] + " (copy)"),
    "gerd/bro1200.dat": S(gas="BrO", T=298, kind="differential", scale=1e-17,
                          ref="Heidelberg cell BrO reference, 1200 l/mm grating, 0.16 nm (G. Hoenninger)",
                          note="differential cross section, atmospheric pressure"),
    "io.dat": S(exclude="arbitrary units (IO, not baseline corrected)"),
    "iofit.dat": S(gas="IO", T=293, kind="differential",
                   ref="A. Richter / F. Wittrock, IUP Bremen (1995), scaled to 2.8e-17 cm2 at 427.2 nm (Laszlo et al. 1995)",
                   note="relative spectrum scaled to one band; use as differential cross section"),
    "iohuie.dat": S(gas="IO", T=None, ref="R.E. Huie et al., NIST (1997)"),
    "gerd/io_hd.dat": S(gas="IO", T=295, kind="differential",
                        ref="G. Hoenninger, Heidelberg flow cell, calibrated to Cox et al., GRL 26, 1857 (1999)",
                        note="differential cross section, air wavelengths, 10 mbar"),
    "gerd/io_xsn.dat": S(gas="IO", T=295, kind="differential", ref="G. Hoenninger, Heidelberg flow cell (copy)"),
    "obro.dat": S(exclude="relative absorption (OBrO), no cross-section unit"),
    "clompi.dat": S(gas="ClO", T=None, ref="P.C. Simon et al., J. Photochem. (1989)"),
    "closan.dat": S(gas="ClO", T=None, ref="S.P. Sander, R.R. Friedl, J. Phys. Chem. (1989)"),
    "cl2ompi.dat": S(gas="Cl2O", T=None, ref="P.C. Simon, PhD thesis (1989)"),
    "cl2o2.dat": S(gas="Cl2O2", T=None, ref="J.B. Burkholder et al., NOAA (1988, preliminary)"),
    **{"oclowah.%s" % t: S(gas="OClO", T=int(t), ref=R["wahner_oclo"]) for t in ("204", "298", "378")},
    "scratch.dat": S(gas="OClO", T=204, ref=R["wahner_oclo"] + " (copy)"),
    "oclo-iup.dat": S(gas="OClO", ref=R["kromminga"], x="cm-1", vac=True,
                      temps=[(1, 293), (2, 273), (3, 253), (4, 233), (5, 213)]),
    "oclo_iupbremen233k.dat": S(gas="OClO", T=233, ref=R["kromminga"] + " (233 K extract)", x="cm-1", vac=True),
    "oclopub233.dat": S(gas="OClO", T=233, ref=R["kromminga"], vac=True),
    "ocloiupb233.dat": S(gas="OClO", T=233, ref=R["kromminga"] + " (233 K copy without header)", vac=True,
                         note="vacuum wavelengths (= 1e7 / vacuum wavenumber)"),
    "oclofro.5_p": S(gas="OClO", T=200, ref=R["frost_oclo"] + ", 0.5 cm-1", scale=1e-17),
    "oclofro1.1_p": S(gas="OClO", T=200, ref=R["frost_oclo"] + ", 0.1 cm-1", scale=1e-17,
                      parts=["oclofro1.1_p", "oclofro2.1_p", "oclofro3.1_p"]),
    "oclofro2.1_p": S(part_of="oclofro1.1_p"),
    "oclofro3.1_p": S(part_of="oclofro1.1_p"),
    "oclofro.dat": S(gas="OClO", T=200, ref=R["frost_oclo"] + ", 0.1 cm-1 (all parts, no header)", scale=1e-17),
    # ---- halogens -------------------------------------------------------------
    "br2cont.dat": S(gas="Br2", T=None, ref="MPI Mainz (1991, preliminary continuum)", note="bad below 360 nm (header)"),
    "br2mcm.dat": S(gas="Br2", T=298.15, ref="V. McMillan, in Calvert & Pitts, Photochemistry p. 569"),
    "br2pc298.dat": S(gas="Br2", T=298, ref="A.A. Passchier, J.D. Christian, N.W. Gregory, J. Phys. Chem. 71, 937 (1967)"),
    "br2red.dat": S(gas="Br2", T=None, ref="MPI Mainz (1991, preliminary, banded part)"),
    "br2sb298.dat": S(gas="Br2", T=298, ref=R["seery"]),
    "brclmar.dat": S(gas="BrCl", T=None, ref="Maric (1992), corrected for Cl2 and Br2"),
    "brclmpi.dat": S(gas="BrCl", T=295, ref="MPI Mainz (1992), Cl2 and Br2 contributions subtracted"),
    **{"cl2ab%s.dat" % t: S(gas="Cl2", T=int(t), ref=R["cl2_ab"]) for t in ("291", "456", "593", "737", "849", "982")},
    **{"cl2gb%s.dat" % t: S(gas="Cl2", T={"038": 1038}.get(t, int(t)), ref=R["cl2_gb"])
       for t in ("038", "291", "441", "554", "699", "853")},
    "cl2bb195.dat": S(gas="Cl2", T=195, ref=R["cl2_bb"]),
    "cl2bb301.dat": S(gas="Cl2", T=301, ref=R["cl2_bb"]),
    "cl2hs293.dat": S(gas="Cl2", T=293, ref="H. von Halban, K. Siedentopf, Z. Elektrochem. 28, 496 (1922)"),
    "cl2js291.dat": S(gas="Cl2", T=291, ref="F.W. Jones, W. Spooner, Trans. Faraday Soc. 31, 811 (1935)"),
    "cl2mar.dat": S(gas="Cl2", T=298, ref="MPI Mainz (1991)"),
    "cl2mpi.int": S(gas="Cl2", T=None, ref="MPI Mainz (1987)"),
    "cl2red.dat": S(gas="Cl2", T=298, ref="MPI Mainz (1989), banded part 476-496 nm"),
    "cl2rox.dat": S(gas="Cl2", T=None, ref="C. Roxlo, A. Mandl, J. Appl. Phys. 51, 2969 (1980)"),
    "cl2sb298.dat": S(gas="Cl2", T=298, ref=R["seery"]),
    # ---- halocarbons ----------------------------------------------------------
    "ch2cl2st.dat": S(gas="CH2Cl2", T=None, ref="C. Hubrich, F. Stuhl, J. Photochem. 12, 93 (1980)"),
    "ch3cl295.dat": S(gas="CH3Cl", T=295, ref=R["simon"]),
    "ch3clrob.dat": S(gas="CH3Cl", T=295, ref="D.E. Robbins (1976)"),
    "ch3clsim.txt": S(gas="CH3Cl", ref=R["simon"], scale=1e-21,
                      temps=[(1, 295), (2, 270), (3, 250), (4, 230), (5, 210)]),
    "chcl3sim.dat": S(gas="CHCl3", T=295, ref=R["simon"]),
    "dbm295.dat": S(gas="CH2Br2", T=295, ref=R["gillotay"]),
    "dbmgil.txt": S(gas="CH2Br2", ref=R["gillotay"], scale=1e-21, reader="varcols",
                    temps=[(1, 295), (2, 270), (3, 250), (4, 230), (5, 210)]),
    "dbmmol.dat": S(gas="CH2Br2", T=None, ref="L.T. Molina, M.J. Molina, F.S. Rowland, J. Phys. Chem. 86, 2672 (1982)"),
    "tbmcox.dat": S(gas="CHBr3", T=None, ref="R.A. Cox, S.A. Penkett (1987)"),
    "tbmgil.dat": S(gas="CHBr3", T=None, ref="D. Gillotay et al., Planet. Space Sci. (1989)"),
    "tbmmpi.dat": S(gas="CHBr3", T=None, ref="MPI Mainz (1990)"),
    "tbmmpi.old": S(gas="CHBr3", T=None, ref="MPI Mainz (1988, earlier version)"),
    "tbmtat.dat": S(exclude="optical density in arbitrary units"),
    "ch2i2_294k_s1.txt": S(gas="CH2I2", T=294, ref="Schneider, von Helden, MPI Mainz (1989)"),
    "dimmpi.int": S(gas="CH2I2", T=None, ref="Schneider, von Helden, MPI Mainz (very preliminary)"),
    "mejred.dat": S(gas="CH3I", T=296.16, ref="MPI Mainz (1989, preliminary)"),
    "mejvuv.dat": S(gas="CH3I", T=296.16, ref="MPI Mainz (1989, preliminary, UV part)"),
    "cof2chou.int": S(gas="COF2", T=None, ref="C.C. Chou et al. (1987), atmospheric intervals"),
    "cof2jpl.int": S(gas="COF2", T=None, ref="Workman et al., J. Chem. Phys. 52, 3204, scaled to JPL"),
    "cof2mol.int": S(gas="COF2", T=None, ref="Molina et al., JPL report 1987, atmospheric intervals"),
    "cof2mpi.dat": S(gas="COF2", T=296, ref="Uni Frankfurt and MPI Mainz (1990)"),
    "cof2wo1.dat": S(gas="COF2", T=295, ref="Workman et al., J. Chem. Phys. 52, 3204, scaled to JPL"),
    "cof2wo2.dat": S(gas="COF2", T=295, ref="Workman et al., detailed figure, scaled to JPL"),
    "psgjpl.dat": S(gas="COCl2", T=None, ref="JPL report and CODATA preferred values"),
    "psgmpi.int": S(gas="COCl2", T=None, ref="Schneider, Helden, Wilson, Moortgat, MPI Mainz"),
    "hcofmpi.dat": S(gas="HCOF", T=None, ref="R. Meller, MPI Mainz (1992)"),
    "tfacmpi.int": S(gas="CF3COCl", T=296, ref="MPI Mainz (1991)"),
    "tfafmpi.int": S(gas="CF3COF", T=296, ref="MPI Mainz (1991)"),
    "cs2mpi.int": S(gas="CS2", T=None, ref="MPI Mainz (1990, preliminary)"),
    "cs2wu.dat": S(gas="CS2", T=None, ref="C.Y.R. Wu, D.L. Judge, Geophys. Res. Lett. 8, 769 (1981)"),
    "cs2wu.org": S(gas="CS2", T=None, ref="C.Y.R. Wu, D.L. Judge (1981), original table"),
    # ---- oxygenated VOCs --------------------------------------------------------
    "acetolmp.dat": S(gas="Acetol", T=296, ref="R. Meller, J. Crowley, MPI Mainz (1990)"),
    "achmpi.dat": S(gas="Acetaldehyde", T=None, ref="MPI Mainz (1986, preliminary)"),
    "achmpi.int": S(gas="Acetaldehyde", T=None, ref="MPI Mainz (1989, interpolated)"),
    "acnmpi.int": S(gas="Acetone", T=None, ref="MPI Mainz (1987, preliminary)"),
    "acnlak.dat": S(exclude="decadic/natural molar extinction (epsilon) - base not stated"),
    "macmpi.dat": S(gas="Methacrolein", T=None, ref="MPI Mainz (1990, preliminary)"),
    "macmpi.int": S(gas="Methacrolein", T=None, ref="MPI Mainz (1990, interpolated)"),
    "macbir.dat": S(exclude="plotter millimetres (Y-unit mm) without a conversion factor"),
    "mvkmpi.dat": S(gas="MVK", T=None, ref="MPI Mainz (1986, preliminary)"),
    "mvkbir.dat": S(gas="MVK", T=300, ref="R. Birge, J. Am. Chem. Soc. 93, 6715 (1971)",
                    note="digitised from a figure"),
    "mekmpi.dat": S(gas="MEK", T=None, ref="MPI Mainz (1988, preliminary)"),
    "methar.dat": S(gas="Methanol", T=295.2, ref="A.J. Harrison et al., J. Chem. Phys. 30, 355 (1959)"),
    "mgmpi.dat": S(gas="Methylglyoxal", T=296, ref="R. Meller, Raber, MPI Mainz (1991)"),
    "mgmpi.int": S(gas="Methylglyoxal", T=296, ref="R. Meller, Raber, MPI Mainz (1991, averaged)"),
    "mgpitts.dat": S(gas="Methylglyoxal", T=None, ref="Plum, Pitts et al., Environ. Sci. Technol. (1983)"),
    "panhen.dat": S(gas="PAN", T=None, ref="Senum et al. (1984), extrapolated by Kanakidou & Hennig (1989)"),
    "panste.dat": S(gas="PAN", T=None, ref="E.R. Stephens (1960)"),
    "panste.org": S(gas="PAN", T=None, ref="E.R. Stephens (1960), original table"),
    "pansen.dat": S(exclude="three values only"),
    # ---- aromatics --------------------------------------------------------------
    **{f: S(gas=g, T=t, ref=R["trost"], note=TROST_NOTE)
       for f, g, t in (("benz_hd.dat", "Benzene", 293.65), ("tol_hd.dat", "Toluene", 293.15),
                       ("o_xyl-hd.dat", "o-Xylene", 294.15), ("m_xyl-hd.dat", "m-Xylene", 293.65),
                       ("p_xyl-hd.dat", "p-Xylene", 293.15), ("phen_hd.dat", "Phenol", 293.65),
                       ("p_kresol.dat", "p-Cresol", 293.15), ("beald_hd.dat", "Benzaldehyde", 293.15))},
    **{"aromate/xls/%s.xls" % f: S(gas=g, T=None, ref=R["hd_arom"], reader="xls",
                                   note="room temperature (not stated)")
       for f, g in (("124tmb", "1,2,4-Trimethylbenzene"), ("135tmb", "1,3,5-Trimethylbenzene"),
                    ("23dmp", "2,3-Dimethylphenol"), ("246tmp", "2,4,6-Trimethylphenol"),
                    ("24dmp", "2,4-Dimethylphenol"), ("25dmp", "2,5-Dimethylphenol"),
                    ("26dmp", "2,6-Dimethylphenol"), ("34dmp", "3,4-Dimethylphenol"),
                    ("35dmp", "3,5-Dimethylphenol"), ("benzalde", "Benzaldehyde"), ("benzene", "Benzene"),
                    ("ethylbe", "Ethylbenzene"), ("mcresol", "m-Cresol"), ("mtolyla", "m-Toluidine"),
                    ("mxylol", "m-Xylene"), ("ocresol", "o-Cresol"), ("otolyla", "o-Toluidine"),
                    ("oxylol", "o-Xylene"), ("pcresol", "p-Cresol"), ("phenol", "Phenol"),
                    ("ptolyla", "p-Toluidine"), ("pxylol", "p-Xylene"), ("styrene", "Styrene"),
                    ("toluene", "Toluene"))},
    # ---- O2, O4, H2O ------------------------------------------------------------
    "o2gree.dat": S(gas="O2", T=296, ref="Greenblatt et al., J. Geophys. Res. 95, 18577 (1990), NOAA"),
    "o2gree.org": S(gas="O2", T=296, ref="Greenblatt et al. (1990), original"),
    "o2yosh.dat": S(gas="O2", T=None, ref="K. Yoshino et al., Schumann-Runge bands"),
    "o2yosh.org": S(gas="O2", T=None, ref="K. Yoshino et al., Schumann-Runge bands (original, wavenumber)",
                    x="cm-1", vac=True),
    "o4283ha.dat": S(gas="O4", T=283, unit="cm5/molecule2", scale=1e-46, vac=True,
                     ref="D. Newnham, Rutherford Appleton Laboratory (1997), FTS 5 cm-1",
                     note="1000 hPa; collision-induced (binary) cross section"),
    "o4_herm.dat": S(gas="O4", T=None, unit="cm5/molecule2",
                     ref="unknown (no header; the file name suggests Hermans et al.)",
                     note="collision-induced (binary) cross section"),
    "o4green.dat": S(exclude="O2 absorbance at 55 atm, relative units"),
    "o4_1s1s.dat": S(exclude="relative units (R. Volkamer 1996)"),
    "o4/o4_100b.dat": S(exclude="relative units, no header"),
    "o4/o4_200b.dat": S(exclude="relative units, no header"),
    "h2o_95.273": S(gas="H2O", T=273, reader="erle", ref="F. Erle (1997), calculated from HITRAN'96",
                    note="95 hPa; 440-680 nm, 0.001 nm"),
    "h2o_100.298": S(exclude="single column without wavelength information"),
    "h2o_hard.dat": S(gas="H2O", T=None, scale=1e-25, ref="J.W. Harder, J.W. Brault, J. Geophys. Res. 102, 6245 (1997)"),
    "h2o/h2oref.dat": S(exclude="no header; smooth values ~1e-30 cm2 at 400-410 nm - identity unclear"),
    "h2o/esawv.dat": S(exclude="no numeric data readable"),
    "h2o/8000-end.par": S(exclude="HITRAN line parameters, not a cross section"),
}
ALIASES = {"oclopub233.dat": None}


# =============================================================================
# Reading
# =============================================================================
_SPLIT = re.compile(r"[,;\t ]+")


def _text_lines(raw):
    return raw.decode("latin-1").replace("\r\n", "\n").replace("\r", "\n").split("\n")


def rows_varcols(lines):
    """Numeric rows with their own column counts, and the header lines."""
    rows, header = [], []
    for ln in lines:
        s = ln.strip().replace('"', "")
        if not s:
            continue
        if s[0] in "#;!%*":
            header.append(s.lstrip("#;!%* ")); continue
        try:
            vals = [float(p.replace("D", "E")) for p in _SPLIT.split(s) if p]
        except ValueError:
            header.append(s); continue
        rows.append(vals)
    if rows and sum(len(r) >= 2 for r in rows) > len(rows) / 2:
        rows = [r for r in rows if len(r) >= 2]
    return rows, header


def read_dataset(raw, spec, name):
    """[(T, x, y, header)] for one file (several for multi-temperature tables)."""
    reader = spec.get("reader")
    if reader == "xls":
        import xlrd
        b = xlrd.open_workbook(file_contents=raw, encoding_override="cp1252")
        sh = b.sheet_by_index(0)
        x, y, blocks = [], [], 0
        cx, cy = spec.get("xls_cols", (1, 2))
        for r in range(sh.nrows):
            try:
                a, s = float(sh.cell_value(r, cx)), float(sh.cell_value(r, cy))
            except (ValueError, TypeError):
                if x: blocks += 1
                continue
            x.append(a); y.append(s)
        hdr = ["%s (Excel sheet, %d rows)" % (" | ".join(str(sh.cell_value(0, c)) for c in range(sh.ncols)), sh.nrows)]
        x, y = np.array(x), np.array(y)
        segs = split_blocks(x)
        if len(segs) > 1:            # spectra at several grating settings: the first whole, then what it lacks
            keep = np.zeros(len(x), bool)
            lo, hi = np.inf, -np.inf
            for i0, i1 in segs:
                xs = x[i0:i1]
                keep[i0:i1] = (xs < lo) | (xs > hi)
                lo, hi = min(lo, xs.min()), max(hi, xs.max())
            x, y = x[keep], y[keep]
            hdr.append("%d grating settings joined: the first in full, the others where it has no data" % len(segs))
        return [(spec.get("T"), x, y, hdr)]
    lines = _text_lines(raw)
    if reader == "noaa":
        hdr = [l for l in lines[:12] if l and not re.match(r"^\s*[-\d.]+\s*$", l)]
        first = float(re.search(r"First Wavelength\s*=\s*([\d.]+)", "\n".join(lines[:15])).group(1))
        step = float(re.search(r"spacing\s*=\s*([\d.]+)", "\n".join(lines[:15])).group(1))
        y = np.array([float(l) for l in lines if re.match(r"^\s*-?[\d.]+(e[-+]?\d+)?\s*$", l, re.I)])
        return [(spec.get("T"), first + step * np.arange(len(y)), y, hdr)]
    if reader == "erle":
        hdr = [l for l in lines[:8] if l and not re.match(r"^\s*[-\d.]", l)]
        y = np.array([float(l) for l in lines if re.match(r"^\s*-?[\d.]+(e[-+]?\d+)?\s*$", l, re.I)])
        return [(spec.get("T"), 440.0 + 0.001 * np.arange(len(y)), y, hdr)]
    rows, header = rows_varcols(lines)
    if not rows:
        raise ValueError("no numeric rows")
    if "blocks" in spec:
        segs = split_blocks(np.array([r[0] for r in rows]))
        out = []
        for b, T, what in spec["blocks"]:
            i0, i1 = segs[b]
            a = np.array([r[:2] for r in rows[i0:i1]])
            out.append((T, a[:, 0], a[:, 1], header + [what]))
        return out
    if "temps" in spec:
        out = []
        for col, T in spec["temps"]:
            pts = [(r[0], r[col]) for r in rows if len(r) > col]
            if not pts:
                continue
            a = np.array(pts)
            out.append((T, a[:, 0], a[:, 1], header))
        return out
    k = min(len(r) for r in rows)
    data = np.array([r[:k] for r in rows])
    if data.shape[1] < 2:
        raise ValueError("one column only")
    col = spec.get("col")
    if col is None:
        col = xsm._sigma_column(data, 0) if data.shape[1] > 2 else 1
    return [(spec.get("T"), data[:, 0], data[:, col], header)]


def split_blocks(x, jump=5.0):
    """[(start, stop)] of the runs of x between jumps larger than `jump` (nm or cm-1)
    against the main direction - separate data sets written one after the other."""
    x = np.asarray(x, float)
    d = np.diff(x)
    main = np.sign(np.sum(np.sign(d)))
    cut = np.where(d * main < -jump)[0] + 1
    edges = [0] + list(cut) + [len(x)]
    return list(zip(edges[:-1], edges[1:]))


def convert(x, y, spec):
    """Apply scale, wavenumber -> wavelength, vacuum -> air; sort, drop no-data, average repeats."""
    steps = []
    x = np.asarray(x, float); y = np.asarray(y, float) * spec.get("scale", 1.0)
    if spec.get("scale", 1.0) != 1.0:
        steps.append("sigma x %g" % spec["scale"])
    ok = np.isfinite(x) & np.isfinite(y)
    raw = y / spec.get("scale", 1.0)
    for m in spec.get("missing", ()):
        ok &= raw != m                                  # the marker itself, exactly (0 must not catch 1e-19)
    x, y = x[ok], y[ok]
    big = 1e-40 if spec.get("unit") == "cm5/molecule2" else 1e-14     # no UV-vis cross section is this large
    bad = np.abs(y) > big
    if bad.any():
        x, y = x[~bad], y[~bad]; steps.append("%d non-physical value(s) > %g removed (header text read as data)"
                                              % (bad.sum(), big))
    if spec.get("x", "nm" if np.median(x) < 1000 else "cm-1") == "cm-1":
        x = 1e7 / x; steps.append("wavenumber (cm-1) -> wavelength (nm)")
    if spec.get("vac"):
        x = xsm.vac_to_air(x); steps.append("vacuum -> air wavelength (Edlen)")
    o = np.argsort(x, kind="stable"); x, y = x[o], y[o]
    if np.any(np.diff(x) <= 0):
        ux, inv = np.unique(np.round(x, 6), return_inverse=True)
        y = np.bincount(inv, y) / np.bincount(inv); x = ux
        steps.append("repeated wavelengths averaged")
    return x, y, steps


def validate(x, y, spec):
    if len(x) < 10:
        return "only %d points" % len(x)
    if x[0] < 100 or x[-1] > 3000:
        return "x axis %.4g-%.4g is not a wavelength in nm" % (x[0], x[-1])
    a = np.abs(y[y != 0])
    if not len(a):
        return "all values zero"
    med = float(np.median(a))
    lo, hi = (1e-50, 1e-42) if spec.get("unit") == "cm5/molecule2" else (1e-28, 1e-15)
    if not lo < med < hi:
        return "values (median %.3g) are not a cross section in %s" % (med, spec.get("unit", "cm2/molecule"))
    return None


def same_data(a, b):
    """True when every point of a is a point of b (same wavelength to 1e-3 nm and value to 0.5 %)."""
    xa, ya = a; xb, yb = b
    if xa[0] < xb[0] - 1e-3 or xa[-1] > xb[-1] + 1e-3:
        return False
    i = np.clip(np.searchsorted(xb, xa), 1, len(xb) - 1)
    j = np.where(np.abs(xb[i - 1] - xa) < np.abs(xb[i] - xa), i - 1, i)
    dx = np.abs(xb[j] - xa)
    dy = np.abs(yb[j] - ya) <= 5e-3 * np.maximum(np.abs(yb[j]), np.abs(ya)) + 1e-30
    return np.mean((dx < 1e-3) & dy) > 0.98


def _one_nm(x, y):
    g = np.arange(np.ceil(x[0]), np.floor(x[-1]) + 1, 1.0)
    if len(g) < 3:
        return g, np.full(len(g), np.nan)
    idx = np.searchsorted(x, np.concatenate([g - 0.5, g[-1:] + 0.5]))
    cs = np.concatenate([[0], np.cumsum(y)])
    n = np.diff(idx)
    m = np.full(len(g), np.nan)
    m[n > 0] = (cs[idx[1:]] - cs[idx[:-1]])[n > 0] / n[n > 0]
    return g, m


@np.errstate(invalid="ignore")
def quality_flags(made, kept):
    """Things worth a look, per data set: negative values, few points, and an
    absolute level far from the other data sets of the same gas (median ratio
    of 1 nm means over the common range, against each other set)."""
    data = {}
    for gas, lst in kept.items():
        for label, (x, y) in lst:
            data[label] = (gas, _one_nm(x, y), y)
    for r in made:
        gas, (g, m), y = data[r["source"]]
        flags = []
        # noise around zero in the wings is normal; values below -1 % of the peak are not
        neg = float(np.mean(y < -0.01 * np.max(y)))
        if neg > 0.02:
            flags.append("%.0f %% of the values below -1 %% of the peak" % (100 * neg))
        if r["points"] < 50:
            flags.append("only %d points" % r["points"])
        if r["kind"] == "absolute":
            ratios = []
            for q in made:
                if q is r or q["gas"] != gas or q["kind"] != "absolute":
                    continue
                g2, m2 = data[q["source"]][1]
                gx, ia, ib = np.intersect1d(g, g2, return_indices=True)
                if len(gx) < 5:
                    continue
                va, vb = m[ia], m2[ib]
                ok = np.isfinite(va) & np.isfinite(vb) & (vb > 0.05 * np.nanmax(vb))
                if ok.sum() >= 5:
                    ratios.append(np.median(va[ok] / vb[ok]))
            ratios = [v for v in ratios if np.isfinite(v)]
            if ratios:
                med = float(np.median(ratios))
                if not 0.6 < med < 1.6:
                    flags.append("level %.2f x the other %s data sets" % (med, gas))
        r["quality"] = "; ".join(flags)


def safe(s):
    return re.sub(r"[^A-Za-z0-9.,+-]+", "_", s).strip("_")


# =============================================================================
# Build
# =============================================================================
def sources():
    """(relative name, raw bytes, file key) for every file and zip member, loose files first."""
    loose, zipped = [], []
    for root, _, files in os.walk(SRC):
        for f in sorted(files):
            p = os.path.join(root, f)
            rel = os.path.relpath(p, SRC).replace("\\", "/")
            if f.lower().endswith(".zip"):
                with zipfile.ZipFile(p) as z:
                    for n in z.namelist():
                        if n.endswith("/"):
                            continue
                        raw = z.read(n)
                        if n.lower().endswith(".zip"):          # a zip inside the zip
                            with zipfile.ZipFile(io.BytesIO(raw)) as z2:
                                for n2 in z2.namelist():
                                    if not n2.endswith("/"):
                                        zipped.append(("%s::%s::%s" % (rel, n, n2), z2.read(n2), n2))
                            continue
                        zipped.append(("%s::%s" % (rel, n), raw, n))
            else:
                loose.append((rel, open(p, "rb").read(), rel))
    return loose + zipped


def spec_for(rel, key):
    """The FILES entry of a file or zip member (members are looked up by their own name,
    a whole-archive entry like 'stutz_hono.zip' or 'bremen/no2_293h.zip' by the archive)."""
    k = key.lower()
    arch = rel.split("::")[0].lower()
    if "::" in rel:
        if arch in FILES and (FILES[arch].get("member") in (None, os.path.basename(k))):
            if not FILES[arch].get("member") and not re.search(r"\.(dat|txt)$", k):
                return None
            return FILES[arch]
        return FILES.get(os.path.basename(k)) or FILES.get(k) or _by_basename(k)
    return FILES.get(k)


def _is_copy(spec):
    ref = (spec or {}).get("ref", "").lower()
    return any(w in ref for w in ("copy", "extract", "without header", "part only"))


def _by_basename(k):
    """An entry whose path ends in this name (a zip copy of a file in a subfolder)."""
    b = "/" + os.path.basename(k)
    hits = [v for kk, v in FILES.items() if kk.endswith(b)]
    return hits[0] if len(hits) == 1 else None


def build():
    import shutil
    made, excluded, seen_hash, kept = [], [], {}, {}
    parts_raw = {}
    srcs = sources()
    srcs.sort(key=lambda t: _is_copy(spec_for(t[0], t[2])))     # stable: originals first
    for rel, raw, key in srcs:                                  # the raw text of multi-part data sets
        parts_raw.setdefault(os.path.basename(key).lower(), raw)
    for d in os.listdir(OUT) if os.path.isdir(OUT) else []:     # a clean rebuild of the database folders
        p = os.path.join(OUT, d)
        if os.path.isdir(p):
            shutil.rmtree(p)
    for rel, raw, key in srcs:
        name = os.path.basename(key).lower()
        low = rel.lower()
        h = hashlib.md5(raw.replace(b"\r\n", b"\n")).hexdigest()
        if h in seen_hash:
            excluded.append((rel, "identical copy of %s" % seen_hash[h])); continue
        seen_hash[h] = rel
        reason = next((why for pat, why in NOT_XS
                       if re.search(pat, name) or re.search(pat, low.split("::")[-1])), None)
        spec = spec_for(rel, key)
        if spec is None and reason is None:
            reason = "not identified in the index or headers - not used"
        if spec is not None and reason is not None and "exclude" not in spec and "gas" in spec:
            reason = None                                       # an explicit entry wins over the name rule
        if reason:
            excluded.append((rel, reason)); continue
        if "exclude" in spec:
            excluded.append((rel, spec["exclude"])); continue
        if "part_of" in spec:
            excluded.append((rel, "joined into the %s data set" % spec["part_of"])); continue
        try:
            if "parts" in spec:
                sets = []
                for p in spec["parts"]:
                    sets += read_dataset(parts_raw[p], spec, p)
                T, h0 = sets[0][0], sets[0][3]
                sets = [(T, np.concatenate([s[1] for s in sets]), np.concatenate([s[2] for s in sets]), h0)]
            else:
                sets = read_dataset(raw, spec, name)
        except Exception as e:
            excluded.append((rel, "unreadable: %s" % e)); continue
        for k, why in spec.get("block_skip", {}).items():
            excluded.append(("%s  [block %d]" % (rel, k + 1), why))
        for T, x, y, header in sets:
            x, y, steps = convert(x, y, spec)
            bad = validate(x, y, spec)
            label = rel + ("" if T == spec.get("T") or "temps" not in spec else "  [%g K column]" % T)
            if bad:
                excluded.append((label, bad)); continue
            gas = spec["gas"]
            dup = next((k for k, v in kept.get(gas, []) if same_data((x, y), v)), None)
            if dup:
                excluded.append((label, "same data as %s" % dup)); continue
            kept.setdefault(gas, []).append((label, (x, y)))
            tlab = "%gK" % T if T is not None else "T-unknown"
            stem = safe(rel.split("::")[-1].split("/")[-1])
            fname = "%s_%s_%s.xs" % (safe(gas), tlab, stem)
            folder = os.path.join(OUT, safe(gas))
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, fname)
            n = 2
            while os.path.exists(path):
                path = os.path.join(folder, fname.replace(".xs", "_%d.xs" % n)); n += 1
            kind = spec.get("kind", "absolute")
            unit = spec.get("unit", "cm2/molecule")
            scale_txt = ("air (converted from vacuum)" if spec.get("vac") else
                         "as given in the source (assumed air)")
            notes = ["T %s" % ("%g K" % T if T is not None else "unknown"),
                     "reference: %s" % spec["ref"],
                     "source: Input/modataXs/%s" % label,
                     "kind: %s%s" % (kind, " - broadband removed or relative scale: use as a DOAS reference only"
                                     if kind == "differential" else ""),
                     "unit: %s" % unit,
                     "wavelength scale: %s" % scale_txt]
            if steps:
                notes.append("conversions: %s" % "; ".join(steps))
            if spec.get("note"):
                notes.append("note: %s" % spec["note"])
            if header:
                notes.append("original header: %s" % " / ".join(h for h in header[:3] if h)[:200])
            # the wavelength scale is recorded explicitly so load_xs never converts again
            xsm.save_xs(path, {"gas": gas, "wl": x, "sigma": y},
                        notes=[n.replace("vacuum", "vac.") if n.startswith("original header") else n for n in notes])
            made.append({"gas": gas, "T_K": T, "file": os.path.relpath(path, OUT).replace("\\", "/"),
                         "lo_nm": round(float(x[0]), 3), "hi_nm": round(float(x[-1]), 3), "points": int(len(x)),
                         "step_nm": round(float(np.median(np.diff(x))), 5), "sigma_max": float(np.max(y)),
                         "kind": kind, "unit": unit, "wavelength_scale": "air" if spec.get("vac") else "as given",
                         "reference": spec["ref"], "source": label, "note": spec.get("note", ""),
                         "quality": ""})
    quality_flags(made, kept)
    made.sort(key=lambda r: (r["gas"].lower(), r["T_K"] if r["T_K"] is not None else 1e9, r["file"]))
    cols = list(made[0].keys())
    with open(os.path.join(OUT, "catalog.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(made)
    json.dump(made, open(os.path.join(OUT, "catalog.json"), "w", encoding="utf-8"), indent=1)
    with open(os.path.join(OUT, "excluded.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["source", "reason"]); w.writerows(sorted(excluded))
    return made, excluded


if __name__ == "__main__":
    made, excluded = build()
    gases = sorted({m["gas"] for m in made}, key=str.lower)
    print("%d data sets of %d species written to %s" % (len(made), len(gases), OUT))
    for g in gases:
        ms = [m for m in made if m["gas"] == g]
        print("  %-24s %3d  %s" % (g, len(ms), ", ".join(sorted({"%g K" % m["T_K"] if m["T_K"] else "?" for m in ms}))))
    print("%d files / columns not used (excluded.csv)" % len(excluded))
