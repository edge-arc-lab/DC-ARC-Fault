#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
t1_tensione_tutti_v1.py - t1 dalla corrente e t1_V dalla tensione per TUTTI i file .mat (train, val e test)
===========================================================================================================

Serve a rietichettare anche il train set con l'innesco dalla tensione (vedi rietichetta_dataset_V_v1.py).
Per ogni istanza .mat scrive: file (chiave 'nome' o 'nome#2' per gli omonimi), t1_I_s (pipeline: < 0.95 x nominale per 300 ms),
t1_V_s (tensione, come etichette_tensione_v3: soglia 5 V, conferma 100 ms, persistenza 300 ms), nominale_A, rif_instabile
(> 5% dei campioni dei primi 0.2 s oltre la soglia dalla mediana). Stessi parametri della v3/v4 (modificabili da riga di comando).

Uso (dalla cartella della repo)
    python -B script/t1_tensione_tutti_v1.py --repo . --mat-root "<cartella dei .mat>" --out ..\ris_t1_tutti
    python -B script/t1_tensione_tutti_v1.py --selftest
Richiede etichette_tensione_v3.py, etichette_tensione_v4.py e lab_analysis_2.0.py (o sensibilita_etichette_v2.py) nelle cartelle di script/.
Tempo: alcuni minuti (sola lettura dei .mat, nessuna inferenza).
"""
import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

VERSIONE = "v1"
QUI = Path(__file__).resolve().parent


def _carica(nome, repo):
    cand = []
    for base in (QUI, QUI.parent, repo / "script"):
        if base.is_dir():
            cand += list(base.rglob(nome))
    if not cand:
        sys.exit(f"non trovo {nome}")
    sp = importlib.util.spec_from_file_location(nome[:-3], cand[0])
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def calcola(tens, corr, fs, s, v4, nom_n, par):
    nom = float(np.median(corr[:nom_n]))
    if nom < s.BASE["min_nom"] or len(corr) < s.WIN_N + s.STEP_N:
        return None
    tI = s.trova_t1(corr, nom, s.BASE["thresh"], s.BASE["confirm_ms"])
    vs, rif, rum = v4.prepara_v(tens, fs, par["base_s"], par["smooth_ms"])
    tV = v4.t1_da_prep(vs, rif, rum, fs, par["dv"], par["k_mad"], par["confirm_ms"], par["persist_ms"])
    sp = v4.rif_sporco(tens, fs, par["base_s"], par["dv"], par["smooth_ms"], 0.05)
    return nom, tI, tV, sp


def selftest(repo):
    s = _carica("lab_analysis_2.0.py", repo) if list(repo.rglob("lab_analysis_2.0.py")) else _carica("sensibilita_etichette_v2.py", repo)
    v4 = _carica("etichette_tensione_v4.py", repo)
    fs = 10000
    t = np.arange(3 * fs) / fs
    rng = np.random.default_rng(0)
    i = np.where(t < 1.0, 2.0, 1.0) + rng.normal(0, 0.005, len(t))
    v = np.where(t < 0.4, 0.5, 30.0) + rng.normal(0, 0.3, len(t))
    par = dict(base_s=0.2, smooth_ms=5.0, dv=5.0, k_mad=10.0, confirm_ms=100.0, persist_ms=300.0)
    nom, tI, tV, sp = calcola(v, i, fs, s, v4, s.NOMINAL_N, par)
    ok = abs(tI / fs - 1.0) < 0.01 and abs(tV / fs - 0.4) < 0.01 and not sp
    print(f"t1_I = {tI / fs:.3f} s (atteso 1.0), t1_V = {tV / fs:.3f} s (atteso 0.4)")
    print("SELFTEST", "OK" if ok else "FALLITO")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--mat-root", default=None)
    ap.add_argument("--out", default="ris_t1_tutti")
    ap.add_argument("--base-s", type=float, default=0.2)
    ap.add_argument("--dv", type=float, default=5.0)
    ap.add_argument("--k-mad", type=float, default=10.0)
    ap.add_argument("--smooth-ms", type=float, default=5.0)
    ap.add_argument("--confirm-ms", type=float, default=100.0)
    ap.add_argument("--persist-ms", type=float, default=300.0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    repo = Path(a.repo) if a.repo else (QUI.parent if (QUI.parent / "dataset").is_dir() else Path.cwd())
    if a.selftest:
        sys.exit(selftest(repo))
    if not a.mat_root:
        sys.exit("serve --mat-root (o --selftest)")
    try:
        s = _carica("lab_analysis_2.0.py", repo)
    except SystemExit:
        s = _carica("sensibilita_etichette_v2.py", repo)
    v4 = _carica("etichette_tensione_v4.py", repo)
    from scipy.io import loadmat
    mats = s.indice_mat(a.mat_root)
    istanze = s.chiavi_istanze(sorted(mats), mats)
    par = dict(base_s=a.base_s, smooth_ms=a.smooth_ms, dv=a.dv, k_mad=a.k_mad, confirm_ms=a.confirm_ms, persist_ms=a.persist_ms)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    righe, saltati = [], []
    t0 = time.time()
    for n, (k, path) in enumerate(istanze, 1):
        if n % 25 == 0 or n == len(istanze):
            print(f"\r  {n}/{len(istanze)}  ({time.time() - t0:.0f} s)", end="", flush=True)
        m = loadmat(path, squeeze_me=False)
        ch = {x.lower(): x for x in m if not x.startswith("__")}
        if "currentdata" not in ch or "voltagedata" not in ch:
            saltati.append(k)
            continue
        c = m[ch["currentdata"]].flatten().astype(np.float64)
        v = m[ch["voltagedata"]].flatten().astype(np.float64)
        n_ = min(len(c), len(v))
        r = calcola(v[:n_], c[:n_], s.FS, s, v4, s.NOMINAL_N, par)
        if r is None:
            saltati.append(k)
            continue
        nom, tI, tV, sp = r
        righe.append([k, "" if tI is None else tI / s.FS, "" if tV is None else tV / s.FS, nom, int(sp)])
    print()
    with open(out / "t1_per_file.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "t1_I_s", "t1_V_s", "nominale_A", "rif_instabile"])
        w.writerows(righe)
    print(f"Istanze: {len(istanze)}; scritte {len(righe)}; saltate {len(saltati)}; senza t1 corrente: "
          f"{sum(1 for r in righe if r[1] == '')}; senza t1_V: {sum(1 for r in righe if r[2] == '')}; rif. instabile: {sum(r[4] for r in righe)}")
    print(f"Scritto {out.resolve() / 't1_per_file.csv'}")


if __name__ == "__main__":
    main()
