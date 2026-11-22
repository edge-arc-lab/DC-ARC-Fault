#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calcolo dell'energia dissipata dall'arco elettrico (in Joule):
    E = integrale di v_arco(t) * i(t) dt, da t1 a t1 + T_int
    Discretizzato: E ≈ sum( v[k] * i[k] ) * dt

Caratteristiche:
- Valutazione dell'energia dissipata per differenti intervalli di interruzione sintetici (T_int).
- Compensazione della resistenza parassita/di limitazione in serie: v_arco = max(0, v - R * i).
- Integrazione con i tempi di rilevazione effettivi dei modelli (FP32 e INT8) e tempi di attuazione relè.
- Supporto opzionale per innesco ricavato dalla caduta di tensione (t1_V).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

QUI = Path(__file__).resolve().parent
REPO = QUI.parents[1] if (QUI.parents[1] / "dataset").is_dir() else QUI.parent
for p in [str(QUI), str(REPO / "script"), str(REPO / "script" / "causal_current_norm"), str(REPO)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import causal_norm as nc


def energia(v, i, t1_idx, n_campioni, fs, r_serie=0.0):
    """Energia [J] tra t1_idx e t1_idx + n_campioni (troncata alla fine della traccia)."""
    fine = min(len(i), t1_idx + n_campioni)
    ii = i[t1_idx:fine].astype(np.float64)
    vv = v[t1_idx:fine].astype(np.float64)
    if r_serie:
        vv = np.clip(vv - r_serie * ii, 0.0, None)
    return float(np.sum(vv * ii) / fs)


def selftest(p):
    fs = p.FS_HZ
    t = np.arange(int(2 * fs)) / fs
    i = np.where(t < 0.5, 2.0, 1.0)             # arco: la corrente scende a 1 A a 0.5 s
    v = np.full_like(t, 60.0)                    # tensione costante di 60 V
    t1, _ = p.find_t1(i)
    ok = t1 is not None and abs(t1 / fs - 0.5) < 1e-3
    e100 = energia(v, i, t1, int(0.1 * fs), fs)  # 60 V * 1 A * 0.1 s = 6 J
    e_r = energia(v, i, t1, int(0.1 * fs), fs, r_serie=10.0)   # (60 - 10*1) * 1 * 0.1 = 5 J
    print(f"t1 = {t1 / fs:.3f} s   E(100 ms) = {e100:.3f} J (atteso 6)   con R = 10 ohm {e_r:.3f} J (atteso 5)")
    ok &= abs(e100 - 6.0) < 1e-6 and abs(e_r - 5.0) < 1e-6
    print("SELFTEST", "OK" if ok else "FALLITO")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--mat-root", default=None)
    ap.add_argument("--out", default="ris_energia")
    ap.add_argument("--ritardi", type=float, nargs="+", default=[20, 50, 100, 150, 200, 300],
                    help="tempo totale innesco-interruzione in ms")
    ap.add_argument("--r-serie", type=float, default=0.0, help="resistenza in serie [ohm] da sottrarre (default 0)")
    ap.add_argument("--rilevazione-csv", default=None, help="ritardo_per_file.csv generato da detection_delay.py")
    ap.add_argument("--attuazione", type=float, nargs="+", default=[0, 10, 50], help="tempo di attuazione [ms]")
    ap.add_argument("--downsample-factor", type=int, default=1)
    ap.add_argument("--t1-csv", default=None, help="CSV con etichette da tensione: integra da t1_V invece che da t1")
    ap.add_argument("--tutti", action="store_true", help="tutti i .mat invece dei soli file del test set")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo) if args.repo else REPO
    p = nc.carica_parametri(repo)
    if args.selftest:
        sys.exit(selftest(p))
    if not args.mat_root:
        sys.exit("serve --mat-root (cartella dei .mat a 10 kHz) oppure --selftest")
    from scipy.io import loadmat

    mats = nc.indice_mat(args.mat_root)
    if args.tutti:
        files = sorted(mats)
    else:
        meta_path = repo / "dataset/dataset_new/arc_dataset_meta_test.csv"
        with open(meta_path, newline="", encoding="utf-8") as f:
            files = sorted({m["filename"] for m in csv.DictReader(f)})
        mancanti = [x for x in files if x not in mats]
        if mancanti:
            sys.exit(f"{len(mancanti)} file .mat non trovati in {args.mat_root}, es. {mancanti[:3]}")
    fs = p.FS_HZ
    tv_map = {}
    if args.t1_csv:
        with open(args.t1_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["t1_V_s"] != "" and "#" not in r["file"]:
                    tv_map[r["file"]] = float(r["t1_V_s"])
    saltati = {}
    righe = []
    dati = {}
    for fn in files:
        mat = loadmat(mats[fn], squeeze_me=False)
        if "CurrentData" not in mat or "VoltageData" not in mat:
            saltati[fn] = "CurrentData o VoltageData assente"
            continue
        i = mat["CurrentData"].flatten()[::args.downsample_factor].astype(np.float64)
        v = mat["VoltageData"].flatten()[::args.downsample_factor].astype(np.float64)
        n = min(len(i), len(v))
        i, v = i[:n], v[:n]
        t1, nom = p.find_t1(i)
        if t1 is None:
            saltati[fn] = "nessun arco rilevato (t1 assente)"
            continue
        off = 0
        if args.t1_csv:
            if fn not in tv_map:
                saltati[fn] = "t1_V assente nel csv"
                continue
            tvi = min(int(round(tv_map[fn] * fs / args.downsample_factor)), t1)
            off, t1 = t1 - tvi, tvi            # off = campioni tra t1_V e t1 (corrente)
        riga = dict(file=fn, t1_s=t1 / fs, nominale_A=nom, v_media_arco_V=float(np.mean(v[t1:])),
                    i_media_arco_A=float(np.mean(i[t1:])))
        for d in args.ritardi:
            riga[f"E_{d:g}ms_J"] = energia(v, i, t1, int(round(d / 1000 * fs)), fs, args.r_serie)
        righe.append(riga)
        dati[fn] = (v, i, t1, off)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not righe:
        sys.exit("nessun file con arco: nulla da calcolare")
    with open(out / "energia_per_file.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(righe[0].keys()))
        w.writeheader()
        w.writerows(righe)
    riepilogo = []
    print(f"\nEsperimenti con arco: {len(righe)} (saltati: {len(saltati)})   R serie = {args.r_serie:g} ohm")
    print(f"{'T_int [ms]':>11}{'mediana [J]':>13}{'95° perc. [J]':>15}{'massimo [J]':>13}")
    for d in args.ritardi:
        e = np.array([r[f"E_{d:g}ms_J"] for r in righe])
        riepilogo.append(dict(T_int_ms=d, mediana_J=float(np.median(e)), p95_J=float(np.percentile(e, 95)),
                              massimo_J=float(e.max()), file_massimo=righe[int(e.argmax())]["file"]))
        print(f"{d:>11g}{np.median(e):>13.3f}{np.percentile(e, 95):>15.3f}{e.max():>13.3f}")
    per_modello = []
    dettaglio = []
    if args.rilevazione_csv:
        rit = {}
        with open(args.rilevazione_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rit[(r["file"], r["modello"], int(r["K"]))] = float(r["ritardo_ms"]) if r["ritardo_ms"] not in ("", "None") else None
        combos = sorted({(m, K) for (_, m, K) in rit})
        print(f"\nEnergia con il ritardo di rilevazione reale + attuazione (J); mancati = esperimenti non rilevati (integrati fino a fine traccia)")
        print(f"{'Modello':<10}{'K':>3}{'att. [ms]':>10}{'mancati':>9}{'mediana':>10}{'95° perc.':>11}{'massimo':>10}")
        for m, K in combos:
            for att in args.attuazione:
                e, mancati = [], 0
                for fn, (v, i, t1, off) in dati.items():
                    d = rit.get((fn, m, K), "assente")
                    if d == "assente":
                        continue
                    if d is None:
                        mancati += 1
                        n_c = len(i) - t1
                    else:
                        n_c = off + int(round((d + att) / 1000 * fs))
                    e.append(energia(v, i, t1, n_c, fs, args.r_serie))
                    dettaglio.append(dict(file=fn, modello=m, K=K, attuazione_ms=att,
                                          ritardo_ms=("" if d is None else d), rilevato=int(d is not None),
                                          energia_J=e[-1]))
                e = np.array(e)
                per_modello.append(dict(modello=m, K=K, attuazione_ms=att, mancati=mancati, n=len(e),
                                        mediana_J=float(np.median(e)), p95_J=float(np.percentile(e, 95)),
                                        massimo_J=float(e.max())))
                print(f"{m:<10}{K:>3}{att:>10g}{mancati:>9}{np.median(e):>10.3f}{np.percentile(e, 95):>11.3f}{e.max():>10.3f}")
    if dettaglio:
        with open(out / "energia_rilevazione_per_file.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(dettaglio[0].keys()))
            w.writeheader()
            w.writerows(dettaglio)
    with open(out / "riepilogo_energia.json", "w", encoding="utf-8") as f:
        json.dump(dict(start=("tensione" if args.t1_csv else "corrente"), args={k: v for k, v in vars(args).items() if k != "mat_root"},
                       n_esperimenti=len(righe), saltati=saltati, riepilogo=riepilogo, con_rilevazione=per_modello), f, indent=1)
    print(f"\nSalvato in {out}/ (energia_per_file.csv, riepilogo_energia.json, energia_rilevazione_per_file.csv)")


if __name__ == "__main__":
    main()
