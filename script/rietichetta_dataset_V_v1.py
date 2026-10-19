#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
rietichetta_dataset_V_v1.py - train/test del dataset con etichette dall'INNESCO DALLA TENSIONE (stesse finestre, stesso split)
===========================================================================================================================

Parte dai dataset pubblicati (dataset/dataset_new/arc_dataset_{train,test}.npz + meta) e cambia SOLO l'etichetta di ogni finestra:
    arco (1)     se t_start >= t1_V
    pre-arco (0) se t_end   <= t1_V
    scartata     se la finestra e' a cavallo di t1_V, o se il file non ha t1_V
dove t1_V e' l'innesco dalla tensione (t1_per_file.csv di t1_tensione_tutti_v1.py, oppure etichette_tensione_v3.csv: colonne file, t1_V_s, t1_I_s).
Per gli omonimi si sceglie l'istanza il cui t1_I e' piu' vicino al t1 dei metadati. Le finestre sono quelle pubblicate (10 al massimo
per classe e per file con le etichette dalla corrente): le finestre "pre-arco" che cadono dopo t1_V diventano arco, quelle a cavallo
vengono tolte; il conteggio per classe puo' quindi differire da una ricostruzione da zero. Lo split train/test (per esperimento)
resta quello pubblicato: nessun nuovo leakage.

Scrive in --out-dir: arc_dataset_train_V.npz, arc_dataset_test_V.npz (X, y), arc_dataset_meta_train_V.csv, arc_dataset_meta_test_V.csv, rietichetta_report.txt.
Poi si riaddestra con gli script della repo, per esempio:
    python -B script/training/msrcfe/train_msrcfe.py --train <out>\arc_dataset_train_V.npz --test <out>\arc_dataset_test_V.npz --out ..\ris_msrcfe_V --seed 42
    python -B script/training/inception_time/train_inceptiontime.py --train <out>\arc_dataset_train_V.npz --test <out>\arc_dataset_test_V.npz --out ..\ris_inception_V --seed 42

Uso (dalla cartella della repo)
    python -B script/rietichetta_dataset_V_v1.py --repo . --t1-csv ..\ris_t1_tutti\t1_per_file.csv --out-dir ..\dataset_V
    python -B script/rietichetta_dataset_V_v1.py --selftest
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

VERSIONE = "v1"
QUI = Path(__file__).resolve().parent


def etichette_V(meta, inst):
    lab = np.full(len(meta), -1, dtype=int)
    for i, m in enumerate(meta):
        c = inst.get(m["filename"])
        if not c:
            continue
        t1m = float(m["t1_s"])
        tV = min(c, key=lambda x: abs(x[0] - t1m))[1]
        a, b = float(m["t_start_s"]), float(m["t_end_s"])
        lab[i] = 1 if a >= tV else (0 if b <= tV else -1)
    return lab


def leggi_t1(path):
    inst = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("t1_V_s", "") == "" or r.get("t1_I_s", "") == "":
                continue
            inst.setdefault(r["file"].split("#")[0], []).append((float(r["t1_I_s"]), float(r["t1_V_s"])))
    return inst


def selftest():
    meta = [dict(filename="f", t1_s="1.0", t_start_s=str(s), t_end_s=str(s + 0.1)) for s in (0.0, 0.35, 0.45, 0.9)]
    lab = etichette_V(meta, {"f": [(1.0, 0.4)]})
    ok = list(lab) == [0, -1, 1, 1]
    ok &= list(etichette_V(meta, {})) == [-1] * 4
    print("SELFTEST", "OK" if ok else "FALLITO")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--t1-csv", default=None)
    ap.add_argument("--out-dir", default="dataset_V")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.t1_csv:
        sys.exit("serve --t1-csv (o --selftest)")
    repo = Path(a.repo) if a.repo else (QUI.parent if (QUI.parent / "dataset").is_dir() else Path.cwd())
    inst = leggi_t1(a.t1_csv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    righe_rep = [f"rietichetta_dataset_V {VERSIONE}"]
    for sp in ("train", "test"):
        d = np.load(repo / f"dataset/dataset_new/arc_dataset_{sp}.npz")
        X, y0 = d["X"], d["y"].astype(int)
        with open(repo / f"dataset/dataset_new/arc_dataset_meta_{sp}.csv", newline="", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            campi = rd.fieldnames
            meta = list(rd)
        if len(meta) != len(y0) or not np.array_equal(np.array([int(m["label"]) for m in meta]), y0):
            sys.exit(f"metadati {sp} non allineati")
        lab = etichette_V(meta, inst)
        keep = lab >= 0
        senza_t1 = sum(1 for m in meta if m["filename"] not in inst)
        msg = (f"{sp}: {len(y0)} finestre pubblicate ({int((y0 == 1).sum())} arco / {int((y0 == 0).sum())} pre-arco) -> "
               f"{int(keep.sum())} ({int((lab == 1).sum())} arco / {int((lab == 0).sum())} pre-arco); scartate {int((~keep).sum())} "
               f"(di cui {senza_t1} senza t1_V); cambiate 0->1: {int(((y0 == 0) & (lab == 1)).sum())}; 1->0: {int(((y0 == 1) & (lab == 0)).sum())}")
        print(msg)
        righe_rep.append(msg)
        if keep.sum() == 0:
            print(f"  nessuna finestra {sp} rietichettata: manca t1_V per i suoi file")
            continue
        np.savez(out / f"arc_dataset_{sp}_V.npz", X=X[keep], y=lab[keep].astype(np.int32))
        with open(out / f"arc_dataset_meta_{sp}_V.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=campi)
            w.writeheader()
            for m, l, k in zip(meta, lab, keep):
                if k:
                    m = dict(m)
                    m["label"] = int(l)
                    w.writerow(m)
    (out / "rietichetta_report.txt").write_text("\n".join(righe_rep) + "\n", encoding="utf-8")
    print(f"Scritto in {out.resolve()}")


if __name__ == "__main__":
    main()
