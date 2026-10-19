#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
etichette_tensione_v1.py - istante di innesco dall'andamento della TENSIONE, confrontato con t1 dalla corrente
=============================================================================================================

Obiettivo: verificare se l'istante di innesco dell'arco ricavato dalla tensione ai capi degli elettrodi
(VoltageData) coincide con t1 definito sulla corrente (corrente < 95% del nominale per 300 ms, pipeline).
La tensione NON e' un ingresso dei modelli: serve solo a costruire un'etichetta indipendente dal livello
di corrente.

Definizione di t1_V (parametri modificabili):
  - riferimento v_base = mediana di VoltageData nei primi --base-s secondi (default 0.2 s, come I_nom);
  - rumore = 1.4826 * MAD di VoltageData nello stesso tratto;
  - soglia = max(--dv, --k-mad * rumore) [V];
  - VoltageData e' filtrata con una mediana mobile di --smooth-ms (default 5 ms);
  - t1_V = primo istante in cui |v_filtrata - v_base| >= soglia per almeno --confirm-ms (default 20 ms).

Per ogni file (test set; --tutti per tutti i .mat) scrive etichette_tensione.csv con
    t1_I_s, t1_V_s, differenza (t1_I - t1_V, in ms), v_base, soglia, livello medio della corrente nei 100 ms
    dopo t1_V (media / nominale) e il minimo nello stesso tratto.
Stampa un riepilogo: file senza t1_V, distribuzione della differenza, e quanti archi hanno livello
medio >= 0.95 subito dopo t1_V (cioe' archi che l'etichetta sul livello non vedrebbe), piu' grafici
(--n-grafici) dei file con la differenza maggiore.

Uso (dalla cartella della repo)
    python -B script/etichette_tensione_v1.py --repo . --mat-root "<cartella dei .mat>" --out ..\\ris_etichette_V
    python -B script/etichette_tensione_v1.py --repo . --mat-root "<...>" --out ..\\ris_etichette_V --dv 10 --confirm-ms 50
    python -B script/etichette_tensione_v1.py --repo . --selftest
Non modifica nulla e non usa i modelli. Cerca i moduli in script/ e nelle sue sottocartelle.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

VERSIONE = "v1"
QUI = Path(__file__).resolve().parent
_radici = [QUI] + ([QUI.parent] if QUI.parent.name == "script" else [])
for _r in list(_radici):
    _radici += [x for x in _r.iterdir() if x.is_dir()]
for _d in _radici:
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))
import norm_causale_v2 as nc  # noqa: E402


def mediana_mobile(x, n):
    """Mediana mobile centrata su n campioni (n dispari), bordi replicati."""
    n = max(1, int(n) | 1)
    if n == 1:
        return x.copy()
    pad = n // 2
    xp = np.pad(x, pad, mode="edge")
    from numpy.lib.stride_tricks import sliding_window_view
    return np.median(sliding_window_view(xp, n), axis=1)


def t1_da_tensione(v, fs, base_s=0.2, dv=5.0, k_mad=10.0, smooth_ms=5.0, confirm_ms=20.0):
    """Restituisce (indice di t1_V o None, v_base, soglia)."""
    nb = max(10, int(round(base_s * fs)))
    base = v[:nb]
    v_base = float(np.median(base))
    rumore = 1.4826 * float(np.median(np.abs(base - v_base)))
    soglia = max(dv, k_mad * rumore)
    vs = mediana_mobile(v, int(round(smooth_ms / 1000 * fs)))
    sopra = np.abs(vs - v_base) >= soglia
    nc_ = max(1, int(round(confirm_ms / 1000 * fs)))
    if len(sopra) < nc_:
        return None, v_base, soglia
    # prima posizione con nc_ campioni consecutivi sopra soglia
    c = np.convolve(sopra.astype(np.int32), np.ones(nc_, dtype=np.int32), mode="valid")
    idx = np.where(c >= nc_)[0]
    return (int(idx[0]) if len(idx) else None), v_base, soglia


def selftest():
    fs = 10000
    rng = np.random.default_rng(0)
    t = np.arange(int(3 * fs)) / fs
    v = np.where(t < 0.5, 0.2, 50.0) + rng.normal(0, 0.3, len(t))
    i = np.where(t < 0.7, 4.0, 2.0) + rng.normal(0, 0.02, len(t))
    idx, vb, soglia = t1_da_tensione(v, fs)
    ok = idx is not None and abs(idx / fs - 0.5) < 0.01 and abs(vb - 0.2) < 0.2
    # nessun salto: nessun t1_V
    v2 = np.full(len(t), 10.0) + rng.normal(0, 0.3, len(t))
    ok &= t1_da_tensione(v2, fs)[0] is None
    # picco isolato breve (< conferma): non conta
    v3 = v2.copy(); v3[5000:5050] += 40
    ok &= t1_da_tensione(v3, fs)[0] is None
    print(f"t1_V = {idx / fs:.4f} s (atteso 0.5)  v_base = {vb:.2f} V  soglia = {soglia:.2f} V")
    print("SELFTEST", "OK" if ok else "FALLITO")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--mat-root", default=None)
    ap.add_argument("--out", default="ris_etichette_V")
    ap.add_argument("--base-s", type=float, default=0.2)
    ap.add_argument("--dv", type=float, default=5.0, help="soglia minima di variazione della tensione [V]")
    ap.add_argument("--k-mad", type=float, default=10.0)
    ap.add_argument("--smooth-ms", type=float, default=5.0)
    ap.add_argument("--confirm-ms", type=float, default=20.0)
    ap.add_argument("--n-grafici", type=int, default=6)
    ap.add_argument("--tutti", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if not args.mat_root:
        sys.exit("serve --mat-root oppure --selftest")

    repo = Path(args.repo) if args.repo else (QUI.parent if (QUI.parent / "dataset").is_dir() else Path.cwd())
    p = nc.carica_parametri(repo)
    fs = p.FS_HZ
    from scipy.io import loadmat
    mats = nc.indice_mat(args.mat_root)
    if args.tutti:
        files = sorted(mats)
    else:
        sys.path.insert(0, str(repo / "script"))
        try:
            import analisi_test_set_v1 as a
        except ImportError:
            import test_set_analysis as a
        with open(repo / a.TEST_META, newline="", encoding="utf-8") as f:
            files = sorted({m["filename"] for m in csv.DictReader(f)})

    righe, senza_tI, senza_tV, dati = [], [], [], {}
    win = int(round(0.1 * fs))
    for fn in files:
        if fn not in mats:
            continue
        m = loadmat(mats[fn], squeeze_me=False)
        chiavi = {k.lower(): k for k in m if not k.startswith("__")}
        i = m[chiavi["currentdata"]].flatten().astype(np.float64)
        v = m[chiavi["voltagedata"]].flatten().astype(np.float64)
        n = min(len(i), len(v))
        i, v = i[:n], v[:n]
        tI, nom = p.find_t1(i)
        tV, vb, soglia = t1_da_tensione(v, fs, args.base_s, args.dv, args.k_mad, args.smooth_ms, args.confirm_ms)
        r = dict(file=fn, nominale_A=nom, t1_I_s=(tI / fs if tI is not None else ""),
                 t1_V_s=(tV / fs if tV is not None else ""), v_base=vb, soglia_V=soglia,
                 differenza_ms="", livello_medio_100ms_dopo_tV="", livello_min_100ms_dopo_tV="")
        if tI is None:
            senza_tI.append(fn)
        if tV is None:
            senza_tV.append(fn)
        if tV is not None and nom:
            seg = i[tV:tV + win] / nom
            if len(seg):
                r["livello_medio_100ms_dopo_tV"] = float(seg.mean())
                r["livello_min_100ms_dopo_tV"] = float(seg.min())
        if tI is not None and tV is not None:
            r["differenza_ms"] = (tI - tV) / fs * 1000
        righe.append(r)
        dati[fn] = (i, v, nom, tI, tV)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "etichette_tensione.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(righe[0].keys()))
        w.writeheader()
        w.writerows(righe)

    n = len(righe)
    d = np.array([r["differenza_ms"] for r in righe if r["differenza_ms"] != ""], dtype=float)
    liv = np.array([r["livello_medio_100ms_dopo_tV"] for r in righe if r["livello_medio_100ms_dopo_tV"] != ""], dtype=float)
    print(f"\nFile esaminati: {n}; senza t1 dalla corrente: {len(senza_tI)}; senza t1 dalla tensione: {len(senza_tV)}")
    riep = dict(versione=VERSIONE, args={k: v for k, v in vars(args).items() if k != "mat_root"},
                n_file=n, senza_t1_corrente=len(senza_tI), senza_t1_tensione=len(senza_tV))
    if len(d):
        print("Differenza t1_I - t1_V [ms] (positiva = la corrente segnala l'arco DOPO la tensione):")
        q = np.percentile(d, [5, 25, 50, 75, 95])
        print(f"  5° {q[0]:.0f}   25° {q[1]:.0f}   mediana {q[2]:.0f}   75° {q[3]:.0f}   95° {q[4]:.0f}   min {d.min():.0f}   max {d.max():.0f}")
        for soglia_ms in (50, 200, 1000):
            print(f"  |differenza| > {soglia_ms} ms: {int((np.abs(d) > soglia_ms).sum())} file su {len(d)}")
        riep["differenza_ms"] = dict(p5=q[0], p25=q[1], mediana=q[2], p75=q[3], p95=q[4], min=float(d.min()), max=float(d.max()),
                                     oltre_50ms=int((np.abs(d) > 50).sum()), oltre_200ms=int((np.abs(d) > 200).sum()),
                                     oltre_1000ms=int((np.abs(d) > 1000).sum()))
    if len(liv):
        print("Livello medio della corrente (media / nominale) nei 100 ms dopo t1_V:")
        q = np.percentile(liv, [5, 25, 50, 75, 95])
        print(f"  5° {q[0]:.3f}   25° {q[1]:.3f}   mediana {q[2]:.3f}   75° {q[3]:.3f}   95° {q[4]:.3f}")
        print(f"  archi con livello >= 0.95 subito dopo t1_V: {int((liv >= 0.95).sum())} su {len(liv)} "
              f"(l'etichetta sul livello non li vedrebbe)")
        riep["livello_dopo_tV"] = dict(p5=q[0], p25=q[1], mediana=q[2], p75=q[3], p95=q[4],
                                       con_livello_almeno_095=int((liv >= 0.95).sum()), n=len(liv))
    with open(out / "riepilogo_etichette_tensione.json", "w", encoding="utf-8") as f:
        json.dump(riep, f, indent=1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cand = [r for r in righe if r["differenza_ms"] != ""]
        cand.sort(key=lambda r: -abs(r["differenza_ms"]))
        for r in cand[:args.n_grafici]:
            i, v, nom, tI, tV = dati[r["file"]]
            t = np.arange(len(i)) / fs
            fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
            ax[0].plot(t, i / nom, "k", lw=0.6)
            ax[0].axhline(0.95, color="r", ls="--", lw=0.8)
            ax[0].set_ylabel("corrente / nominale")
            ax[1].plot(t, v, "b", lw=0.6)
            ax[1].set_ylabel("tensione [V]")
            ax[1].set_xlabel("tempo [s]")
            for a_ in ax:
                if tI is not None:
                    a_.axvline(tI / fs, color="r", ls=":", lw=1, label="t1 corrente")
                if tV is not None:
                    a_.axvline(tV / fs, color="g", ls=":", lw=1, label="t1 tensione")
            ax[0].legend(loc="best", fontsize=8)
            fig.suptitle(r["file"], fontsize=8)
            fig.tight_layout()
            fig.savefig(out / ("grafico_" + r["file"].replace(" ", "_").replace(".mat", "") + ".png"), dpi=120)
            plt.close(fig)
    except Exception as e:
        print(f"(grafici non creati: {e})")
    print(f"\nSalvato in {out}/ (etichette_tensione.csv, riepilogo_etichette_tensione.json, grafici)")


if __name__ == "__main__":
    main()
