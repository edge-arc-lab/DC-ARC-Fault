#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analisi dello spettro di frequenza medio del segnale di corrente: pre-arco vs arco.

Confronta la densita' spettrale di potenza mediata nel tempo suddivisa per
finestre temporali progressive rispetto all'innesco (0-0.5 s, 0.5-1 s, 1-2 s, >2 s)
e per bande di frequenza UL 1699B.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

QUI = Path(__file__).resolve().parent
REPO = QUI.parents[1] if (QUI.parents[1] / "dataset").is_dir() else QUI.parent
for p in [str(QUI), str(REPO / "script"), str(REPO / "script" / "causal_current_norm"), str(REPO)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import multichannel_analysis as tm

nc = tm.nc
FASCE = [(10, 100), (100, 500), (500, 1000), (1000, 2000), (2000, 5000)]
CHIAVI = ["no", "a0", "a1", "a2", "a3", "arco"]
ETICH = {"no": "senza arco", "a0": "arco 0-0,5 s dall'innesco", "a1": "arco 0,5-1 s", "a2": "arco 1-2 s", "a3": "arco oltre 2 s", "arco": "arco, tutte le finestre"}
COL = {"no": "tab:blue", "a0": "#fcae91", "a1": "#fb6a4a", "a2": "#de2d26", "a3": "#a50f15", "arco": "black"}
LIMITI_T = [0.0, 0.5, 1.0, 2.0, 1e9]


def finestre_registrazione(r, p, normalizza):
    x = np.asarray(r["sig"]["I"], dtype=np.float64)
    if normalizza:
        x = x / r["nom"]["I"]
    st = np.arange(0, len(x) - p.WIN_N + 1, p.STEP_N)
    lab = tm.etichette_finestre(st, r["t1"], p)
    idx = np.where(lab >= 0)[0]
    if len(idx) == 0:
        return None
    W = np.stack([x[st[i]:st[i] + p.WIN_N] for i in idx])
    W = W - W.mean(1, keepdims=True)
    w = np.hanning(p.WIN_N)
    S = np.abs(np.fft.rfft(W * w, axis=1)) ** 2 / ((w ** 2).sum() * p.FS_HZ)
    t = (st[idx] - r["t1"]) / p.FS_HZ
    return S, lab[idx], t


def gruppi(S, lab, t):
    out = {"no": S[lab == 0], "arco": S[lab == 1]}
    for j in range(4):
        m = (lab == 1) & (t >= LIMITI_T[j]) & (t < LIMITI_T[j + 1])
        out[f"a{j}"] = S[m]
    return {k: v for k, v in out.items() if len(v) > 0}


def riassunto(S, f, df):
    tot = S[:, 1:].sum(1) + 1e-30
    bv = np.column_stack([S[:, (f >= a) & (f < b)].sum(1) * df for a, b in FASCE])
    br = np.column_stack([np.log(S[:, (f >= a) & (f < b)].sum(1) / tot + 1e-12) for a, b in FASCE])
    return dict(S=np.median(S, 0), rel=np.median(S / tot[:, None], 0), bv_med=np.median(bv, 0), bv_mean=bv.mean(0), br=np.median(br, 0), n=len(S))


def calcola(recs, p, args):
    f = np.fft.rfftfreq(p.WIN_N, 1 / p.FS_HZ)
    df = f[1] - f[0]
    R = {k: dict(k=[], S=[], rel=[], bv_med=[], bv_mean=[], br=[], n=[]) for k in CHIAVI}
    for i, r in enumerate(recs):
        o = finestre_registrazione(r, p, not args.assoluta)
        if o is not None:
            for k, S in gruppi(*o).items():
                s = riassunto(S, f, df)
                R[k]["k"].append(i)
                for c in ("S", "rel", "bv_med", "bv_mean", "br", "n"):
                    R[k][c].append(s[c])
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(recs)} registrazioni")
    for k in R:
        for c in R[k]:
            R[k][c] = np.array(R[k][c])
    return f, R


def appaiate(R, k):
    comuni, i0, i1 = np.intersect1d(R["no"]["k"], R[k]["k"], return_indices=True)
    return i0, i1


def ylim_da(curve, m=3.0):
    return min(np.min(c) for c in curve) / m, max(np.max(c) for c in curve) * m


def sfondo_bande(ax):
    for a, b in FASCE:
        ax.axvline(a, color="gray", ls=":", lw=0.8)
    ax.axvline(FASCE[-1][1], color="gray", ls=":", lw=0.8)


def grafico_iqr(f, R, chiave, ylabel, titolo, nome, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = f >= 10
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    curve = []
    for k in ("no", "arco"):
        col = "tab:blue" if k == "no" else "tab:red"
        q = np.percentile(R[k][chiave], [25, 50, 75], axis=0)
        ax.fill_between(f[m], q[0][m], q[2][m], color=col, alpha=0.2)
        ax.plot(f[m], q[1][m], color=col, lw=1.3, label=f"{ETICH[k]} ({len(R[k]['k'])} registrazioni)")
        curve += [q[0][m], q[2][m]]
    sfondo_bande(ax)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(*ylim_da(curve))
    ax.set_xlabel("Frequenza [Hz]"); ax.set_ylabel(ylabel); ax.set_title(titolo, fontsize=10)
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    fig.savefig(out / nome, dpi=200)
    plt.close(fig)


def grafico_tempo(f, R, chiave, ylabel, titolo, nome, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = f >= 10
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    curve = []
    for k in ("no", "a0", "a1", "a2", "a3"):
        if len(R[k]["k"]) == 0:
            continue
        y = np.median(R[k][chiave], axis=0)
        ax.plot(f[m], y[m], color=COL[k], lw=1.3, label=f"{ETICH[k]} ({len(R[k]['k'])} reg.)")
        curve.append(y[m])
    sfondo_bande(ax)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(*ylim_da(curve, 2.0))
    ax.set_xlabel("Frequenza [Hz]"); ax.set_ylabel(ylabel); ax.set_title(titolo, fontsize=10)
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)
    fig.savefig(out / nome, dpi=200)
    plt.close(fig)


def grafico_rapporto(f, R, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    i0, i1 = appaiate(R, "arco")
    m = f >= 10
    r = np.log10(R["arco"]["rel"][i1][:, m] + 1e-30) - np.log10(R["no"]["rel"][i0][:, m] + 1e-30)
    q = np.percentile(r, [25, 50, 75], axis=0)
    frac = (r > 0).mean(0)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 6.4), sharex=True, constrained_layout=True)
    a1.fill_between(f[m], q[0], q[2], color="tab:purple", alpha=0.2)
    a1.plot(f[m], q[1], color="tab:purple", lw=1.3)
    a1.axhline(0, color="k", lw=0.8)
    a1.set_ylabel("log10 del rapporto", fontsize=9)
    a1.set_title(f"Quota di energia, arco / senza arco, stesse registrazioni (n = {len(i0)})", fontsize=10)
    a2.plot(f[m], 100 * frac, color="tab:purple", lw=1.0)
    a2.axhline(50, color="k", lw=0.8)
    a2.set_ylabel("% registrazioni con rapporto > 1", fontsize=9); a2.set_ylim(0, 100)
    a2.set_xlabel("Frequenza [Hz]")
    for a in (a1, a2):
        a.set_xscale("log"); a.grid(True, which="both", alpha=0.3); sfondo_bande(a)
    fig.savefig(out / "spettro_rapporto_appaiato.png", dpi=200)
    plt.close(fig)


def stampa(R, args):
    print("\nregistrazioni per gruppo: " + "  ".join(f"{k}: {len(R[k]['k'])}" for k in CHIAVI))
    print("\n=== Energia assoluta per banda (varianza di i/I_nom nella banda, mediana tra registrazioni delle mediane per finestra)")
    print(f"  {'gruppo':<28}" + "".join(f"{a}-{b}".rjust(14) for a, b in FASCE))
    for k in CHIAVI:
        v = np.median(R[k]["bv_med"], 0)
        print(f"  {ETICH[k]:<28}" + "".join(f"{x:>14.2e}" for x in v))
    print("\n=== Rapporto con arco / senza arco, stesse registrazioni: mediana del rapporto per registrazione (registrazioni con rapporto > 1)")
    print(f"  {'gruppo':<28}" + "".join(f"{a}-{b}".rjust(18) for a, b in FASCE))
    for k in CHIAVI[1:]:
        i0, i1 = appaiate(R, k)
        if len(i0) == 0:
            continue
        cel = []
        for j in range(len(FASCE)):
            rr = R[k]["bv_med"][i1][:, j] / (R["no"]["bv_med"][i0][:, j] + 1e-30)
            cel.append(f"{np.median(rr):>8.2f} ({100 * (rr > 1).mean():>4.0f}%)")
        print(f"  {ETICH[k]:<28}" + "".join(f"{c:>18}" for c in cel) + f"   n={len(i0)}")
    print("\n=== Media contro mediana per finestra (arco, tutte le finestre): rapporto con arco / senza arco della banda 10-100 Hz")
    i0, i1 = appaiate(R, "arco")
    for nome in ("bv_mean", "bv_med"):
        rr = R["arco"][nome][i1][:, 0] / (R["no"][nome][i0][:, 0] + 1e-30)
        print(f"  {'media per registrazione' if nome == 'bv_mean' else 'mediana per registrazione':<28} rapporto mediano {np.median(rr):>9.2f}   registrazioni con rapporto > 1: {100 * (rr > 1).mean():.1f}%")
    print("\n=== Quota relativa (log di energia banda / energia totale): differenza rispetto a senza arco, stesse registrazioni")
    print(f"  {'gruppo':<28}" + "".join(f"{a}-{b}".rjust(14) for a, b in FASCE))
    righe = []
    for k in CHIAVI[1:]:
        i0, i1 = appaiate(R, k)
        if len(i0) == 0:
            continue
        d = R[k]["br"][i1] - R["no"]["br"][i0]
        print(f"  {ETICH[k]:<28}" + "".join(f"{np.median(d[:, j]):>+14.2f}" for j in range(len(FASCE))))
        for j, (a, b) in enumerate(FASCE):
            righe.append(dict(gruppo=k, banda=f"{a}-{b}", n=len(i0), diff_rel_mediana=np.median(d[:, j]), frazione_positiva=(d[:, j] > 0).mean(),
                              rapporto_assoluto_mediano=np.median(R[k]["bv_med"][i1][:, j] / (R["no"]["bv_med"][i0][:, j] + 1e-30))))
    return righe


def salva(f, R, righe, out):
    with open(out / "spettro_medio_v3_bande.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in righe for k in r}))
        w.writeheader(); w.writerows(righe)
    with open(out / "spettro_medio_v3_mediane.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["freq_hz"] + [f"ass_{k}" for k in CHIAVI] + [f"rel_{k}" for k in CHIAVI])
        A = {k: np.median(R[k]["S"], 0) if len(R[k]["k"]) else np.full(len(f), np.nan) for k in CHIAVI}
        Rr = {k: np.median(R[k]["rel"], 0) if len(R[k]["k"]) else np.full(len(f), np.nan) for k in CHIAVI}
        for i in range(len(f)):
            w.writerow([f[i]] + [A[k][i] for k in CHIAVI] + [Rr[k][i] for k in CHIAVI])


def esegui(recs, p, args):
    f, R = calcola(recs, p, args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if len(R["no"]["k"]) < 3 or len(R["arco"]["k"]) < 3:
        sys.exit("registrazioni insufficienti")
    um = "(i/I_nom)^2 / Hz" if not args.assoluta else "A^2 / Hz"
    grafico_iqr(f, R, "S", f"Densità spettrale di potenza [{um}]", "Spettro assoluto: mediana tra registrazioni (mediana per registrazione) e interquartile", "spettro_assoluto_mediana.png", out)
    grafico_iqr(f, R, "rel", "Quota di energia per bin (S / energia totale)", "Spettro normalizzato: mediana tra registrazioni e interquartile", "spettro_normalizzato_mediana.png", out)
    grafico_tempo(f, R, "S", f"Densità spettrale di potenza [{um}]", "Spettro assoluto per tempo dall'innesco", "spettro_assoluto_per_tempo.png", out)
    grafico_tempo(f, R, "rel", "Quota di energia per bin (S / energia totale)", "Spettro normalizzato per tempo dall'innesco", "spettro_normalizzato_per_tempo.png", out)
    grafico_rapporto(f, R, out)
    righe = stampa(R, args)
    salva(f, R, righe, out)
    return f, R


def selftest(p, args):
    rng = np.random.default_rng(1)
    mk = lambda n: [tm.prepara(tm.sint_ft(rng, p, float(rng.choice([1.5, 3.0]))), p) for _ in range(n)]
    recs = tm.con_tv(mk(24), p, args)
    f, R = esegui(recs, p, args)
    out = Path(args.out)
    ok = len(R["no"]["k"]) > 3 and (out / "spettro_normalizzato_per_tempo.png").exists() and np.isfinite(R["arco"]["rel"]).all()
    print("\nselftest:", "OK" if ok else "FALLITO")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="Spettro per registrazione con mediane robuste e arco suddiviso per tempo dall'innesco")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--mat-root", default=None)
    ap.add_argument("--out", default="ris_spettro_medio_v3")
    ap.add_argument("--etichette", choices=["tensione", "corrente"], default="tensione")
    ap.add_argument("--insiemi", choices=["train", "test", "entrambi"], default="entrambi")
    ap.add_argument("--assoluta", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--soglia-v", type=float, default=5.0)
    ap.add_argument("--downsample-factor", type=int, default=1)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.repo:
        repo = Path(args.repo)
    else:
        repo = Path.cwd()
        for cand in [QUI, QUI.parent, QUI.parent.parent, Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]:
            if (cand / "dataset").is_dir():
                repo = cand
                break
    repo = repo.resolve()
    p = nc.carica_parametri(repo)
    if args.selftest:
        sys.exit(selftest(p, args))
    if not args.mat_root:
        sys.exit("serve --mat-root (cartella dei .mat) oppure --selftest")
    mats = nc.indice_mat(args.mat_root)
    recs = []
    for nome, chiave in (("arc_dataset_meta_train.csv", "train"), ("arc_dataset_meta_test.csv", "test")):
        if args.insiemi in (chiave, "entrambi"):
            recs += tm.carica(repo, nome, mats, p, args.downsample_factor)
    if args.etichette == "tensione":
        recs = tm.con_tv(recs, p, args)
    if not recs:
        sys.exit("nessuna registrazione valida")
    print(f"registrazioni: {len(recs)}  etichette: {args.etichette}")
    esegui(recs, p, args)
    print(f"\nSalvato in {args.out}")


if __name__ == "__main__":
    main()
