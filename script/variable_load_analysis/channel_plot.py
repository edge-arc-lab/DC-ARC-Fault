import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

QUI = Path(__file__).resolve().parent
if str(QUI) not in sys.path:
    sys.path.insert(0, str(QUI))
import multichannel_analysis as tm

nc = tm.nc


def tensione_ricetta(nome):
    m = re.search(r"_(\d{3,4})V_", nome)
    return float(m.group(1)) if m else float("nan")


def statistiche(nome, rec, p):
    s = rec["sig"]
    t1 = rec["t1"]
    pre = s["V"][:p.NOMINAL_N]
    post = s["V"][t1:]
    prima = s["V"][:t1]
    vmax = float(np.max(s["V"]))
    return dict(
        file=nome,
        corrente_nominale_A=rec["nom"]["I"],
        tensione_ricetta_V=tensione_ricetta(nome),
        V_primi_0p2s_mediana=float(np.median(pre)),
        V_prima_t1_media=float(np.mean(prima)),
        V_dopo_t1_media=float(np.mean(post)),
        V_max=vmax,
        quota_dopo_t1_vicina_al_max=float(np.mean(np.abs(post - vmax) < 0.01 * max(abs(vmax), 1e-9))),
    )


def scegli(elenco, n):
    elenco = sorted(elenco, key=lambda x: x[1]["nom"]["I"])
    if len(elenco) <= n:
        return elenco
    idx = np.unique(np.round(np.linspace(0, len(elenco) - 1, n)).astype(int))
    return [elenco[i] for i in idx]


def disegna(scelti, p, out, passo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(scelti)
    fig, ax = plt.subplots(3, n, figsize=(3.2 * n, 7), squeeze=False, sharex="col")
    colori = {"I": "#1f77b4", "V": "#d98e04", "P": "#2e7d32"}
    unita = {"I": "I [A]", "V": "V [V]", "P": "P [W]"}
    for j, (nome, rec) in enumerate(scelti):
        t = np.arange(len(rec["sig"]["I"])) / p.FS_HZ
        for i, c in enumerate(("I", "V", "P")):
            a = ax[i][j]
            a.plot(t[::passo], rec["sig"][c][::passo], color=colori[c], lw=0.7)
            a.axvline(rec["t1"] / p.FS_HZ, color="#c0392b", ls="--", lw=0.9)
            a.axvspan(0, p.NOMINAL_N / p.FS_HZ, color="#999999", alpha=0.15, lw=0)
            if j == 0:
                a.set_ylabel(unita[c])
            if i == 0:
                a.set_title(f"{tensione_ricetta(nome):.0f} V, {rec['nom']['I']:.2f} A", fontsize=9)
        ax[2][j].set_xlabel("t [s]")
    fig.suptitle("Corrente, tensione e potenza: linea rossa = t1, fascia grigia = primi 0,2 s (nominale)", fontsize=10)
    fig.tight_layout()
    Path(out).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(out) / "canali.png", dpi=130)
    plt.close(fig)


def riassunto(righe):
    v0 = np.array([r["V_primi_0p2s_mediana"] for r in righe])
    v1 = np.array([r["V_dopo_t1_media"] for r in righe])
    vr = np.array([r["tensione_ricetta_V"] for r in righe])
    q = np.array([r["quota_dopo_t1_vicina_al_max"] for r in righe])
    ok = np.isfinite(vr) & (vr > 0)
    print(f"\nregistrazioni analizzate: {len(righe)}")
    print(f"V nei primi 0.2 s (mediana tra file): {np.median(v0):.2f} V;  dopo t1 (media tra file, mediana): {np.median(v1):.2f} V")
    if ok.any():
        print(f"V nei primi 0.2 s / tensione della ricetta: mediana {np.median(v0[ok] / vr[ok]):.3f}  (vicino a 1: tensione lato sorgente; vicino a 0: tensione ai capi del gap)")
        print(f"V dopo t1 / tensione della ricetta: mediana {np.median(v1[ok] / vr[ok]):.3f}")
    print(f"file con |V| nei primi 0.2 s sotto 1 V: {100 * np.mean(np.abs(v0) < 1):.1f}%")
    print(f"quota dei campioni dopo t1 entro l'1% del massimo (saturazione): mediana {np.median(q):.3f}")


def esegui(elenco, p, args, out):
    righe = [statistiche(n, r, p) for n, r in elenco]
    riassunto(righe)
    scelti = scegli(elenco, args.n)
    print("\nregistrazioni disegnate:")
    for nome, rec in scelti:
        print(f"  {nome}  ({tensione_ricetta(nome):.0f} V, {rec['nom']['I']:.2f} A)")
    disegna(scelti, p, out, args.passo)
    Path(out).mkdir(parents=True, exist_ok=True)
    with open(Path(out) / "canali_statistiche.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(righe[0].keys()))
        w.writeheader(); w.writerows(righe)
    print(f"\nSalvati {Path(out) / 'canali.png'} e {Path(out) / 'canali_statistiche.csv'}")


def selftest(p, args):
    rng = np.random.default_rng(0)
    elenco = []
    for k, (V, A) in enumerate([(150, 1.5), (250, 2.5), (350, 3.5), (500, 5.0), (400, 4.0), (200, 2.0)]):
        n = 30000
        i = A * (1 + 0.003 * rng.standard_normal(n)); i[8000:] = A * 0.88 * (1 + 0.01 * rng.standard_normal(n - 8000))
        v = np.full(n, 3.0) + 0.05 * rng.standard_normal(n); v[8000:] = 50.0
        rec = tm.prepara(dict(I=i, V=v, P=v * i), p)
        elenco.append((f"E00{k}_{V:04d}V_{A:05.2f}A_x.mat", rec))
    d_selftest = QUI / "ris_canali_selftest"
    args.n = 4
    esegui(elenco, p, args, d_selftest)
    ok = ((d_selftest / "canali.png").stat().st_size > 5000)
    print("selftest:", "OK" if ok else "FALLITO")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="Grafici e statistiche di corrente, tensione e potenza di alcune registrazioni")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--mat-root", default=None)
    ap.add_argument("--out", default="ris_canali")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--passo", type=int, default=10)
    ap.add_argument("--max-file", type=int, default=0)
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
    with open(repo / "dataset/dataset_new/arc_dataset_meta_test.csv", newline="", encoding="utf-8") as f:
        fns = sorted({m["filename"] for m in csv.DictReader(f)})
    if args.max_file:
        fns = fns[:args.max_file]
    elenco = []
    for fn in fns:
        if fn not in mats:
            continue
        try:
            r = tm.prepara(tm.rec_da_mat(mats[fn], args.downsample_factor), p)
        except KeyError:
            r = None
        if r is not None:
            elenco.append((fn, r))
    if not elenco:
        sys.exit("nessuna registrazione utilizzabile")
    esegui(elenco, p, args, args.out)


if __name__ == "__main__":
    main()
