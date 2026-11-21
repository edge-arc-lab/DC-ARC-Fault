#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
make_paper_figures.py - Generazione figure e tabelle LaTeX per il paper
=======================================================================

Legge i file dei risultati generati nella repository (senza ricaricare modelli o file .mat)
e genera:
    <out>/figures/*.pdf + *.png   figure vettoriali (PDF) e raster (PNG)
    <out>/tables/*.tex            tabelle formattate in booktabs per inclusione LaTeX
    <out>/PROVENANCE.md           mappatura input-output con hash SHA-256

Se energia_rilevazione_per_file.csv e' presente sotto results_data/, il pannello (b)
di fig_energy traccia la CDF empirica completa dell'energia dissipata fino alla rilevazione
per modello (K=1, attuazione 0 ms).

Input attesi (opzionali tranne i primi due):
    results_data/ris_ritardo/ris_ritardo/ritardo_per_file.csv       (ritardo rilevamento per file/modello/K)
    results_data/ris_ritardo/ris_ritardo/riepilogo_ritardo.json
    results_data/ris_energia2/riepilogo_energia.json                 (energia vs T_int, energia con rilevazione)
    results_data/ris_sensibilita.zip                                 (classificazione e sensibilita' etichette)
    --etichette-v2 <dir>  cartella di output di etichette_tensione.py (valutazione_etichette.csv)

Output:
    fig_delay_ecdf      CDF empirica del ritardo di rilevazione, K=1 e K=3, 4 modelli, limite UL 1699B (2.5 s)
    fig_energy          energia dissipata: (a) vs T_int, (b) con rilevamento (mediana/p95/max), limite UL 1699B (750 J)
    tab_classification  DR, FPR, FN, FP con CI 95%, etichette base (4 modelli)
    tab_sensitivity     DR/FPR dei 4 modelli per variante di etichettatura
    tab_delay           mediana / p95 / max ritardo per modello e K
    tab_energy          energia con rilevazione (attuazione 0 ms), per modello e K
    tab_level_bands     DR per fascia di livello con etichette da tensione

Uso:
    python -B script/make_paper_figures.py --repo . --out ../paper_assets
    python -B script/make_paper_figures.py --repo . --out ../paper_assets --etichette-v2 ../ris_etichette_V2
    python -B script/make_paper_figures.py --selftest
"""
import argparse
import hashlib
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

LIM_T_MS, LIM_E_J = 2500.0, 750.0
MODELS = ["inc_fp32", "inc_int8", "ms_fp32", "ms_int8"]
NAMES = {"inc_fp32": "InceptionTime FP32", "inc_int8": "InceptionTime INT8",
         "ms_fp32": "MS-RCFE+Ridge FP32", "ms_int8": "MS-RCFE+Ridge INT8"}
LONG = {"InceptionTime FP32": "inc_fp32", "InceptionTime INT8": "inc_int8",
        "MS-RCFE + Ridge FP32": "ms_fp32", "MS-RCFE + Ridge INT8": "ms_int8"}
# colour-blind-safe palette + distinct line styles (never colour alone)
STYLE = {"inc_fp32": ("#0072B2", "-"), "inc_int8": ("#0072B2", "--"),
         "ms_fp32": ("#D55E00", "-"), "ms_int8": ("#D55E00", "--")}
PROV = []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def note(out, src):
    PROV.append((out, src))


def find(base, *cands):
    for c in cands:
        p = Path(base) / c
        if p.exists():
            return p
    return None


def read_zip_or_dir(base, zipname, member):
    """Read <member> from <base>/<zipname> (zip) or from the unzipped folder; returns (bytes, source) or (None, None)."""
    z = Path(base) / zipname
    if z.exists():
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if n.endswith(member):
                    return zf.read(n), f"{zipname}:{n}"
    d = Path(base) / zipname.replace(".zip", "")
    if d.is_dir():
        for p in d.rglob(member):
            return p.read_bytes(), str(p.relative_to(base))
    return None, None


def tex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_tex(path, header, rows, align, caption=None):
    lines = [r"\begin{tabular}{" + align + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def ci(v, lo, hi, d=1):
    return f"{v:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def savefig(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / "figures" / f"{name}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")


# ------------------------------------------------------------------ figures
def fig_delay(res, out, plt):
    p = find(res, "ris_ritardo/ris_ritardo/ritardo_per_file.csv", "ris_ritardo/ritardo_per_file.csv")
    if p is None:
        print("skip fig_delay: ritardo_per_file.csv not found")
        return None
    d = pd.read_csv(p)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for a_, K in zip(ax, (1, 3)):
        for m in MODELS:
            v = np.sort(d[(d.modello == m) & (d.K == K)].ritardo_ms.dropna().to_numpy())
            if len(v) == 0:
                continue
            c, ls = STYLE[m]
            a_.step(v, np.arange(1, len(v) + 1) / len(v) * 100, where="post", color=c, ls=ls, lw=1.4, label=NAMES[m])
        a_.axvline(LIM_T_MS, color="k", ls=":", lw=1)
        a_.set_xscale("log")
        a_.set_xlabel("Detection delay [ms]")
        a_.set_title(f"K = {K} consecutive positive windows", fontsize=9)
        a_.grid(alpha=0.25)
    ax[0].set_ylabel("Experiments detected within delay [%]")
    ax[1].text(LIM_T_MS * 0.93, 8, "2.5 s limit", rotation=90, ha="right", va="bottom", fontsize=8)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    savefig(fig, out, "fig_delay_ecdf")
    plt.close(fig)
    note("figures/fig_delay_ecdf.{pdf,png}", f"{p.relative_to(res)} (sha {sha(p)})")
    return d


def load_energy(res):
    p = find(res, "ris_energia2/riepilogo_energia.json", "riepilogo_energia.json")
    return (json.loads(p.read_text(encoding="utf-8")), p) if p else (None, None)


def fig_energy(res, out, plt):
    j, p = load_energy(res)
    if j is None:
        print("skip fig_energy: riepilogo_energia.json not found")
        return
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    T = [r["T_int_ms"] for r in j["riepilogo"]]
    for key, ls, lab in (("mediana_J", "-", "median"), ("p95_J", "--", "95th percentile"), ("massimo_J", ":", "maximum")):
        ax[0].plot(T, [r[key] for r in j["riepilogo"]], ls=ls, marker="o", ms=3, color="#0072B2", label=lab)
    ax[0].set_xlabel(r"Time from onset to interruption $T_{int}$ [ms]")
    ax[0].set_ylabel("Arc energy [J]")
    ax[0].set_title("(a) fixed interruption time", fontsize=9)
    ax[0].legend(fontsize=7, frameon=False)
    ax[0].grid(alpha=0.25)
    pe = next(iter(sorted(Path(res).rglob("energia_rilevazione_per_file.csv"))), None)
    if pe is not None:
        d = pd.read_csv(pe)
        d = d[(d.K == 1) & (d.attuazione_ms == 0)]
        for m in MODELS:
            v = np.sort(d[d.modello == m].energia_J.to_numpy())
            if len(v) == 0:
                continue
            c, ls = STYLE[m]
            ax[1].step(v, np.arange(1, len(v) + 1) / len(v) * 100, where="post", color=c, ls=ls, lw=1.4, label=NAMES[m])
        ax[1].axvline(LIM_E_J, color="k", ls=":", lw=1)
        ax[1].set_xscale("log")
        ax[1].set_xlabel("Energy until detection [J]")
        ax[1].set_ylabel("Experiments [%]")
        ax[1].set_title("(b) with detection (K = 1, 0 ms actuation)", fontsize=9)
        ax[1].text(LIM_E_J * 0.9, 8, "750 J limit", rotation=90, ha="right", va="bottom", fontsize=8)
        ax[1].legend(fontsize=7, frameon=False, loc="lower right", bbox_to_anchor=(0.9, 0.08))
        ax[1].grid(alpha=0.25)
        note("figures/fig_energy.{pdf,png} panel (b)", f"{pe.relative_to(res)} (sha {sha(pe)})")
    else:
        sel = [r for r in j["con_rilevazione"] if r["attuazione_ms"] == 0.0 and r["K"] == 1]
        x = np.arange(len(sel))
        for key, lab, col, mk in (("mediana_J", "median", "#56B4E9", "o"), ("p95_J", "95th percentile", "#0072B2", "s"),
                                  ("massimo_J", "maximum", "#D55E00", "^")):
            ax[1].plot(x, [r[key] for r in sel], mk, color=col, ms=6, label=lab)
        ax[1].axhline(LIM_E_J, color="k", ls=":", lw=1)
        ax[1].text(len(sel) - 0.6, LIM_E_J * 1.1, "750 J limit", ha="right", fontsize=8)
        ax[1].set_yscale("log")
        ax[1].set_ylim(1.5, 1500)
        ax[1].set_xlim(-0.5, len(sel) - 0.5)
        ax[1].set_xticks(x)
        ax[1].set_xticklabels([NAMES[r["modello"]].replace("MS-RCFE+Ridge", "MS-RCFE").replace("InceptionTime", "Incep.").replace(" ", "\n") for r in sel], fontsize=7)
        ax[1].set_ylabel("Energy until detection [J]")
        ax[1].set_title("(b) with detection (K = 1, 0 ms actuation)", fontsize=9)
        ax[1].legend(fontsize=7, frameon=False, loc="lower center", ncol=3, columnspacing=0.8, handletextpad=0.3)
        ax[1].grid(alpha=0.25, axis="y")
    fig.tight_layout()
    savefig(fig, out, "fig_energy")
    plt.close(fig)
    note("figures/fig_energy.{pdf,png}", f"{p.relative_to(res)} (sha {sha(p)})")


# ------------------------------------------------------------------ tables
def tab_delay(res, out):
    p = find(res, "ris_ritardo/ris_ritardo/riepilogo_ritardo.json", "ris_ritardo/riepilogo_ritardo.json")
    if p is None:
        print("skip tab_delay")
        return
    j = json.loads(p.read_text(encoding="utf-8"))
    rows = []
    for r in sorted(j["riepilogo"], key=lambda r: (MODELS.index(r["modello"]), r["K"])):
        rows.append([NAMES[r["modello"]], str(r["K"]), f"{r['mediana_ms']:.0f}", f"{r['p95_ms']:.0f}",
                     f"{r['massimo_ms']:.0f}", str(r["esperimenti_con_falsi_allarmi_pre"])])
    write_tex(out / "tables/tab_delay.tex", ["Model", "K", "Median [ms]", "P95 [ms]", "Max [ms]", "Exp.\\ with pre-onset alarms"],
              rows, "lrrrrr")
    note("tables/tab_delay.tex", f"{p.relative_to(res)} (sha {sha(p)}); n = {j['n_esperimenti_con_arco']} experiments with arc")


def tab_energy(res, out):
    j, p = load_energy(res)
    if j is None:
        print("skip tab_energy")
        return
    rows = []
    sel = [r for r in j["con_rilevazione"] if r["attuazione_ms"] == 0.0]
    for r in sorted(sel, key=lambda r: (MODELS.index(r["modello"]), r["K"])):
        rows.append([NAMES[r["modello"]], str(r["K"]), f"{r['mediana_J']:.1f}", f"{r['p95_J']:.1f}", f"{r['massimo_J']:.1f}"])
    write_tex(out / "tables/tab_energy.tex", ["Model", "K", "Median [J]", "P95 [J]", "Max [J]"], rows, "lrrrr")
    note("tables/tab_energy.tex", f"{p.relative_to(res)} (sha {sha(p)}); series resistance {j['args'].get('r_serie')} ohm "
                                  "(upper bound), actuation 0 ms")


def sens_df(res):
    raw, src = read_zip_or_dir(res, "ris_sensibilita.zip", "sensibilita_etichette.csv")
    if raw is None:
        return None, None
    return pd.read_csv(io.BytesIO(raw)), src


def tab_classification_sens(res, out):
    d, src = sens_df(res)
    if d is None:
        print("skip tab_classification / tab_sensitivity: sensibilita_etichette.csv not found")
        return
    base = d[d.variante == "base"]
    rows = []
    for _, r in base.iterrows():
        rows.append([r.modello, ci(r.DR, r.DR_lo, r.DR_hi), ci(r.FPR, r.FPR_lo, r.FPR_hi, 2), str(int(r.FN)), str(int(r.FP))])
    n = base.iloc[0]
    write_tex(out / "tables/tab_classification.tex", ["Model", "DR [\\%] (95\\% CI)", "FPR [\\%] (95\\% CI)", "FN", "FP"], rows, "lrrrr")
    note("tables/tab_classification.tex", f"{src}; base labels: {int(n.n_arco)} arc and {int(n.n_pre)} pre-arc windows; CI bootstrap per experiment")
    rows = []
    for v in d.variante.unique():
        dv = d[d.variante == v]
        cells = []
        for m in LONG:
            r = dv[dv.modello == m]
            cells.append(f"{r.DR.iloc[0]:.1f}/{r.FPR.iloc[0]:.1f}" if len(r) else "--")
        rows.append([tex_escape(v)] + cells)
    write_tex(out / "tables/tab_sensitivity.tex", ["Labelling variant"] + [f"{NAMES[LONG[m]]}" for m in LONG],
              rows, "lrrrr")
    note("tables/tab_sensitivity.tex", f"{src}; each cell = DR/FPR in %; models are NOT retrained")


def tab_bands(etv2, out):
    p = Path(etv2) / "valutazione_etichette.csv"
    if not p.exists():
        print("skip tab_level_bands: valutazione_etichette.csv not found")
        return
    d = pd.read_csv(p)
    d = d[(d.etichette == "tensione") & (d.file_set == "tutti i file") & (d.fascia_livello != "tutte")]
    rows = []
    for m in LONG:
        cells = []
        for fa in ("<0.90", "0.90-0.95", ">=0.95"):
            r = d[(d.modello == m) & (d.fascia_livello == fa)]
            cells.append(f"{r.DR.iloc[0]:.1f} [{r.DR_lo.iloc[0]:.1f}, {r.DR_hi.iloc[0]:.1f}] (n={int(r.n_finestre.iloc[0])})" if len(r) else "--")
        rows.append([m] + cells)
    write_tex(out / "tables/tab_level_bands.tex", ["Model", "level $<0.90$", "$0.90$--$0.95$", "$\\geq 0.95$"], rows, "lccc")
    note("tables/tab_level_bands.tex", f"{p.name} (sha {sha(p)}); DR on windows labelled as arc by the voltage-based onset, all files")


# ------------------------------------------------------------------ main
def run(res, out, etv2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    fig_delay(res, out, plt)
    fig_energy(res, out, plt)
    tab_delay(res, out)
    tab_energy(res, out)
    tab_classification_sens(res, out)
    if etv2:
        tab_bands(etv2, out)
    lines = ["# Provenance (make_paper_figures)", "",
             "Limits 2.5 s / 750 J as supplied for UL 1699B (not checked against the standard text).", "",
             "| output | source |", "|---|---|"] + [f"| {a} | {b} |" for a, b in PROV]
    (out / "PROVENANCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Written {len(PROV)} outputs in {out.resolve()} (see PROVENANCE.md)")


def selftest():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        res = Path(td) / "results_data"
        (res / "ris_ritardo/ris_ritardo").mkdir(parents=True)
        (res / "ris_energia2").mkdir(parents=True)
        rows = [(f"f{i}", m, K, 0.3, float(rng.gamma(4, 30) * K), 0) for i in range(60) for m in MODELS for K in (1, 2, 3)]
        pd.DataFrame(rows, columns=["file", "modello", "K", "t1_s", "ritardo_ms", "falsi_allarmi_pre"]).to_csv(
            res / "ris_ritardo/ris_ritardo/ritardo_per_file.csv", index=False)
        (res / "ris_ritardo/ris_ritardo/riepilogo_ritardo.json").write_text(json.dumps(
            dict(n_esperimenti_con_arco=60, riepilogo=[dict(modello=m, K=K, n=60, mancati=0, mediana_ms=100, p95_ms=200,
                                                            massimo_ms=300, esperimenti_con_falsi_allarmi_pre=1)
                                                       for m in MODELS for K in (1, 2, 3)])))
        (res / "ris_energia2/riepilogo_energia.json").write_text(json.dumps(dict(
            args=dict(r_serie=0.0),
            riepilogo=[dict(T_int_ms=t, mediana_J=t / 10, p95_J=t / 6, massimo_J=t / 5, file_massimo="x") for t in (20, 100, 300)],
            con_rilevazione=[dict(modello=m, K=K, attuazione_ms=0.0, mancati=0, n=60, mediana_J=9, p95_J=19, massimo_J=40)
                             for m in MODELS for K in (1, 2, 3)])))
        csv = "variante,soglia,conferma_ms,min_nom_A,n_finestre,n_arco,n_pre,modello,DR,DR_lo,DR_hi,FPR,FPR_lo,FPR_hi,ACC,ACC_lo,ACC_hi,FN,FP\n"
        for v in ("base", "soglia 0.97"):
            for m in LONG:
                csv += f"{v},0.95,300,0.1,7161,4520,2641,{m},99.0,98.0,99.5,0.5,0.1,1.0,99,98,99.5,10,5\n"
        with zipfile.ZipFile(res / "ris_sensibilita.zip", "w") as z:
            z.writestr("ris_sensibilita/sensibilita_etichette.csv", csv)
        out = Path(td) / "out"
        run(res, out, None)
        need = ["figures/fig_delay_ecdf.pdf", "figures/fig_energy.pdf", "tables/tab_delay.tex", "tables/tab_energy.tex",
                "tables/tab_classification.tex", "tables/tab_sensitivity.tex", "PROVENANCE.md"]
        ok = all((out / n).exists() and (out / n).stat().st_size > 0 for n in need)
    print("SELFTEST", "OK" if ok else "FALLITO")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--results", default=None, help="results_data folder (default: <repo>/results_data)")
    ap.add_argument("--out", default="paper_assets")
    ap.add_argument("--etichette-v2", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    qui = Path(__file__).resolve().parent
    repo = Path(a.repo) if a.repo else (qui.parent if (qui.parent / "results_data").is_dir() else Path.cwd())
    res = Path(a.results) if a.results else repo / "results_data"
    if not res.is_dir():
        sys.exit(f"results folder not found: {res}")
    run(res, Path(a.out), a.etichette_v2)


if __name__ == "__main__":
    main()
