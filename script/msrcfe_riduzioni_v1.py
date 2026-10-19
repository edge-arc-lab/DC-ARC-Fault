#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
msrcfe_riduzioni_v1.py - MS-RCFE ridotto (3 e 4 rami) pronto per ST Edge AI: ONNX + Ridge + calibrazione + metriche
====================================================================================================================

Parte dall'estrattore PUBBLICATO (msrcfe_extractor.pt, gia' misurato sulla scheda) e tiene solo alcuni rami convoluzionali.
Per ogni variante, il Ridge e' RIFATTO su TUTTO il training set (alpha=1, come train_msrcfe.py) sulle sole feature dei rami tenuti.

Varianti (rami = kernel k, dilatazione d)
  b3    k3d1 + k3d2 + k3d4                 192 feature, 288 k MAC (17.6% dell'estrattore completo)
  b4    k3d1 + k3d2 + k3d4 + k5d2          256 feature, 448 k MAC (27.5%)
  full  tutti e 9 i rami (riferimento, stessa procedura di export: serve a confrontare ONNX e metriche a parita' di pipeline)
(altre combinazioni con --varianti nome=k3d1,k5d2 ...)

Per ogni variante scrive in --out\<nome>\
  msrcfe_<nome>.onnx         grafo FP32, ingresso [1,1,1000] statico, opset 13 (come export_msrcfe.py): Conv, Relu, ReduceMax, ReduceMean, Concat
  ridge_weights.h            N_FEATURES, RIDGE_INTERCEPT, RIDGE_COEF (stesso formato dell'export originale)
  calibration_<nome>.npz     400 finestre (200 per classe, seme 42) con chiave 'input', forma [N,1,1,1000], per la quantizzazione ST
  metrics.json               DR, FPR, accuratezza bilanciata, AUC sul TEST (feature calcolate dall'ONNX con onnxruntime + Ridge); MAC; confronto numpy/ONNX
e in --out: riepilogo_varianti.csv e LEGGIMI.txt (procedura su ST Edge AI).
Il grafo e' costruito con il pacchetto onnx (senza PyTorch). Se in repo c'e' script/training/msrcfe/export_msrcfe/msrcfe.onnx, il grafo "full" e'
confrontato numericamente con quello (--ref-onnx).

Quantizzazione INT8
  - Per i benchmark sulla scheda: far quantizzare a ST Edge AI Developer Cloud con calibration_<nome>.npz (stessa procedura dei modelli gia' misurati).
  - Con --int8-locale si produce anche msrcfe_<nome>_int8_locale.onnx (onnxruntime, QDQ per canale) e se ne valutano le metriche: e' solo un'ANTEPRIMA dell'effetto
    della quantizzazione; NON coincide con la quantizzazione di ST e le metriche INT8 da riportare sono quelle del modello quantizzato da ST.

Uso (dalla cartella della repo)
    python -B script/msrcfe_riduzioni_v1.py --repo . --out ..\ris_riduzioni
    python -B script/msrcfe_riduzioni_v1.py --repo . --out ..\ris_riduzioni --feat-cache ..\ris_comb\feat_pubblicato.npz --int8-locale
    python -B script/msrcfe_riduzioni_v1.py --repo . --train ..\dataset_V\arc_dataset_train_V.npz --test ..\dataset_V\arc_dataset_test_V.npz --out ..\ris_riduzioni_V
    python -B script/msrcfe_riduzioni_v1.py --selftest
Richiede msrcfe_prune_v1.py nella stessa cartella e i pacchetti numpy, scikit-learn, scipy, onnx, onnxruntime.
Limiti: un solo estrattore (quello pubblicato); la latenza va MISURATA sulla scheda; test set molto separabile per livello.
"""
import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

VERSIONE = "v1"
QUI = Path(__file__).resolve().parent
T = 1000
NK = 32
VARIANTI = {"b3": ["k3d1", "k3d2", "k3d4"], "b4": ["k3d1", "k3d2", "k3d4", "k5d2"],
            "full": [f"k{k}d{d}" for k in (3, 5, 9) for d in (1, 2, 4)]}


def carica_prune():
    sp = importlib.util.spec_from_file_location("pr", QUI / "msrcfe_prune_v1.py")
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def rami_da_nomi(pr, nomi):
    out = []
    for n in nomi:
        k, d = n[1:].split("d")
        out.append(pr.BR.index((int(k), int(d))))
    return sorted(out)


def costruisci_onnx(pr, sd, rami, path):
    """Grafo FP32: per ogni ramo Conv(+bias, dilatazione, padding) -> Relu -> ReduceMax e ReduceMean sull'asse tempo; Concat sull'asse 1.
    Ordine delle feature identico a extract_features: per ramo [max (32), media (32)]."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    nodes, inits, outs = [], [], []
    for r in rami:
        k, d = pr.BR[r]
        W, b = sd[f"convs.{r}.weight"].astype(np.float32), sd[f"convs.{r}.bias"].astype(np.float32)
        wn, bn = f"conv{r}_w", f"conv{r}_b"
        inits += [numpy_helper.from_array(W, wn), numpy_helper.from_array(b, bn)]
        pad = (k // 2) * d
        nodes += [
            helper.make_node("Conv", ["input", wn, bn], [f"c{r}"], kernel_shape=[k], dilations=[d], pads=[pad, pad], strides=[1]),
            helper.make_node("Relu", [f"c{r}"], [f"r{r}"]),
            helper.make_node("ReduceMax", [f"r{r}"], [f"mx{r}"], axes=[2], keepdims=0),
            helper.make_node("ReduceMean", [f"r{r}"], [f"mn{r}"], axes=[2], keepdims=0),
        ]
        outs += [f"mx{r}", f"mn{r}"]
    nodes.append(helper.make_node("Concat", outs, ["features"], axis=1))
    g = helper.make_graph(nodes, "msrcfe_ridotto",
                          [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, T])],
                          [helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 2 * NK * len(rami)])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


def feat_onnx(path, X, bs=1):
    import onnxruntime as ort
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = np.zeros((len(X), s.get_outputs()[0].shape[-1]), np.float32)
    for i in range(len(X)):
        out[i] = s.run(None, {"input": X[i:i + 1, None, :].astype(np.float32)})[0][0]
    return out


def scrivi_header(coef, intercept, path):
    coef = np.asarray(coef, np.float32).ravel()
    with open(path, "w") as f:
        f.write("#ifndef RIDGE_WEIGHTS_H\n#define RIDGE_WEIGHTS_H\n\n")
        f.write(f"#define N_FEATURES {len(coef)}\n\n")
        f.write(f"static const float RIDGE_INTERCEPT = {float(intercept)}f;\n\n")
        f.write("static const float RIDGE_COEF[] = {\n" + ", ".join(f"{v}f" for v in coef) + "\n};\n\n")
        f.write("static inline int ridge_predict(const float* x) {\n    float score = RIDGE_INTERCEPT;\n"
                "    for (int i = 0; i < N_FEATURES; i++)\n        score += x[i] * RIDGE_COEF[i];\n"
                "    return score > 0.0f ? 1 : 0;\n}\n\n#endif /* RIDGE_WEIGHTS_H */\n")


def calibrazione(Xtr, ytr, path, n_per_class=200):
    rng = np.random.default_rng(42)
    i0 = rng.choice(np.where(ytr == 0)[0], min(n_per_class, int((ytr == 0).sum())), replace=False)
    i1 = rng.choice(np.where(ytr == 1)[0], min(n_per_class, int((ytr == 1).sum())), replace=False)
    idx = np.concatenate([i0, i1])
    rng.shuffle(idx)
    np.savez(path, input=Xtr[idx].astype(np.float32)[:, None, None, :])
    return len(i0), len(i1)


def metriche(pr, y, s):
    p = s > 0
    dr, fpr = 100 * float(np.mean(p[y == 1])), 100 * float(np.mean(p[y == 0]))
    return dict(DR=dr, FPR=fpr, BA=(dr + 100 - fpr) / 2, AUC=pr.auc(y, s), FN=int(((~p) & (y == 1)).sum()), FP=int((p & (y == 0)).sum()))


def int8_locale(onnx_fp32, X_cal, out_path):
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    class R(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{"input": x[None, None, :].astype(np.float32)} for x in X_cal])

        def get_next(self):
            return next(self.it, None)
    quantize_static(str(onnx_fp32), str(out_path), R(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)


def selftest():
    pr = carica_prune()
    import onnxruntime as ort
    from sklearn.linear_model import RidgeClassifier
    import tempfile
    sp = importlib.util.spec_from_file_location("sd", QUI / "msrcfe_seed_v1.py")
    sm = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(sm)
    sd = sm.estrattore_casuale(pr, 7)
    X = np.random.default_rng(0).normal(1, 0.05, (6, T)).astype(np.float32)
    ok = True
    with tempfile.TemporaryDirectory() as td:
        for nomi in (VARIANTI["b3"], VARIANTI["b4"], VARIANTI["full"]):
            rami = rami_da_nomi(pr, nomi)
            p = Path(td) / "m.onnx"
            costruisci_onnx(pr, sd, rami, p)
            F = feat_onnx(p, X)
            Fn = pr.feats(sd, X)[:, pr.idx_rami(rami)]
            ok &= F.shape == (6, 64 * len(rami)) and np.allclose(F, Fn, atol=1e-4, rtol=1e-4)
        y = np.array([0, 1, 0, 1, 0, 1])
        clf = RidgeClassifier(alpha=1.0).fit(Fn, y)
        h = Path(td) / "r.h"
        scrivi_header(clf.coef_, clf.intercept_[0], h)
        ok &= "N_FEATURES 576" in h.read_text()
        n0, n1 = calibrazione(X, y, Path(td) / "c.npz", 2)
        ok &= np.load(Path(td) / "c.npz")["input"].shape == (4, 1, 1, T)
    print("SELFTEST", "OK" if ok else "FALLITO")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--train", default=None)
    ap.add_argument("--test", default=None)
    ap.add_argument("--extractor", default=None)
    ap.add_argument("--feat-cache", default=None, help="feat_pubblicato.npz (tr, te: 576 colonne) di msrcfe_comb_v1/msrcfe_prune_v1, per non ricalcolarle")
    ap.add_argument("--ref-onnx", default=None, help="msrcfe.onnx originale per il confronto numerico del grafo completo")
    ap.add_argument("--varianti", nargs="*", default=None, help="nome=k3d1,k5d2 ... (default b3, b4, full)")
    ap.add_argument("--int8-locale", action="store_true")
    ap.add_argument("--out", default="ris_riduzioni")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    from sklearn.linear_model import RidgeClassifier
    pr = carica_prune()
    repo = Path(a.repo) if a.repo else Path.cwd()
    tr_p = Path(a.train) if a.train else repo / "dataset/dataset_new/arc_dataset_train.npz"
    te_p = Path(a.test) if a.test else repo / "dataset/dataset_new/arc_dataset_test.npz"
    ex_p = Path(a.extractor) if a.extractor else repo / "script/training/msrcfe/results/msrcfe_extractor.pt"
    ref_p = Path(a.ref_onnx) if a.ref_onnx else repo / "script/training/msrcfe/export_msrcfe/msrcfe.onnx"
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    var = dict(VARIANTI)
    if a.varianti:
        var = {}
        for v in a.varianti:
            n, r = v.split("=")
            var[n] = r.split(",")
    dtr, dte = np.load(tr_p), np.load(te_p)
    Xtr, ytr, Xte, yte = dtr["X"].astype(np.float32), dtr["y"].astype(int), dte["X"].astype(np.float32), dte["y"].astype(int)
    sd = pr.carica_state(ex_p)
    if a.feat_cache and Path(a.feat_cache).exists() and np.load(a.feat_cache)["tr"].shape[0] == len(Xtr) and np.load(a.feat_cache)["te"].shape[0] == len(Xte):
        z = np.load(a.feat_cache)
        Ftr, Fte = z["tr"], z["te"]
        print("feature lette dalla cache")
    else:
        t0 = time.time()
        Ftr, Fte = pr.feats(sd, Xtr), pr.feats(sd, Xte)
        np.savez(out / "feat_pubblicato.npz", tr=Ftr, te=Fte)
        print(f"feature calcolate in {time.time() - t0:.0f} s")
    righe = []
    macs_full = pr.macs(list(range(9)))
    for nome, nomi in var.items():
        rami = rami_da_nomi(pr, nomi)
        d = out / nome
        d.mkdir(exist_ok=True)
        onx = d / f"msrcfe_{nome}.onnx"
        costruisci_onnx(pr, sd, rami, onx)
        idx = pr.idx_rami(rami)
        clf = RidgeClassifier(alpha=1.0).fit(Ftr[:, idx], ytr)
        np.save(d / "ridge_coef.npy", clf.coef_)
        np.save(d / "ridge_intercept.npy", clf.intercept_)
        scrivi_header(clf.coef_, clf.intercept_[0], d / "ridge_weights.h")
        n0, n1 = calibrazione(Xtr, ytr, d / f"calibration_{nome}.npz")
        # verifica numerica: grafo ONNX contro estrattore numpy (su 50 finestre di test) e metriche sul test intero
        chk = np.random.default_rng(0).choice(len(Xte), 50, replace=False)
        Fo = feat_onnx(onx, Xte[chk])
        diff = float(np.abs(Fo - Fte[chk][:, idx]).max())
        t0 = time.time()
        Fo_te = feat_onnx(onx, Xte)
        s = Fo_te @ clf.coef_.ravel() + clf.intercept_[0]
        m = metriche(pr, yte, s)
        m_np = metriche(pr, yte, clf.decision_function(Fte[:, idx]))
        rec = dict(variante=nome, rami="+".join(nomi), n_feat=len(idx), MAC=pr.macs(rami), frac_MAC=pr.macs(rami) / macs_full,
                   lat_stim_int8_ms=69.77 * pr.macs(rami) / macs_full, lat_stim_fp32_ms=84.75 * pr.macs(rami) / macs_full,
                   max_diff_onnx_numpy=diff, onnx_bytes=onx.stat().st_size, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
                   DR_numpy=round(m_np["DR"], 4), FPR_numpy=round(m_np["FPR"], 4))
        if nome == "full" and ref_p.exists():
            Fr = feat_onnx(ref_p, Xte[chk])
            rec["max_diff_vs_onnx_originale"] = float(np.abs(Fr - Fo).max())
        if a.int8_locale:
            q = d / f"msrcfe_{nome}_int8_locale.onnx"
            cal = np.load(d / f"calibration_{nome}.npz")["input"][:, 0, 0, :]
            int8_locale(onx, cal, q)
            Fq = feat_onnx(q, Xte)
            mq = metriche(pr, yte, Fq @ clf.coef_.ravel() + clf.intercept_[0])
            rec.update({f"INT8loc_{k}": round(v, 4) if isinstance(v, float) else v for k, v in mq.items()})
        json.dump(rec, open(d / "metrics.json", "w"), indent=2)
        righe.append(rec)
        print(f"[{nome:<5}] {rec['n_feat']:>3} feat  MAC {rec['MAC']:>8}  ({100 * rec['frac_MAC']:.1f}%)  DR {m['DR']:.2f}  FPR {m['FPR']:.2f}  BA {m['BA']:.2f}  AUC {m['AUC']:.4f}"
              f"  | diff ONNX-numpy {diff:.1e}" + (f"  | INT8 locale DR {mq['DR']:.2f} FPR {mq['FPR']:.2f}" if a.int8_locale else "")
              + f"  ({time.time() - t0:.0f} s)")
    campi = sorted({k for r in righe for k in r})
    with open(out / "riepilogo_varianti.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campi)
        w.writeheader()
        w.writerows(righe)
    with open(out / "LEGGIMI.txt", "w", encoding="utf-8") as f:
        f.write("""PROCEDURA SU ST EDGE AI (stessa dei modelli gia' misurati)
1. Per ogni variante (cartelle b3, b4, full): in ST Edge AI Developer Cloud caricare msrcfe_<nome>.onnx.
2. Quantizzazione INT8 per canale con il dataset di calibrazione calibration_<nome>.npz (chiave 'input', forma [N,1,1,1000]).
3. Benchmark sulla scheda STM32H7S78-DK (stesso protocollo: tre sessioni, media), in FP32 e in INT8. Annotare latenza (ms), flash, RAM.
4. Il Ridge (ridge_weights.h) non e' nel grafo: la sua parte e' trascurabile (ridge_predict) ma va sommata al tempo totale se si vuole il tempo per finestra.
5. Il benchmark 'full' di questa procedura deve riprodurre 84.75 ms (FP32) e 69.77 ms (INT8): se non coincide, i risultati delle varianti vanno confrontati con QUESTO valore.
6. Le metriche DR/FPR delle varianti (metrics.json) sono calcolate con onnxruntime su CPU in FP32; le metriche INT8 reali richiedono il modello quantizzato da ST.
""")
    print(f"\nScritto in {out}")
    print("Limiti: un solo estrattore (quello pubblicato); latenza stimata per proporzione ai MAC (va misurata); i valori INT8 locali sono solo un'anteprima.")


if __name__ == "__main__":
    main()
