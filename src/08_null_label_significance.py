"""
08_null_label_significance.py — 打亂標籤對照檢定(統計顯著性)+ 專案內單獨建模的探索

目的:回答「模型學到的是不是真訊號,還是統計巧合?」
做法:把每個專案內的標籤隨機打亂(保留每個專案自己的正例比例不變),
重新跑一次 LOPO,得到「雜訊水準」的 AUC。若真實 AUC 明顯高於雜訊水準
(例如相差 3 個標準差以上),代表模型學到的不是巧合。

⚠️ 版本注意:這支程式與 06/07 一樣沿用早期探索版本設定(欄位來自 features.csv,
   22 特徵、80 棵樹),不是最終 39 特徵/200 棵樹版本。
   若要在報告中引用「雜訊水準 vs 真實成績」的比較,兩者要用同一個特徵/模型設定
   才公平,建議之後補跑一次 39 特徵版本(把 --features 換成 features_ext.csv 重跑
   stage=lopo 與 stage=lopo_null 即可,程式邏輯不用改)。

用法:
    python 08_null_label_significance.py lopo          --label y_dur_abs
    python 08_null_label_significance.py lopo_null      --label y_dur_abs
    python 08_null_label_significance.py by_category    --label y_dur_abs
    python 08_null_label_significance.py within_project_cv       --label y_dur_abs
    python 08_null_label_significance.py within_project_cv_null  --label y_dur_abs

輸出:
    results/significance_{label}.json (累積寫入,每個 stage 各自一個 key)
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


def make_model():
    return RandomForestClassifier(n_estimators=80, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced")


def prepare(df, feature_cols):
    return SimpleImputer(strategy="median").fit_transform(df[feature_cols].replace([np.inf, -np.inf], np.nan).values)


def lopo(label, df, feature_cols, shuffle=False, seed=RANDOM_STATE):
    """留一專案交叉驗證;shuffle=True 時,在各專案內部打亂標籤(對照組)"""
    d = df.dropna(subset=[label]).copy()
    if shuffle:
        rng = np.random.RandomState(seed)
        d[label] = d.groupby("code")[label].transform(lambda s: rng.permutation(s.values))
    out = []
    for code in d["code"].unique():
        train, test = d[d["code"] != code], d[d["code"] == code]
        if test[label].nunique() < 2 or len(test) < 20 or train[label].nunique() < 2:
            continue
        model = make_model()
        model.fit(prepare(train, feature_cols), train[label])
        pred = model.predict_proba(prepare(test, feature_cols))[:, 1]
        out.append((code, float(roc_auc_score(test[label], pred)), len(test)))
    return out


def within_project_cv(label, df, feature_cols, shuffle=False, seed=RANDOM_STATE, min_tasks=40):
    """單一專案內部 5 折交叉驗證(不借用其他專案資料),用來看『只靠自己工地的歷史資料』能做到多好"""
    d = df.dropna(subset=[label])
    out = []
    for code, group in d.groupby("code"):
        if len(group) < min_tasks or group[label].nunique() < 2:
            continue
        y = group[label].values.astype(int)
        if shuffle:
            y = np.random.RandomState(seed).permutation(y)
        if min(np.bincount(y)) < 5:
            continue
        x = prepare(group, feature_cols)
        try:
            skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
            pred = np.zeros(len(y))
            for train_idx, test_idx in skf.split(x, y):
                model = make_model()
                model.fit(x[train_idx], y[train_idx])
                pred[test_idx] = model.predict_proba(x[test_idx])[:, 1]
            k = min(10, len(y))
            top_k = np.argsort(-pred)[:k]
            out.append((code, float(roc_auc_score(y, pred)), len(y), float(y.mean()), float(y[top_k].mean())))
        except ValueError:
            pass
    return out


def load_results(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def main(stage, label, features_csv, out_dir):
    df = pd.read_csv(features_csv)
    feature_cols = [c for c in df.columns if c.startswith("f_")]

    out_path = os.path.join(out_dir, f"significance_{label}.json")
    os.makedirs(out_dir, exist_ok=True)
    results = load_results(out_path)

    if stage == "lopo":
        detail = lopo(label, df, feature_cols)
        auc = np.array([x[1] for x in detail])
        results["lopo"] = dict(n=len(auc), mean=float(auc.mean()), median=float(np.median(auc)),
                                pct_beat_random=float((auc > 0.5).mean() * 100), detail=detail)
        print(f"[全域 LOPO] {len(auc)} 折  平均 AUC {auc.mean():.3f}  中位 {np.median(auc):.3f}  >0.5 比例 {(auc > 0.5).mean() * 100:.0f}%")

    elif stage == "lopo_null":
        means = []
        for seed in range(3):
            detail = lopo(label, df, feature_cols, shuffle=True, seed=seed)
            if detail:
                means.append(float(np.mean([x[1] for x in detail])))
        results["lopo_null"] = dict(mean=float(np.mean(means)), std=float(np.std(means)), vals=means)
        print(f"[LOPO 打亂標籤對照] {np.mean(means):.3f} ± {np.std(means):.3f}")

    elif stage == "by_category":
        for category in df["cat"].unique():
            detail = lopo(label, df[df["cat"] == category], feature_cols)
            if detail:
                auc = np.array([x[1] for x in detail])
                results[f"lopo_{category}"] = dict(n=len(auc), mean=float(auc.mean()),
                                                     pct_beat_random=float((auc > 0.5).mean() * 100))
                print(f"[只用{category}] {len(auc)} 折  平均 AUC {auc.mean():.3f}  >0.5 比例 {(auc > 0.5).mean() * 100:.0f}%")

    elif stage == "within_project_cv":
        detail = within_project_cv(label, df, feature_cols)
        auc = np.array([x[1] for x in detail])
        base = np.array([x[3] for x in detail])
        p10 = np.array([x[4] for x in detail])
        lift = np.divide(p10, base, out=np.zeros_like(p10), where=base > 0)
        results["within_project_cv"] = dict(
            n=len(auc), mean=float(auc.mean()), median=float(np.median(auc)),
            pct_beat_random=float((auc > 0.5).mean() * 100), p10=float(p10.mean()),
            base=float(base.mean()), lift=float(lift.mean()),
            detail=[(c, float(a), int(n), float(b), float(p)) for c, a, n, b, p in detail],
        )
        print(f"[單專案內部CV] {len(auc)} 專案  平均 AUC {auc.mean():.3f}  中位 {np.median(auc):.3f}  >0.5 比例 {(auc > 0.5).mean() * 100:.0f}%")
        print(f"   Precision@10 {p10.mean():.3f} (基準 {base.mean():.3f})  提升 {lift.mean():.2f}x")

    elif stage == "within_project_cv_null":
        means = []
        for seed in range(3):
            detail = within_project_cv(label, df, feature_cols, shuffle=True, seed=seed)
            if detail:
                means.append(float(np.mean([x[1] for x in detail])))
        results["within_project_cv_null"] = dict(mean=float(np.mean(means)), std=float(np.std(means)), vals=means)
        print(f"[單專案內部CV 打亂標籤對照] {np.mean(means):.3f} ± {np.std(means):.3f}")

    else:
        raise ValueError(f"未知 stage: {stage}")

    json.dump(results, open(out_path, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["lopo", "lopo_null", "by_category", "within_project_cv", "within_project_cv_null"])
    parser.add_argument("--label", default="y_dur_abs", choices=["y_dur_abs", "y_dur_rel"])
    parser.add_argument("--features", default="../data/features.csv")
    parser.add_argument("--out-dir", default="../results")
    args = parser.parse_args()
    main(args.stage, args.label, args.features, args.out_dir)
