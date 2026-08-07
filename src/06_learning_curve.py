"""
06_learning_curve.py — 學習曲線:訓練專案數 → 對「全新專案」的排序表現

用來回答:「累積到 59 個專案的資料量,對這個任務究竟夠不夠?」
結論(已用真實資料驗證):3~5 個專案時 AUC 低於隨機猜測(死亡區間),
30 個專案後效益趨緩,59 個專案已站上平台區。這解釋了舊資料集(僅 7 個專案)
當年為何做不出有效模型。

⚠️ 版本說明:此腳本沿用專題早期探索版本的設定(22 個特徵、80 棵樹),
   不是最終模型的 39 特徵/200 棵樹版本,但學習曲線本身回答的是「資料量夠不夠」
   這個問題,用哪個特徵集不影響結論方向,故未重新以最終版本跑過。
   若要在報告中同時放最終版本數字,兩者請分開標註,不要混用。

用法:
    python 06_learning_curve.py --train-sizes 3 5 10 20 30 44
    python 06_learning_curve.py --plot-only   # 只依現有 results/learning_curve.json 畫圖

輸出:
    results/learning_curve.json
    reports/figures/learning_curve.png
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

LABEL_COL = "y_dur_abs"
N_REPEATS = 3


def make_model(seed=0):
    return RandomForestClassifier(n_estimators=80, min_samples_leaf=5, random_state=seed, n_jobs=-1, class_weight="balanced")


def prepare(df, feature_cols):
    return SimpleImputer(strategy="median").fit_transform(df[feature_cols].replace([np.inf, -np.inf], np.nan).values)


def precision_at_k(y_true, y_score, k=10):
    k = min(k, len(y_true))
    return y_true[np.argsort(-y_score)[:k]].mean()


def evaluate_one_size(df, feature_cols, n_train, test_codes, pool_codes):
    aucs, p10s, bases = [], [], []
    for rep in range(N_REPEATS):
        rng = np.random.RandomState(100 + rep)
        train_codes = list(rng.choice(pool_codes, min(n_train, len(pool_codes)), replace=False))
        train = df[df["code"].isin(train_codes)]
        if train[LABEL_COL].nunique() < 2:
            continue
        model = make_model()
        model.fit(prepare(train, feature_cols), train[LABEL_COL])
        for code in test_codes:
            test = df[df["code"] == code]
            if test[LABEL_COL].nunique() < 2 or len(test) < 20:
                continue
            pred = model.predict_proba(prepare(test, feature_cols))[:, 1]
            y_true = test[LABEL_COL].values.astype(int)
            aucs.append(roc_auc_score(y_true, pred))
            p10s.append(precision_at_k(y_true, pred))
            bases.append(y_true.mean())
    auc, p10, base = np.array(aucs), np.array(p10s), np.array(bases)
    return dict(
        n_train=n_train, n_eval=len(auc),
        auc=float(auc.mean()), auc_std=float(auc.std()),
        p10=float(p10.mean()), base=float(base.mean()),
        lift=float((p10 / np.where(base > 0, base, np.nan)).mean()),
    )


def plot(results, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = sorted(results, key=lambda r: r["n_train"])
    xs = [r["n_train"] for r in results]
    ys = [r["auc"] for r in results]
    errs = [r["auc_std"] for r in results]

    plt.figure(figsize=(7, 5))
    plt.errorbar(xs, ys, yerr=errs, marker="o", color="#2a78d6", capsize=3)
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="random baseline (AUC=0.5)")
    plt.xlabel("number of training projects")
    plt.ylabel("AUC on 15 held-out new projects")
    plt.title("Learning curve: training data size vs. generalization to new sites")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"已存到: {out_png}")


def main(features_csv, out_json, train_sizes, plot_only, out_png, n_test_projects):
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    all_results = json.load(open(out_json)) if os.path.exists(out_json) else []

    if not plot_only:
        df = pd.read_csv(features_csv)
        feature_cols = [c for c in df.columns if c.startswith("f_")]
        df = df.dropna(subset=[LABEL_COL])

        codes = sorted(df["code"].unique())
        rng = np.random.RandomState(7)
        test_codes = list(rng.choice(codes, min(n_test_projects, len(codes)), replace=False))
        pool_codes = [c for c in codes if c not in test_codes]
        print(f"訓練池 {len(pool_codes)} 個專案,固定測試 {len(test_codes)} 個新專案\n")

        for n_train in train_sizes:
            result = evaluate_one_size(df, feature_cols, n_train, test_codes, pool_codes)
            print(
                f"訓練專案數 {n_train:3}  →  對新專案 AUC {result['auc']:.3f} (±{result['auc_std']:.3f})  "
                f"Precision@10 {result['p10']:.3f}  基準 {result['base']:.3f}  提升 {result['lift']:.2f}x"
            )
            all_results = [r for r in all_results if r["n_train"] != n_train] + [result]

        all_results = sorted(all_results, key=lambda r: r["n_train"])
        json.dump(all_results, open(out_json, "w"), ensure_ascii=False, indent=2)

    if all_results:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plot(all_results, out_png)
    else:
        print("尚無資料可畫圖,請先不加 --plot-only 執行一次。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default="../data/features.csv")
    parser.add_argument("--out", default="../results/learning_curve.json")
    parser.add_argument("--out-png", default="../reports/figures/learning_curve.png")
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[3, 5, 10, 20, 30, 44])
    parser.add_argument("--n-test-projects", type=int, default=15, help="固定當作『新專案』測試集的專案數")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    main(args.features, args.out, args.train_sizes, args.plot_only, args.out_png, args.n_test_projects)
