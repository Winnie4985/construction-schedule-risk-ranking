"""
experiments/algorithm_comparison.py — 演算法選型比較(隨機森林 vs 梯度提升 vs 邏輯迴歸)

⚠️ 版本注意:此比較在早期探索版本(22 特徵、features.csv)上進行,非最終版本,
   結論(隨機森林為最佳基準,梯度提升無明顯改善,邏輯迴歸明顯較差)在特徵工程
   後續擴充到 39 個特徵後未重新驗證,列為既有結論但非最終版本的直接證據。

用法:
    python algorithm_comparison.py --label y_dur_abs

輸出(印在終端機,並存成 CSV 供簡報引用):
    results/algorithm_comparison.csv
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def random_forest():
    return RandomForestClassifier(n_estimators=80, min_samples_leaf=5, random_state=0, n_jobs=-1, class_weight="balanced")


def gradient_boosting():
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.06, max_leaf_nodes=15,
        min_samples_leaf=20, l2_regularization=1.0, random_state=0,
    )


def logistic_regression():
    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )


MODELS = {"random_forest": random_forest, "gradient_boosting": gradient_boosting, "logistic_regression": logistic_regression}


def precision_at_k(y_true, y_score, k=10):
    k = min(k, len(y_true))
    return y_true[np.argsort(-y_score)[:k]].mean()


def run_lopo(df, feature_cols, label, model_fn):
    d = df.dropna(subset=[label])
    aucs, p10s, bases = [], [], []
    for code in d["code"].unique():
        train, test = d[d["code"] != code], d[d["code"] == code]
        if test[label].nunique() < 2 or len(test) < 20 or train[label].nunique() < 2:
            continue
        x_train = train[feature_cols].replace([np.inf, -np.inf], np.nan).values
        x_test = test[feature_cols].replace([np.inf, -np.inf], np.nan).values
        imputer = SimpleImputer(strategy="median").fit(x_train)
        model = model_fn()
        model.fit(imputer.transform(x_train), train[label])
        pred = model.predict_proba(imputer.transform(x_test))[:, 1]
        y_true = test[label].values.astype(int)
        aucs.append(roc_auc_score(y_true, pred))
        p10s.append(precision_at_k(y_true, pred))
        bases.append(y_true.mean())
    auc, p10, base = np.array(aucs), np.array(p10s), np.array(bases)
    lift = np.nanmean(p10 / np.where(base > 0, base, np.nan))
    return dict(n=len(auc), auc=float(auc.mean()), pct_beat_random=float((auc > 0.5).mean() * 100),
                p10=float(p10.mean()), lift=float(lift))


def main(features_csv, label, out_csv):
    df = pd.read_csv(features_csv)
    feature_cols = [c for c in df.columns if c.startswith("f_")]

    print(f"{'模型':40}{'折數':>5}{'AUC':>9}{'>0.5比例':>9}{'P@10':>9}{'提升':>8}")
    rows = []
    for name, model_fn in MODELS.items():
        result = run_lopo(df, feature_cols, label, model_fn)
        result["model"] = name
        rows.append(result)
        print(f"{name:40}{result['n']:>5}{result['auc']:>9.3f}{result['pct_beat_random']:>8.0f}%"
              f"{result['p10']:>9.3f}{result['lift']:>7.2f}x")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n已存到: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default="../../data/features.csv")
    parser.add_argument("--label", default="y_dur_abs")
    parser.add_argument("--out", default="../../results/algorithm_comparison.csv")
    args = parser.parse_args()
    main(args.features, args.label, args.out)
