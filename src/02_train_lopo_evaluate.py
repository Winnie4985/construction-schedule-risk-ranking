"""
02_train_lopo_evaluate.py — 隨機森林 + 留一專案交叉驗證(LOPO)

驗證邏輯:
    59 個專案輪流當一次測試集,其餘 58 個當訓練集(共 59 輪)。
    這模擬「模型從未看過這個工地」的真實使用情境,是本專題的核心驗證方式。
    若某工地測試集裡全部任務同一類別(無延遲任務),AUC 無法計算,該輪跳過不計入平均
    (最終 59 輪中約有 56 輪被計入,詳見 reports/methodology_notes.md)。

用法(建議跑 3 個以上不同種子,取平均與標準差以確認結果穩定):
    python 02_train_lopo_evaluate.py --seed 0
    python 02_train_lopo_evaluate.py --seed 1
    python 02_train_lopo_evaluate.py --seed 2

輸出:
    results/final_lopo_results.json   累積每個種子的整體指標 + 每個專案的個別 AUC(供畫熱力圖用)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

LABEL_COL = "y_dur_abs"  # 最終模型使用「絕對量」標籤:實際工期 > 計畫工期即算超時
MIN_TEST_TASKS = 20      # 測試工地任務數低於此不列入評估


def precision_at_k(y_true, y_score, k=10):
    k = min(k, len(y_true))
    top_k_idx = np.argsort(-y_score)[:k]
    return y_true[top_k_idx].mean()


def run_lopo(df, feature_cols, seed):
    """對每個專案跑一輪 LOPO,回傳每輪結果的 list of dict"""
    per_project = []
    for code in df["code"].unique():
        train = df[df["code"] != code]
        test = df[df["code"] == code]
        if test[LABEL_COL].nunique() < 2 or len(test) < MIN_TEST_TASKS:
            continue

        x_train = train[feature_cols].replace([np.inf, -np.inf], np.nan).values
        imputer = SimpleImputer(strategy="median").fit(x_train)

        model = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, random_state=seed,
            n_jobs=-1, class_weight="balanced",
        )
        model.fit(imputer.transform(x_train), train[LABEL_COL])

        x_test = test[feature_cols].replace([np.inf, -np.inf], np.nan).values
        pred = model.predict_proba(imputer.transform(x_test))[:, 1]
        y_true = test[LABEL_COL].values.astype(int)

        per_project.append(dict(
            code=code,
            auc=float(roc_auc_score(y_true, pred)),
            p_at_10=float(precision_at_k(y_true, pred)),
            base_rate=float(y_true.mean()),
            n_tasks=len(test),
        ))
    return per_project


def summarize(per_project):
    auc = np.array([p["auc"] for p in per_project])
    p10 = np.array([p["p_at_10"] for p in per_project])
    base = np.array([p["base_rate"] for p in per_project])
    lift = float(np.nanmean(p10 / np.where(base > 0, base, np.nan)))
    return dict(
        auc=float(auc.mean()), p_at_10=float(p10.mean()), base_rate=float(base.mean()),
        lift=lift, pct_beat_random=float((auc > 0.5).mean() * 100), n_folds=len(auc),
    )


def main(features_csv, out_json, seed):
    df = pd.read_csv(features_csv)
    feature_cols = [c for c in df.columns if c.startswith("f_")]
    df = df.dropna(subset=[LABEL_COL])

    per_project = run_lopo(df, feature_cols, seed)
    summary = summarize(per_project)
    summary["seed"] = seed
    summary["per_project"] = per_project

    print(
        f"seed {seed}: AUC {summary['auc']:.3f}  P@10 {summary['p_at_10']:.3f}  "
        f"基準 {summary['base_rate']:.3f}  提升 {summary['lift']:.2f}x  "
        f">0.5比例 {summary['pct_beat_random']:.0f}%  有效折數 {summary['n_folds']}"
    )

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    all_results = json.load(open(out_json)) if os.path.exists(out_json) else []
    all_results = [r for r in all_results if r["seed"] != seed] + [summary]
    json.dump(all_results, open(out_json, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default="../data/features_ext.csv")
    parser.add_argument("--out", default="../results/final_lopo_results.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    main(args.features, args.out, args.seed)
