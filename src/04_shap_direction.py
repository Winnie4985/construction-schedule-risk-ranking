"""
04_shap_direction.py — 補算每個特徵「原始值 vs SHAP值」的相關方向,合併進 shap_importance.csv

必須先跑過 03_shap_importance.py,再跑這支(需要它輸出的 shap_importance.csv)。

⚠️ 重要 — SHAP 條件方向不等於管理建議方向:
    這支算出來的是「SHAP 條件方向」(這個特徵的 SHAP 值 vs 這個特徵的原始值的相關)。
    寫管理建議時一律要用「原始邊際方向」(這個特徵的原始值 vs 實際延遲與否的相關),
    兩者在大多數特徵上方向一致,但在 CPM 浮時、任務時程位置上方向相反,
    此時以原始邊際方向為準,SHAP 條件方向的落差在報告中僅能列為「待驗證假說」,
    不能當作已證實的因果解讀。詳見 reports/methodology_notes.md 第四節。

輸出:
    results/shap_importance_with_direction.csv
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

LABEL_COL = "y_dur_abs"
SAMPLE_SIZE = 1500
RANDOM_STATE = 0  # 需與 03_shap_importance.py 一致,才能重現同一組抽樣


def main(features_csv, importance_csv, out_csv):
    print("步驟 1/3:讀資料、訓練模型(與 03_shap_importance.py 同一套設定)...")
    df = pd.read_csv(features_csv)
    feature_cols = [c for c in df.columns if c.startswith("f_")]
    df = df.dropna(subset=[LABEL_COL]).reset_index(drop=True)

    x = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    x_imputed = imputer.fit_transform(x)

    model = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5, random_state=RANDOM_STATE,
        n_jobs=-1, class_weight="balanced",
    )
    model.fit(x_imputed, df[LABEL_COL])

    print("步驟 2/3:計算 SHAP 值(抽樣 1500 筆,約 1~3 分鐘)...")
    t0 = time.time()
    import shap

    rng = np.random.RandomState(RANDOM_STATE)
    n_sample = min(SAMPLE_SIZE, len(x_imputed))
    sample_idx = rng.choice(len(x_imputed), size=n_sample, replace=False)
    x_sample = x_imputed[sample_idx]

    explainer = shap.TreeExplainer(model)
    shap_values_all = explainer.shap_values(x_sample, check_additivity=False)
    shap_values = shap_values_all[:, :, 1]
    print(f"  完成。花費 {time.time() - t0:.1f} 秒")

    print("步驟 3/3:計算相關係數、合併存檔...")
    rows = []
    for i, feature in enumerate(feature_cols):
        rho, p_value = spearmanr(x_sample[:, i], shap_values[:, i])
        # 原始邊際方向:特徵原始值 vs 實際標籤(不是 SHAP值),管理建議一律以此為準
        raw_rho, raw_p = spearmanr(df[feature].values, df[LABEL_COL].values)
        rows.append((feature, rho, p_value, raw_rho, raw_p))
    direction_df = pd.DataFrame(
        rows, columns=["feature", "spearman_corr_shap_conditional", "p_value",
                        "spearman_corr_raw_marginal", "raw_p_value"]
    )

    importance_df = pd.read_csv(importance_csv)
    merged = importance_df.merge(direction_df, on="feature").sort_values("mean_abs_shap", ascending=False)
    merged.to_csv(out_csv, index=False)

    print(f"\n已存到: {out_csv}")
    print("\n前 15 名(重要性排名 + SHAP 條件方向,注意上方檔頭的方向使用限制):")
    print(merged.head(15).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default="../data/features_ext.csv")
    parser.add_argument("--importance-csv", default="../results/shap_importance.csv")
    parser.add_argument("--out", default="../results/shap_importance_with_direction.csv")
    args = parser.parse_args()
    main(args.features, args.importance_csv, args.out)
