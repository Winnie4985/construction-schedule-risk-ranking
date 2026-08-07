"""
05_within_project_contribution.py — 計算「專案內貢獻」與 within_ratio(本專題最重要的方法論修正)

⚠️ 檔案來源說明:
    這支程式是依 docs 交接筆記中對 within_ratio 方法論的文字描述「重建」的版本,
    不是原始就存在、直接搬過來的腳本(原始腳本未包含在整理素材中)。
    邏輯依交接筆記的定義實作,但實際數字請你在自己的環境重新執行一次,
    並對照下面這個健全性檢查:
        交接筆記記載 f_proj_span 的 within_ratio 獨立驗證後約為 23.5%(官方文件寫 24%)。
        如果你重新跑出來的數字落在這個區間附近,代表這支重建腳本邏輯正確;
        如果差異很大,請先檢查抽樣方式或分組門檻是否跟你原本的做法不同,
        再決定要不要把數字寫進報告——這正是「只用驗證過的數字」原則的具體實踐。

為什麼需要這一步:
    39 個特徵中有 8 個在同一專案內數值完全不變(如 f_proj_span 專案總工期)。
    這類特徵的原始 SHAP 重要性經常排名很前面,但那個重要性主要來自「能分辨不同工地」,
    不是「能排序同一工地內的任務」——而後者才是這個模型真正要交付的東西
    (PM 要的是「這個工地裡,先盯哪個任務」,不是「哪個工地比較危險」)。

計算方式:
    整體重要性(overall)   = 該特徵在全部抽樣任務上的平均 |SHAP值|(即 03 的輸出)
    專案內貢獻(within)     = 該特徵在「同一專案內」SHAP 值的標準差,取所有專案的平均
                              (標準差衡量這個特徵能不能拉開同工地任務之間的排名差距)
    within_ratio           = within ÷ overall
                              比值低 → 該特徵的重要性主要來自「分辨不同工地」,
                                        對「工地內任務排序」這個交付物貢獻有限。

輸出:
    results/within_ratio.csv          每個特徵的 overall / within / within_ratio,依 within 排序
    reports/figures/within_ratio_bar.png   長條圖,對比專案層級常數特徵 vs 任務層級特徵
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

LABEL_COL = "y_dur_abs"
SAMPLE_SIZE = 1500
RANDOM_STATE = 0  # 與 03/04 一致,才能重現同一組抽樣
MIN_TASKS_PER_PROJECT = 5  # 專案內抽樣任務數低於此,該專案的標準差不夠可靠,不列入平均

# 已知的 8 個專案層級常數特徵(同一專案內數值完全不變),見 01_build_features.py 檔頭說明
PROJECT_CONSTANT_FEATURES = [
    "f_proj_n_tasks", "f_proj_med_dur", "f_proj_log_cost", "f_proj_span",
    "f_early_spi", "f_early_cpi", "f_early_spit", "f_proj_n_tp",
]


def compute_shap_with_codes(features_csv):
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

    import shap

    rng = np.random.RandomState(RANDOM_STATE)
    n_sample = min(SAMPLE_SIZE, len(x_imputed))
    sample_idx = rng.choice(len(x_imputed), size=n_sample, replace=False)
    x_sample = x_imputed[sample_idx]
    codes_sample = df["code"].values[sample_idx]

    explainer = shap.TreeExplainer(model)
    shap_values_all = explainer.shap_values(x_sample, check_additivity=False)
    shap_values = shap_values_all[:, :, 1]
    return feature_cols, shap_values, codes_sample


def compute_within_ratio(feature_cols, shap_values, codes_sample):
    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df["code"] = codes_sample

    overall = np.abs(shap_values).mean(axis=0)

    within_by_feature = {}
    for j, feature in enumerate(feature_cols):
        stds = []
        for _code, group in shap_df.groupby("code"):
            if len(group) < MIN_TASKS_PER_PROJECT:
                continue
            stds.append(group[feature].std())
        within_by_feature[feature] = float(np.mean(stds)) if stds else float("nan")

    result = pd.DataFrame({
        "feature": feature_cols,
        "overall_importance": overall,
        "within_project_contribution": [within_by_feature[f] for f in feature_cols],
    })
    result["within_ratio"] = result["within_project_contribution"] / result["overall_importance"]
    result["is_project_constant"] = result["feature"].isin(PROJECT_CONSTANT_FEATURES)
    return result.sort_values("within_project_contribution", ascending=False).reset_index(drop=True)


def main(features_csv, out_csv, out_png):
    print("步驟 1/2:訓練模型、計算 SHAP 值(與 03_shap_importance.py 相同設定)...")
    feature_cols, shap_values, codes_sample = compute_shap_with_codes(features_csv)

    print("步驟 2/2:計算專案內貢獻與 within_ratio...")
    result = compute_within_ratio(feature_cols, shap_values, codes_sample)
    result.to_csv(out_csv, index=False)
    print(f"已存到: {out_csv}\n")
    print("依「專案內貢獻」排序的前 10 名(這才是同工地內任務排序真正倚賴的特徵):")
    print(result.head(10).to_string(index=False))

    span_row = result[result["feature"] == "f_proj_span"]
    if not span_row.empty:
        ratio = span_row["within_ratio"].iloc[0] * 100
        print(f"\n健全性檢查:f_proj_span 的 within_ratio = {ratio:.1f}%"
              f"(交接筆記記載獨立驗證值約 23.5%,官方文件約 24%)")
        if abs(ratio - 24) > 8:
            print("  ⚠️ 差異較大,寫進報告前請先確認抽樣方式/分組門檻與原始做法是否一致。")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = result.head(15).iloc[::-1]
    colors = ["#c0392b" if c else "#2a78d6" for c in top["is_project_constant"]]
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["within_project_contribution"], color=colors)
    plt.xlabel("within-project SHAP std (mean over projects)")
    plt.title("Within-project contribution (red = project-level constant feature)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"\n長條圖已存到: {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default="../data/features_ext.csv")
    parser.add_argument("--out-csv", default="../results/within_ratio.csv")
    parser.add_argument("--out-png", default="../reports/figures/within_ratio_bar.png")
    args = parser.parse_args()
    main(args.features, args.out_csv, args.out_png)
