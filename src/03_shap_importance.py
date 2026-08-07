"""
03_shap_importance.py — 計算 SHAP 特徵重要性(工期超時模型, y_dur_abs, 39 特徵, 200 棵樹)

只解釋工期超時這一個模型,不做工期 vs 成本比較(成本這條線已放棄,詳見 methodology_notes.md)。

方法說明(可直接寫進報告方法論段落):
    - 訓練模型用全部有效專案的資料(不是 LOPO 那種一次拿掉一個的做法),
      因為 SHAP 要解釋的是「最終要拿去用的那個模型」,不是驗證用的子模型。
    - SHAP 值本身是隨機抽樣 1500 筆任務去算,不是全部任務——這是常見做法,
      因為 SHAP 計算量大,抽樣不影響排名結果,只是為了讓程式在幾分鐘內跑完。

⚠️ 重要:這裡輸出的是「整體 SHAP 重要性」,也就是還沒扣除「專案層級常數特徵」
   干擾的原始排名。要看「同一工地內任務排序」真正有貢獻的特徵,
   請接著跑 05_within_project_contribution.py,不要只憑這支的排名下結論。

輸出:
    reports/results/shap_importance.csv     每個特徵的平均|SHAP值|,由大到小排序
    reports/figures/shap_importance_bar.png 長條圖版本
    reports/figures/shap_beeswarm.png       每個特徵的 SHAP 值分布圖
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

LABEL_COL = "y_dur_abs"
SAMPLE_SIZE = 1500
RANDOM_STATE = 0


def compute_shap_values(features_csv):
    print("步驟 1/3:讀資料、訓練模型...")
    t0 = time.time()
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
    print(f"  完成,用了 {len(df)} 筆任務、{len(feature_cols)} 個特徵。花費 {time.time() - t0:.1f} 秒")

    print("步驟 2/3:計算 SHAP 值(抽樣 1500 筆,約 1~3 分鐘,請耐心等)...")
    t0 = time.time()
    import shap

    rng = np.random.RandomState(RANDOM_STATE)
    n_sample = min(SAMPLE_SIZE, len(x_imputed))
    sample_idx = rng.choice(len(x_imputed), size=n_sample, replace=False)
    x_sample = x_imputed[sample_idx]
    x_sample_df = pd.DataFrame(x_sample, columns=feature_cols)

    explainer = shap.TreeExplainer(model)
    shap_values_all = explainer.shap_values(x_sample, check_additivity=False)
    shap_values = shap_values_all[:, :, 1]  # 取「延遲」這一類的 SHAP 值
    print(f"  完成,抽樣 {n_sample} 筆計算 SHAP。花費 {time.time() - t0:.1f} 秒")

    return feature_cols, shap_values, x_sample_df, df


def main(features_csv, out_csv, out_bar_png, out_beeswarm_png):
    feature_cols, shap_values, x_sample_df, _df = compute_shap_values(features_csv)

    print("步驟 3/3:整理特徵重要性排名並畫圖...")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = (
        pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(out_csv, index=False)
    print(f"  已存到: {out_csv}")
    print("\n  前 15 名特徵(⚠️ 尚未扣除專案層級常數特徵的干擾,見檔頭說明):")
    print(importance_df.head(15).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    top = importance_df.head(15).iloc[::-1]
    plt.barh(top["feature"], top["mean_abs_shap"], color="#2a78d6")
    plt.xlabel("mean |SHAP value|")
    plt.title("Feature importance - duration overrun model (raw, before within-project correction)")
    plt.tight_layout()
    plt.savefig(out_bar_png, dpi=150)
    print(f"  長條圖已存到: {out_bar_png}")

    plt.figure()
    import shap
    shap.summary_plot(shap_values, x_sample_df, show=False)
    plt.tight_layout()
    plt.savefig(out_beeswarm_png, dpi=150)
    print(f"  beeswarm 圖已存到: {out_beeswarm_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default="../data/features_ext.csv")
    parser.add_argument("--out-csv", default="../results/shap_importance.csv")
    parser.add_argument("--out-bar", default="../reports/figures/shap_importance_bar.png")
    parser.add_argument("--out-beeswarm", default="../reports/figures/shap_beeswarm.png")
    args = parser.parse_args()
    main(args.features, args.out_csv, args.out_bar, args.out_beeswarm)
