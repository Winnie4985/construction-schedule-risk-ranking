"""
07_deployment_comparison.py — 實務部署比較:一個全域模型 vs 每工地一個模型 vs 混合

情境:專案進行到一半,前 50% 任務已完工(知道答案),要預測後 50% 還沒做的任務會不會超時。
這是比 LOPO 更貼近實際使用情境的測試:LOPO 測的是「對全新工地」,
這裡測的是「工地做到一半,能不能用其他工地的經驗 + 自己前半段的資料一起預測」。

三種做法:
    A. 只用其他專案的資料訓練(單一全域模型)
    B. 只用這個專案自己的前半段訓練
    C. 混合:其他專案 + 自己的前半段

結論(已用真實資料驗證):三種做法頭對頭勝負接近五成,基本打平。
實務建議:不需要每個工地各建一個模型,全部工地共用一個全域模型即可,
維運成本更低,效果沒有明顯損失。

用法:
    python 07_deployment_comparison.py

輸出:
    results/deploy_comparison.csv
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

LABEL_COL = "y_dur_abs"
MIN_TASKS = 40


def make_model(seed=0):
    return RandomForestClassifier(n_estimators=80, min_samples_leaf=5, random_state=seed, n_jobs=-1, class_weight="balanced")


def prepare(df, feature_cols):
    return SimpleImputer(strategy="median").fit_transform(df[feature_cols].replace([np.inf, -np.inf], np.nan).values)


def precision_at_k(y_true, y_score, k=10):
    k = min(k, len(y_true))
    return y_true[np.argsort(-y_score)[:k]].mean()


def evaluate_project(code, df, feature_cols):
    test_all = df[df["code"] == code].sort_values("f_rel_pos")  # 依時間順序
    if len(test_all) < MIN_TASKS or test_all[LABEL_COL].nunique() < 2:
        return None
    half = len(test_all) // 2
    early, late = test_all.iloc[:half], test_all.iloc[half:]
    if late[LABEL_COL].nunique() < 2:
        return None

    others = df[df["code"] != code]
    y_true = late[LABEL_COL].values.astype(int)
    x_late = prepare(late, feature_cols)
    out = {"code": code, "n_late": len(late), "base": y_true.mean()}

    # A: 只用其他專案(全域模型)
    model_a = make_model()
    model_a.fit(prepare(others, feature_cols), others[LABEL_COL])
    pred_a = model_a.predict_proba(x_late)[:, 1]
    out["A_auc"], out["A_p10"] = roc_auc_score(y_true, pred_a), precision_at_k(y_true, pred_a)

    # B: 只用這個專案自己的前半段(前半段類別數與樣本數需足夠)
    if early[LABEL_COL].nunique() > 1 and min(np.bincount(early[LABEL_COL].values.astype(int))) >= 3:
        model_b = make_model()
        model_b.fit(prepare(early, feature_cols), early[LABEL_COL])
        pred_b = model_b.predict_proba(x_late)[:, 1]
        out["B_auc"], out["B_p10"] = roc_auc_score(y_true, pred_b), precision_at_k(y_true, pred_b)

        # C: 混合(其他專案 + 自己前半段)
        combined = pd.concat([others, early])
        model_c = make_model()
        model_c.fit(prepare(combined, feature_cols), combined[LABEL_COL])
        pred_c = model_c.predict_proba(x_late)[:, 1]
        out["C_auc"], out["C_p10"] = roc_auc_score(y_true, pred_c), precision_at_k(y_true, pred_c)

    return out


def main(features_csv, out_csv):
    df = pd.read_csv(features_csv).dropna(subset=[LABEL_COL]).copy()
    feature_cols = [c for c in df.columns if c.startswith("f_")]

    rows = [evaluate_project(code, df, feature_cols) for code in sorted(df["code"].unique())]
    rows = [r for r in rows if r is not None]
    result = pd.DataFrame(rows)

    print(f"可評估專案: {len(result)}  (每個都是『用前半段預測後半段』)\n")
    labels = {
        "A": "A. 只用其他專案(單一全域模型)",
        "B": "B. 只用這個專案自己的前半段",
        "C": "C. 混合:其他專案 + 自己前半段",
    }
    print(f"{'做法':34}{'專案數':>7}{'AUC':>8}{'P@10':>8}{'基準':>8}{'提升':>8}")
    for key in ["A", "B", "C"]:
        subset = result.dropna(subset=[f"{key}_auc"])
        if len(subset) == 0:
            continue
        lift = (subset[f"{key}_p10"] / subset["base"].replace(0, np.nan)).mean()
        print(f"{labels[key]:36}{len(subset):>5}{subset[f'{key}_auc'].mean():>8.3f}"
              f"{subset[f'{key}_p10'].mean():>8.3f}{subset['base'].mean():>8.3f}{lift:>7.2f}x")

    comparable = result.dropna(subset=["A_auc", "B_auc", "C_auc"])
    print(f"\n在都能比較的 {len(comparable)} 個專案上直接對打:")
    print(f"  C(混合) 贏過 A(純全域) 的專案: {(comparable['C_auc'] > comparable['A_auc']).sum()}/{len(comparable)}")
    print(f"  C(混合) 贏過 B(純自己) 的專案: {(comparable['C_auc'] > comparable['B_auc']).sum()}/{len(comparable)}")
    print(f"  A(純全域) 贏過 B(純自己) 的專案: {(comparable['A_auc'] > comparable['B_auc']).sum()}/{len(comparable)}")

    result.to_csv(out_csv, index=False)
    print(f"\n已存到: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default="../data/features.csv")
    parser.add_argument("--out", default="../results/deploy_comparison.csv")
    args = parser.parse_args()
    main(args.features, args.out)
