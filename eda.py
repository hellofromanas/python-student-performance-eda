"""Exploratory data analysis and regression on a REAL dataset: UCI "Student Performance" (maths course).

Data: P. Cortez and A. Silva (2008), 395 students from two Portuguese secondary schools, 33 attributes.
Source: https://archive.ics.uci.edu/dataset/320/student+performance  (licence: CC BY 4.0)

Run:  python eda.py        ->  writes images/*.png and results.md
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
IMG = ROOT / "images"
TARGET = "G3"  # final grade, 0-20

NUMERIC = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime", "goout",
           "Dalc", "Walc", "health", "absences"]
BINARY = ["school", "sex", "address", "famsize", "Pstatus", "schoolsup", "famsup", "paid", "activities",
          "nursery", "higher", "internet", "romantic"]
PRIOR = ["G1", "G2"]


def load() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(ROOT / "data" / "student-mat.csv", sep=";")
    info = {
        "rows": len(df),
        "cols": df.shape[1],
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "zero_final": int((df[TARGET] == 0).sum()),
    }
    # Data validation: grades must be on the documented 0-20 scale.
    assert df[["G1", "G2", "G3"]].apply(lambda s: s.between(0, 20).all()).all(), "grade outside 0-20"
    return df, info


def design_matrix(df: pd.DataFrame, use_prior: bool) -> pd.DataFrame:
    """Numeric columns + one-hot binary columns (+ G1/G2 if requested)."""
    cols = NUMERIC + (PRIOR if use_prior else [])
    X = df[cols].astype(float).copy()
    for b in BINARY:
        first = sorted(df[b].unique())[0]
        X[f"{b}={sorted(df[b].unique())[1]}"] = (df[b] != first).astype(float)
    return X


def evaluate(df: pd.DataFrame, use_prior: bool, n_splits: int = 200, seed: int = 42) -> dict:
    """Ordinary least squares (NumPy lstsq) on standardised features; mean test R^2 / RMSE over many random 80/20 splits."""
    X, y = design_matrix(df, use_prior), df[TARGET].to_numpy(float)
    rng = np.random.default_rng(seed)
    r2s, rmses, base_rmses = [], [], []
    for _ in range(n_splits):
        idx = rng.permutation(len(df))
        cut = int(0.8 * len(df))
        tr, te = idx[:cut], idx[cut:]
        mu, sd = X.iloc[tr].mean(), X.iloc[tr].std().replace(0, 1)
        Xtr = np.c_[np.ones(len(tr)), ((X.iloc[tr] - mu) / sd).to_numpy()]
        Xte = np.c_[np.ones(len(te)), ((X.iloc[te] - mu) / sd).to_numpy()]
        beta, *_ = np.linalg.lstsq(Xtr, y[tr], rcond=None)
        res = y[te] - Xte @ beta
        rmses.append(np.sqrt((res**2).mean()))
        r2s.append(1 - (res**2).sum() / ((y[te] - y[te].mean()) ** 2).sum())
        base_rmses.append(np.sqrt(((y[te] - y[tr].mean()) ** 2).mean()))  # baseline: always predict the training mean
    # Coefficients from a fit on all data, for interpretation.
    mu, sd = X.mean(), X.std().replace(0, 1)
    Xa = np.c_[np.ones(len(df)), ((X - mu) / sd).to_numpy()]
    beta, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    coef = pd.Series(beta[1:], index=X.columns, name="effect_on_G3").sort_values(key=abs, ascending=False)
    return {"r2": float(np.mean(r2s)), "rmse": float(np.mean(rmses)), "baseline_rmse": float(np.mean(base_rmses)), "coef": coef}


def make_plots(df: pd.DataFrame) -> dict:
    IMG.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
    out = {}

    # 1) distribution of the final grade
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df[TARGET], bins=np.arange(-0.5, 21.5, 1), color="#2563eb", edgecolor="white")
    ax.axvline(df[TARGET].mean(), color="#dc2626", ls="--", label=f"mean = {df[TARGET].mean():.1f}")
    ax.set(title="Final maths grade (G3)", xlabel="Grade (0-20)", ylabel="Students")
    ax.legend()
    fig.tight_layout(); fig.savefig(IMG / "grade_distribution.png"); plt.close(fig)

    # 2) average grade by weekly study time (with group sizes)
    g = df.groupby("studytime")[TARGET].agg(["mean", "count"])
    labels = {1: "<2 h", 2: "2-5 h", 3: "5-10 h", 4: ">10 h"}
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([labels[i] for i in g.index], g["mean"], color="#2563eb")
    for b, n in zip(bars, g["count"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15, f"n={n}", ha="center", fontsize=8)
    ax.set(title="Average final grade by weekly study time", xlabel="Weekly study time", ylabel="Mean G3")
    ax.set_ylim(0, max(g["mean"]) + 2)
    fig.tight_layout(); fig.savefig(IMG / "studytime_vs_grade.png"); plt.close(fig)
    out["studytime"] = g

    # 3) first-period grade vs final grade
    slope, intercept = np.polyfit(df["G1"], df[TARGET], 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["G1"], df[TARGET], s=14, alpha=0.5, color="#2563eb")
    xs = np.array([df["G1"].min(), df["G1"].max()])
    ax.plot(xs, intercept + slope * xs, color="#dc2626", lw=2, label=f"fit: slope {slope:.2f}")
    ax.set(title="First-period grade vs final grade", xlabel="G1", ylabel="G3")
    ax.legend()
    fig.tight_layout(); fig.savefig(IMG / "g1_vs_g3.png"); plt.close(fig)

    # 4) absences and failures
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].scatter(df["absences"], df[TARGET], s=14, alpha=0.5, color="#2563eb")
    axes[0].set(title="Absences vs final grade", xlabel="Absences", ylabel="G3")
    data = [df.loc[df.failures == k, TARGET] for k in sorted(df.failures.unique())]
    axes[1].boxplot(data, tick_labels=[f"{k}\n(n={len(d)})" for k, d in zip(sorted(df.failures.unique()), data)],
                    patch_artist=True, boxprops=dict(facecolor="#bfdbfe"), medianprops=dict(color="#dc2626"))
    axes[1].set(title="Final grade by past class failures", xlabel="Past failures", ylabel="G3")
    fig.tight_layout(); fig.savefig(IMG / "absences_failures.png"); plt.close(fig)

    # 5) correlation heatmap
    cols = ["studytime", "failures", "absences", "Medu", "Fedu", "goout", "Walc", "health", "G1", "G2", TARGET]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right"); ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8); ax.set_title("Correlation matrix")
    fig.tight_layout(); fig.savefig(IMG / "correlation_heatmap.png"); plt.close(fig)
    out["corr"] = corr[TARGET].drop(TARGET).sort_values(ascending=False)
    return out


def main() -> None:
    df, info = load()
    plots = make_plots(df)
    a = evaluate(df, use_prior=False)
    b = evaluate(df, use_prior=True)
    nz = df[df[TARGET] > 0]
    zero_abs = int(((df[TARGET] == 0) & (df['absences'] == 0)).sum())
    abs_nz = float(nz['absences'].corr(nz[TARGET]))
    c = evaluate(nz.reset_index(drop=True), use_prior=False)

    md = [
        "# Results (auto-generated by eda.py)",
        "",
        "Data: UCI Student Performance (maths course), Cortez & Silva (2008), CC BY 4.0.",
        "",
        "## Data checks",
        f"- {info['rows']} students, {info['cols']} columns; missing values: {info['missing']}; duplicate rows: {info['duplicates']}.",
        f"- Grades validated to lie on the documented 0-20 scale.",
        f"- {info['zero_final']} students have a final grade of 0 ({info['zero_final'] / info['rows']:.0%}); "
        "these are kept in the main analysis and examined separately below.",
        "",
        "## Final grade (G3)",
        f"- Mean {df[TARGET].mean():.2f}, median {df[TARGET].median():.0f}, std {df[TARGET].std():.2f}.",
        "",
        "## Mean final grade by weekly study time",
        plots["studytime"].round(2).rename(columns={"mean": "mean_G3", "count": "students"}).to_markdown(),
        "",
        "## Correlation of G3 with other variables (Pearson)",
        plots["corr"].round(3).to_markdown(),
        "",
        "## Linear regression - mean over 200 random 80/20 splits",
        "| Model | Test R^2 | Test RMSE | Baseline RMSE (predict the mean) |",
        "|---|---|---|---|",
        f"| A. Background and behaviour only (no earlier grades) | {a['r2']:.3f} | {a['rmse']:.2f} | {a['baseline_rmse']:.2f} |",
        f"| B. A + first and second period grades (G1, G2) | {b['r2']:.3f} | {b['rmse']:.2f} | {b['baseline_rmse']:.2f} |",
        f"| C. Model A excluding the {info['zero_final']} students with G3 = 0 | {c['r2']:.3f} | {c['rmse']:.2f} | {c['baseline_rmse']:.2f} |",
        "",
        "### Largest effects in model A (standardised: change in G3 for +1 std dev)",
        a["coef"].head(8).round(2).to_markdown(),
        "",
        "### Largest effects in model B",
        b["coef"].head(5).round(2).to_markdown(),
        "",
        "## Interpretation (association, not causation)",
        f"- **Earlier grades dominate.** Using only background and behaviour, the model explains little (R^2 = {a['r2']:.2f}); adding G1 and G2 raises it to {b['r2']:.2f} "
        f"(RMSE {a['rmse']:.2f} -> {b['rmse']:.2f} grade points, against {a['baseline_rmse']:.2f} for guessing the mean).",
        f"- **Past failures** show the clearest negative association with the final grade (r = {plots['corr']['failures']:.2f}).",
        f"- **Study time** has a modest positive association (r = {plots['corr']['studytime']:.2f}): mean G3 is {plots['studytime']['mean'].iloc[0]:.1f} for students studying under 2 h a week "
        f"and {plots['studytime']['mean'].iloc[-1]:.1f} for those studying over 10 h (only {int(plots['studytime']['count'].iloc[-1])} students in the top group).",
        f"- **Absences look unrelated at first** (r = {plots['corr']['absences']:.2f}), but every one of the {info['zero_final']} students with G3 = 0 also has 0 recorded absences "
        f"({zero_abs} of {info['zero_final']}). A plausible explanation (my inference, not documented in the dataset) is that these are students who left or did not sit the final assessment, so absences were never logged. "
        f"Excluding them, absences are negatively associated with G3 (r = {abs_nz:.2f}). This is an example of a data-quality issue that can hide a real pattern.",
        f"- The {info['zero_final']} students with G3 = 0 are not the only reason the background-only model fits poorly: without them R^2 is still only {c['r2']:.2f}.",
        "- Limits: one maths course, 395 students from two schools, observational data - these results do not show that changing one factor would change grades.",
        "",
    ]
    (ROOT / "results.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
