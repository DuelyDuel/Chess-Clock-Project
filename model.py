"""
Chess Clock Project: how many seconds is a pawn worth in 1+0 bullet?
Input:  positions_full.csv (built by extraction.py)
Output: seconds_per_pawn.png, calibration.png, by_rating.png
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# ---------- Settings ----------
DATA_PATH = "C:/Users/Duelz/positions_full.csv"
EVAL_CUTOFF = 300      # "close" positions: within ±3 pawns
BOOT_ROUNDS = 500      # bootstrap rounds (raise to 500 for final results)
BINS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 61)]
LABELS = ["0-10s", "10-20s", "20-30s", "30-40s", "40s+"]
FORMULA = "white_win ~ eval_capped + clock_diff + rating_diff"
np.random.seed(42)

# ---------- 1. Load and prepare data ----------
df = pd.read_csv(DATA_PATH)
df = df[df["result"].isin(["1-0", "0-1"])]                      # drop draws / unknown results
df = df[(df["white_clock"] <= 60) & (df["black_clock"] <= 60)]   # drop games with added time
df["white_win"] = (df["result"] == "1-0").astype(int)
df["clock_diff"] = df["white_clock"] - df["black_clock"]
df["mate_value"] = np.sign(df["mate"]) * 1000
df["eval_capped"] = df["eval_cp"].clip(-1000, 1000).fillna(df["mate_value"])
df["rating_diff"] = df["white_elo"] - df["black_elo"]
df["min_clock"] = np.minimum(df["white_clock"], df["black_clock"])

# ---------- 2. Train/test split (by game) ----------
game_ids = df["game_id"].unique()
train_ids = np.random.choice(game_ids, size=int(0.8 * len(game_ids)), replace=False)
train = df[df["game_id"].isin(train_ids)]
test = df[~df["game_id"].isin(train_ids)]
close = train[train["eval_capped"].abs() <= EVAL_CUTOFF]
close_test = test[test["eval_capped"].abs() <= EVAL_CUTOFF].copy()

# ---------- 3. Core functions ----------
def spp_by_bin(data):
    """Seconds per pawn in each clock bin (bins use the lower of the two clocks)."""
    results = []
    for low, high in BINS:
        subset = data[(data["min_clock"] >= low) & (data["min_clock"] < high)]
        m = smf.logit(FORMULA, data=subset).fit(disp=0)
        results.append(100 * m.params["eval_capped"] / m.params["clock_diff"])
    return np.array(results)

def bootstrap_ci(data, rounds=BOOT_ROUNDS):
    """95% confidence interval for spp_by_bin, resampling whole games with replacement."""
    ids = data["game_id"].unique()
    results = []
    for _ in range(rounds):
        sample_ids = np.random.choice(ids, size=len(ids), replace=True)
        sample = pd.DataFrame({"game_id": sample_ids}).merge(data, on="game_id")
        results.append(spp_by_bin(sample))
    results = np.array(results)
    return np.percentile(results, 2.5, axis=0), np.percentile(results, 97.5, axis=0)

def bin_counts(data):
    """Number of positions in each clock bin."""
    return [((data["min_clock"] >= lo) & (data["min_clock"] < hi)).sum() for lo, hi in BINS]

# ---------- 4. Main result: seconds per pawn by clock level ----------
spp_all = spp_by_bin(close)
low_all, high_all = bootstrap_ci(close)
counts_all = bin_counts(close)

# ---------- 5. Sensitivity check: eval cutoff ----------
sensitivity = {}
for cutoff in [200, 300, 500]:
    c = train[train["eval_capped"].abs() <= cutoff]
    sensitivity[cutoff] = spp_by_bin(c)

# ---------- 6. By rating group (both players in the same band) ----------
rating_groups = {
    "Under 1500": close[(close["white_elo"] < 1500) & (close["black_elo"] < 1500)],
    "1500–1999": close[(close["white_elo"] >= 1500) & (close["white_elo"] < 2000)
                       & (close["black_elo"] >= 1500) & (close["black_elo"] < 2000)],
    "2000+": close[(close["white_elo"] >= 2000) & (close["black_elo"] >= 2000)],
}
rating_results = {}
for name, data in rating_groups.items():
    est = spp_by_bin(data)
    lo, hi = bootstrap_ci(data)
    rating_results[name] = (est, lo, hi, len(data))

# ---------- 7. Test-set calibration ----------
model_int = smf.logit(
    "white_win ~ eval_capped + clock_diff + rating_diff + min_clock"
    " + eval_capped:min_clock + clock_diff:min_clock",
    data=close,
).fit(disp=0)
close_test["pred"] = model_int.predict(close_test)
close_test["bucket"] = pd.cut(close_test["pred"], bins=10)
calib = close_test.groupby("bucket", observed=False).agg(
    predicted=("pred", "mean"),
    actual=("white_win", "mean"),
    count=("pred", "size"),
)

# ---------- 8. Summary ----------
print(f"Positions: {len(df):,} | train (close): {len(close):,} | test (close): {len(close_test):,}\n")

print("Seconds per pawn by lower clock (95% bootstrap CI):")
for lab, e, lo, hi, n in zip(LABELS, spp_all, low_all, high_all, counts_all):
    print(f"  {lab:>7}: {e:6.2f}  ({lo:6.2f} to {hi:6.2f})   n={n:,}")

print("\nSensitivity to eval cutoff:")
for cutoff, vals in sensitivity.items():
    print(f"  ±{cutoff}: " + "  ".join(f"{v:6.2f}" for v in vals))

print("\nBy rating group, first four bins (95% bootstrap CI):")
for name, (e, lo, hi, n) in rating_results.items():
    cells = "  ".join(f"{e[i]:.2f} ({lo[i]:.2f} to {hi[i]:.2f})" for i in range(4))
    print(f"  {name:>10} (n={n:,}): {cells}")

print("\nCalibration on held-out games:")
print(calib.round(3))

# ---------- 9. Charts ----------
bg, fg, muted, accent, gridc = "#0f1117", "#c9ced6", "#8b95a5", "#4c8bf5", "#2a2f3a"

def style_axes(ax):
    ax.set_facecolor(bg)
    ax.tick_params(colors=fg)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.yaxis.grid(True, color=gridc, linewidth=1)
    ax.set_axisbelow(True)

# Chart 1: seconds per pawn (first four bins; 40s+ reported in subtitle)
xs = np.arange(4)
est4 = spp_all[:4]
tick_labels = [f"{lab}\n(n={n:,})" for lab, n in zip(LABELS[:4], counts_all[:4])]
fig, ax = plt.subplots(figsize=(10, 5.8), facecolor=bg)
style_axes(ax)
ax.plot(xs, est4, color=accent, alpha=0.35, linewidth=1.5)
ax.errorbar(xs, est4, yerr=[est4 - low_all[:4], high_all[:4] - est4], fmt="o",
            color=accent, ecolor=accent, elinewidth=2.5, capsize=0, markersize=7,
            markeredgecolor=bg, markeredgewidth=1.5)
for i in range(4):
    ax.text(xs[i] + 0.1, est4[i], f"{est4[i]:.2f}s", color=fg, fontsize=10, va="center")
ax.axhline(0, color="#4a4f5a", linewidth=1, linestyle="--")
ax.set_xticks(xs)
ax.set_xticklabels(tick_labels, color=fg, fontsize=10)
ax.set_xlabel("Lower of the two clocks", color=fg, fontsize=11)
ax.set_ylabel("Seconds per pawn", color=fg, fontsize=11)
fig.suptitle("Below 10 seconds, a pawn is worth almost nothing on the clock",
             x=0.06, ha="left", color="white", fontsize=15, fontweight="bold")
ax.set_title("1+0 bullet · Lichess April 2017 · positions within ±3 pawns · dots = estimate, lines = 95% bootstrap CI\n"
             f"Not shown: 40s+ ≈ {spp_all[4]:.0f}s per pawn (95% CI {low_all[4]:.0f}–{high_all[4]:.0f}s), too imprecise to plot on this scale",
             loc="left", color=muted, fontsize=9)
fig.tight_layout()
fig.savefig("seconds_per_pawn.png", dpi=200, facecolor=bg)

# Chart 2: calibration
fig2, ax2 = plt.subplots(figsize=(6.5, 6.5), facecolor=bg)
style_axes(ax2)
ax2.xaxis.grid(True, color=gridc, linewidth=1)
sizes = calib["count"] / calib["count"].max() * 200
ax2.plot([0, 1], [0, 1], linestyle="--", color="#4a4f5a", label="Perfect calibration")
ax2.plot(calib["predicted"], calib["actual"], color=accent, alpha=0.5, linewidth=1.5)
ax2.scatter(calib["predicted"], calib["actual"], s=sizes, color=accent,
            edgecolor=bg, linewidth=1.5, zorder=3, label="Model on held-out games")
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_aspect("equal")
ax2.set_xlabel("Predicted chance White wins", color=fg, fontsize=11)
ax2.set_ylabel("Actual White win rate", color=fg, fontsize=11)
fig2.suptitle("The model's probabilities hold up on unseen games",
              x=0.06, ha="left", color="white", fontsize=14, fontweight="bold")
ax2.set_title(f"{len(close_test):,} test positions in 10 buckets · dot size = positions per bucket",
              loc="left", color=muted, fontsize=9)
ax2.legend(frameon=False, labelcolor=fg, loc="upper left")
fig2.tight_layout()
fig2.savefig("calibration.png", dpi=200, facecolor=bg)

# Chart 3: by rating group
group_colors = {"Under 1500": "#8b95a5", "1500–1999": "#4c8bf5", "2000+": "#f5a524"}
offsets = [-0.2, 0, 0.2]
fig3, ax3 = plt.subplots(figsize=(9.5, 5.8), facecolor=bg)
style_axes(ax3)
for (name, (e, lo, hi, n)), off in zip(rating_results.items(), offsets):
    color = group_colors[name]
    e4, lo4, hi4 = e[:4], lo[:4], hi[:4]
    ax3.plot(xs + off, e4, color=color, alpha=0.35, linewidth=1.5)
    ax3.errorbar(xs + off, e4, yerr=[e4 - lo4, hi4 - e4], fmt="o", color=color,
                 ecolor=color, elinewidth=2.5, capsize=0, markersize=6,
                 markeredgecolor=bg, markeredgewidth=1.5, label=name)
ax3.axhline(0, color="#4a4f5a", linewidth=1, linestyle="--")
ax3.set_xticks(xs)
ax3.set_xticklabels(LABELS[:4], color=fg, fontsize=11)
ax3.set_xlabel("Lower of the two clocks", color=fg, fontsize=11)
ax3.set_ylabel("Seconds per pawn", color=fg, fontsize=11)
fig3.suptitle("Stronger players get more seconds' worth out of a pawn",
              x=0.06, ha="left", color="white", fontsize=15, fontweight="bold")
ax3.set_title("1+0 bullet · both players in the same rating band · dots = estimate, lines = 95% bootstrap CI\n"
              "Gray/blue intervals are narrow · groups overlap at 10–20s",
              loc="left", color=muted, fontsize=9)
leg = ax3.legend(title="Both players rated", frameon=False, labelcolor=fg, loc="upper left")
leg.get_title().set_color(fg)
fig3.tight_layout()
fig3.savefig("by_rating.png", dpi=200, facecolor=bg)

plt.show()