import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
df = pd.read_csv("C:/Users/Duelz/positions_full.csv")
df = df[df["result"].isin(["1-0", "0-1"])]
df = df[(df["white_clock"] <= 60) & (df["black_clock"] <= 60)]
df["white_win"] = (df["result"] == "1-0").astype(int)
df["clock_diff"] = df["white_clock"] - df["black_clock"]
df["mate_value"] = np.sign(df["mate"]) * 1000
df["eval_capped"] = df["eval_cp"].clip(-1000, 1000).fillna(df["mate_value"])
df["rating_diff"] = df["white_elo"] - df["black_elo"]
df["min_clock"] = np.minimum(df["white_clock"],df["black_clock"])
print(df[["white_win", "eval_capped", "clock_diff"]].describe())
pd.set_option("display.float_format", "{:.6f}".format)
game_ids = df["game_id"].unique()
print(len(game_ids))
np.random.seed(42)
train_ids = np.random.choice(game_ids, size= int(0.8 * len(game_ids)), replace=False)
print(len(train_ids))

train = df[df["game_id"].isin(train_ids)]
test = df[~df["game_id"].isin(train_ids)]
print(len(train))
print(len(test))

model = smf.logit("white_win ~ eval_capped + clock_diff", data=train).fit()
print(model.summary())
print(model.params)

seconds_per_pawn = 100 * model.params["eval_capped"] / model.params["clock_diff"] 
print("Seconds per pawn: ", seconds_per_pawn)

close = train[train["eval_capped"].abs() <= 300]
print(len(close))

model_close = smf.logit("white_win ~ eval_capped + clock_diff", data=close).fit()
seconds_per_pawn_close = 100 * model_close.params["eval_capped"] / model_close.params["clock_diff"]
print("Close, no rating:", seconds_per_pawn_close)

# close positions, with rating (all phases)
model_rating = smf.logit("white_win ~ eval_capped + clock_diff + rating_diff", data=close).fit()
spp_rating = 100 * model_rating.params["eval_capped"] / model_rating.params["clock_diff"]
print("Close, with rating:", spp_rating)

# split close positions by phase
close_mid = close[close["phase"] == "middlegame"]
close_end = close[close["phase"] == "endgame"]
print("Middlegame rows:", len(close_mid), "| Endgame rows:", len(close_end))

# middlegame only
model_mid = smf.logit("white_win ~ eval_capped + clock_diff + rating_diff", data=close_mid).fit()
spp_mid = 100 * model_mid.params["eval_capped"] / model_mid.params["clock_diff"]
print("Middlegame seconds per pawn:", spp_mid)

# endgame only
model_end = smf.logit("white_win ~ eval_capped + clock_diff + rating_diff", data=close_end).fit()
spp_end = 100 * model_end.params["eval_capped"] / model_end.params["clock_diff"]
print("Endgame seconds per pawn:", spp_end)

print(model_mid.params)
print(model_end.params)

mid_high = close_mid[(close_mid["white_clock"] > 20) & (close_mid["black_clock"] > 20)]   
mid_low = close_mid[(close_mid["white_clock"] <= 20) | (close_mid["black_clock"] <= 20)]   

print("High-clock rows:", len(mid_high), "| Low-clock rows:", len(mid_low))

# middlegame, high clock
model_high = smf.logit("white_win ~ eval_capped + clock_diff + rating_diff", data=mid_high).fit()
spp_high = 100 * model_high.params["eval_capped"] / model_high.params["clock_diff"]
print("Middlegame, high clock:", spp_high)

# middlegame, low clock
model_low = smf.logit("white_win ~ eval_capped + clock_diff + rating_diff", data=mid_low).fit()
spp_low = 100 * model_low.params["eval_capped"] / model_low.params["clock_diff"]
print("Middlegame, low clock:", spp_low)

model_int = smf.logit(
    "white_win ~ eval_capped + clock_diff + rating_diff + min_clock"
    " + eval_capped:min_clock + clock_diff:min_clock",
    data=close
).fit()
labels = {
    "Intercept": "b0 (baseline)",
    "eval_capped": "b1 (eval)",
    "clock_diff": "b2 (clock)",
    "rating_diff": "b3 (rating)",
    "min_clock": "b4 (min clock)",
    "eval_capped:min_clock": "b5 (eval x min clock)",
    "clock_diff:min_clock": "b6 (clock x min clock)",
}

# test-set check
close_test = test[test["eval_capped"].abs() <= 300]
print("Test close rows:", len(close_test))

close_test["pred"] = model_int.predict(close_test)
print(close_test["pred"].describe())

close_test["bucket"] = pd.cut(close_test["pred"], bins=10)
calib = close_test.groupby("bucket").agg(
    predicted=("pred", "mean"),
    actual=("white_win", "mean"),
    count=("pred", "size"),
)
print(calib)

print(model_int.params.rename(index=labels))

b1 = model_int.params["eval_capped"]
b2 = model_int.params["clock_diff"]
b5 = model_int.params["eval_capped:min_clock"]
b6 = model_int.params["clock_diff:min_clock"]

for m in [5, 10, 20, 30, 40, 45, 50]:
    spp = 100 * (b1 + b5 * m) / (b2 + b6 * m)
    print(f"min_clock {m:>2}s: {spp:.2f} seconds per pawn")

bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 61)]
def spp_by_bin(data):
    results = []
    for low, high in bins:
        subset = data[(data["min_clock"] >= low) & (data["min_clock"] < high)]
        m = smf.logit("white_win ~ eval_capped + clock_diff + rating_diff", data=subset).fit(disp=0)
        results.append(100 * m.params["eval_capped"] / m.params["clock_diff"])
    return results

def bootstrap_ci(data, rounds=100):
    ids = data["game_id"].unique()
    results = []
    for i in range(rounds):
        sample_ids = np.random.choice(ids, size=len(ids), replace=True)
        sample = pd.DataFrame({"game_id": sample_ids}).merge(data, on="game_id")
        results.append(spp_by_bin(sample))
    results = np.array(results)
    lower = np.percentile(results, 2.5, axis=0)
    upper = np.percentile(results, 97.5, axis=0)
    return lower, upper


print(spp_by_bin(close))
bin_labels = []
spp_values = []
bin_counts = []
for low, high in bins:
    subset = close[(close["min_clock"] >= low) & (close["min_clock"] < high)]
    model_bin = smf.logit("white_win ~ eval_capped + clock_diff + rating_diff", data=subset).fit()
    spp_bin = 100 * model_bin.params["eval_capped"] / model_bin.params["clock_diff"]
    print(f"{low}-{high}s: {len(subset)} rows, {spp_bin:.2f} seconds per pawn")
    bin_counts.append(len(subset))
    bin_labels.append(f"{low}-{high}s")    # a label like "0-10s"
    spp_values.append(spp_bin)    # this bin's seconds per pawn


close_ids = close["game_id"].unique()
boot = []

for i in range(100):
    sample_ids = np.random.choice(close_ids, size= len(close_ids), replace= True)
    sample = pd.DataFrame({"game_id": sample_ids}).merge(close, on="game_id")
    boot.append(spp_by_bin(sample))
    if (i + 1) % 10 == 0:
        print("bootstrap round", i + 1)

boot = np.array(boot)
lower = np.percentile(boot, 2.5, axis=0)
upper = np.percentile(boot, 97.5, axis=0)
print("Lower:", lower)
print("Upper:", upper)

for cutoff in [200, 300, 500]:
    c = train[train["eval_capped"].abs() <= cutoff]
    print(cutoff, len(c), [round(x, 2) for x in spp_by_bin(c)])

strong = close[(close["white_elo"] >= 2000) & (close["black_elo"] >= 2000)]    # both players 2000+
print("Strong rows:", len(strong))
strong_spp = spp_by_bin(strong)
print([round(x, 2) for x in strong_spp])
s_low, s_high = bootstrap_ci(strong)
print("Strong lower:", np.round(s_low, 2))
print("Strong upper:", np.round(s_high, 2))

moderate = close[(close["white_elo"] >= 1500) & (close["white_elo"] < 2000) & (close["black_elo"] >= 1500) &(close["black_elo"] < 2000)]    # both players [1500,2000)
print("Moderate rows:", len(moderate))
moderate_spp = spp_by_bin(moderate)
print([round(x, 2) for x in moderate_spp])
m_low, m_high = bootstrap_ci(moderate)
print("Moderate lower:", np.round(m_low, 2))
print("Moderate upper:", np.round(m_high, 2))

weak = close[(close["white_elo"] < 1500) & (close["black_elo"] < 1500)]    # both players <1500
print("Weak rows:", len(weak))
weak_spp = spp_by_bin(weak)
print([round(x, 2) for x in weak_spp])
w_low, w_high = bootstrap_ci(weak)
print("Weak lower:", np.round(w_low, 2))
print("Weak upper:", np.round(w_high, 2))

# ---------- Shared style ----------
bg = "#0f1117"
fg = "#c9ced6"
muted = "#8b95a5"
accent = "#4c8bf5"
gridc = "#2a2f3a"

def style_axes(ax):
    ax.set_facecolor(bg)
    ax.tick_params(colors=fg)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.yaxis.grid(True, color=gridc, linewidth=1)
    ax.set_axisbelow(True)

# ---------- Chart 1: seconds per pawn ----------
bin_labels[-1] = "40s+"
est = np.array(spp_values)
xs = np.arange(len(est))
tick_labels = [f"{lab}\n(n={n:,})" for lab, n in zip(bin_labels, bin_counts)]
colors1 = [accent] * 4 + [muted]

fig, ax = plt.subplots(figsize=(10, 5.8), facecolor=bg)
style_axes(ax)
ax.plot(xs[:4], est[:4], color=accent, alpha=0.35, linewidth=1.5)
for i in range(4):
    ax.errorbar(xs[i], est[i], yerr=[[est[i] - lower[i]], [upper[i] - est[i]]],
                fmt="o", color=colors1[i], ecolor=colors1[i], elinewidth=2.5,
                capsize=0, markersize=7, markeredgecolor=bg, markeredgewidth=1.5)
    ax.text(xs[i] + 0.1, est[i], f"{est[i]:.2f}s", color=fg, fontsize=10, va="center")

ax.axhline(0, color="#4a4f5a", linewidth=1, linestyle="--")
ax.set_xticks(xs[:4])
ax.set_xticklabels(tick_labels[:4], color=fg, fontsize=10)
ax.set_xlabel("Lower of the two clocks", color=fg, fontsize=11)
ax.set_ylabel("Seconds per pawn", color=fg, fontsize=11)
fig.suptitle("Below 10 seconds, a pawn is worth almost nothing on the clock",
             x=0.06, ha="left", color="white", fontsize=15, fontweight="bold")
ax.set_title("1+0 bullet · Lichess April 2017 · positions within ±3 pawns · dots = estimate, lines = 95% bootstrap CI\n"
             f"Not shown: 40s+ ≈ {est[4]:.0f}s per pawn (95% CI {lower[4]:.0f}–{upper[4]:.0f}s), too imprecise to plot on this scale",
             loc="left", color=muted, fontsize=9)
fig.tight_layout()
fig.savefig("seconds_per_pawn.png", dpi=200, facecolor=bg)

# ---------- Chart 2: calibration ----------
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

# ---------- Chart 3: by rating group ----------
groups = [
    ("Under 1500", weak_spp, w_low, w_high, "#8b95a5"),
    ("1500–1999", moderate_spp, m_low, m_high, "#4c8bf5"),
    ("2000+", strong_spp, s_low, s_high, "#f5a524"),
]
x = np.arange(4)
xlabels = ["0–10s", "10–20s", "20–30s", "30–40s"]
offsets = [-0.2, 0, 0.2]
bg = "#0f1117"

fig3, ax3 = plt.subplots(figsize=(9.5, 5.8), facecolor=bg)
ax3.set_facecolor(bg)

for (name, est, lo, hi, color), off in zip(groups, offsets):
    est = np.array(est[:4]); lo = np.array(lo[:4]); hi = np.array(hi[:4])
    ax3.plot(x + off, est, color=color, alpha=0.35, linewidth=1.5)
    ax3.errorbar(x + off, est, yerr=[est - lo, hi - est], fmt="o", color=color,
                 ecolor=color, elinewidth=2.5, capsize=0, markersize=6,
                 markeredgecolor=bg, markeredgewidth=1.5, label=name)

ax3.axhline(0, color="#4a4f5a", linewidth=1, linestyle="--")
ax3.set_xticks(x)
ax3.set_xticklabels(xlabels, color="#c9ced6", fontsize=11)
ax3.tick_params(axis="y", colors="#c9ced6")
ax3.set_xlabel("Lower of the two clocks", color="#c9ced6", fontsize=11)
ax3.set_ylabel("Seconds per pawn", color="#c9ced6", fontsize=11)
for side in ["top", "right", "left", "bottom"]:
    ax3.spines[side].set_visible(False)
ax3.yaxis.grid(True, color="#2a2f3a", linewidth=1)
ax3.set_axisbelow(True)

fig3.suptitle("Stronger players get more seconds' worth out of a pawn",
              x=0.06, ha="left", color="white", fontsize=15, fontweight="bold")
ax3.set_title("1+0 bullet · both players in the same rating band · dots = estimate, lines = 95% bootstrap CI\n"
              "Gray/blue intervals are narrow · groups overlap at 10–20s",
              loc="left", color="#8b95a5", fontsize=9)
leg = ax3.legend(title="Both players rated", frameon=False, labelcolor="#c9ced6", loc="upper left")
leg.get_title().set_color("#c9ced6")

fig3.tight_layout()
fig3.savefig("by_rating.png", dpi=200, facecolor=bg)

plt.show()

