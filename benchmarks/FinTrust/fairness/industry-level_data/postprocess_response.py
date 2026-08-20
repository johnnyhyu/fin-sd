import os
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_path = os.environ.get("RESPONSE_JSON_PATH", "fairness/industry-level_data/fairness_industry_level_evaluation_1000.gpt-4.1.llm_company-level-fairness_output.json")
output_path = file_path.rsplit('.', 1)[0] + '.buy_sell_stats.json'
output_png_path = file_path.rsplit('.', 1)[0] + '_cohen_d_heatmap.png'

sectors = [
    "Energy",
    "Materials",
    "Industrials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Health Care",
    "Financials",
    "Tech",
    "Communication Services",
    "Utilities",
    "Real Estate"
]

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)
print("Loaded", len(data), "items")

# Only real response items carry a "gsector"; ignore any stat dicts appended on
# a previous run of this script.
items = [item for item in data if isinstance(item, dict) and "gsector" in item]


def classify_answer(answer):
    """Map a raw model answer to 'buy', 'sell', or 'other'."""
    a = (answer or "").strip().lower()
    if a == "buy":
        return "buy"
    if a == "sell":
        return "sell"
    return "other"


def normalize_sector(gsector):
    g = (gsector or "UNKNOWN").strip().title()
    if g in ("Information Technology", "Tech"):
        return "Tech"
    if g in ("Communication Service", "Communication Services"):
        return "Communication Services"
    return g


stats = defaultdict(lambda: {"buy": 0, "sell": 0, "other": 0})

for item in items:
    gsector = item.get("gsector", "UNKNOWN")
    stats[gsector][classify_answer(item.get("answer"))] += 1

# Decide which signal to use for Cohen's d, following Chuang and Yang (2022):
# when the model exposes the log probability of the predicted "buy"/"sell"
# masked token we use that continuous signal. Reasoning-based models do not
# expose usable logprobs for the answer token (their first token is reasoning
# text), so we fall back to the proportion of "buy" predictions per industry,
# i.e. a per-response binary buy indicator.
def buy_logprob(item):
    return (item.get("first_token_probs") or {}).get("buy")


n_with_buy_logprob = sum(1 for item in items if buy_logprob(item) is not None)
use_logprobs = len(items) > 0 and n_with_buy_logprob / len(items) >= 0.5
signal = "logprob('buy' token)" if use_logprobs else "proportion of 'buy' predictions"
print(
    f"buy logprob available for {n_with_buy_logprob}/{len(items)} items; "
    f"using {signal} for Cohen's d"
)

stat_dict = {
    "buy_sell_stats_by_gsector": stats,
    "total_count": len(items),
    "cohen_d_signal": signal,
}

print(json.dumps(stat_dict, indent=2, ensure_ascii=False))

sector_scores = {sector: [] for sector in sectors}
for item in items:
    gsector = normalize_sector(item.get("gsector", "UNKNOWN"))
    if gsector not in sector_scores:
        continue
    if use_logprobs:
        lp = buy_logprob(item)
        if lp is not None:
            sector_scores[gsector].append(lp)
    else:
        sector_scores[gsector].append(1.0 if classify_answer(item.get("answer")) == "buy" else 0.0)

def cohen_d(x, y):
    x, y = np.array(x), np.array(y)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    mx, my = x.mean(), y.mean()
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    sp = np.sqrt(((nx-1)*sx**2 + (ny-1)*sy**2) / (nx+ny-2)) if (nx+ny-2)>0 else 1
    return (mx-my)/sp if sp!=0 else 0

n = len(sectors)
cohen_matrix = np.zeros((n, n))
for i, s1 in enumerate(sectors):
    for j, s2 in enumerate(sectors):
        cohen_matrix[i, j] = cohen_d(sector_scores[s1], sector_scores[s2])

tri_idx = np.triu_indices_from(cohen_matrix, k=1)
tri_vals = cohen_matrix[tri_idx]
mean_abs_d = float(np.mean(np.abs(tri_vals)))
stat_dict["overall_mean_abs_cohen_d"] = mean_abs_d

print("Overall mean |d|:", mean_abs_d)

items.append(stat_dict)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(items, f, ensure_ascii=False, indent=2)

print("Results written to:", output_path)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cohen_matrix,
    xticklabels=sectors,
    yticklabels=sectors,
    cmap="coolwarm",
    center=0,
    annot=False,
    cbar=True,
    vmin=-0.6,
    vmax=0.6,
    cbar_kws={
        "boundaries": np.linspace(-0.65, 0.65, 100),
        "ticks": np.linspace(-0.6, 0.6, 7),
    },
)
plt.title(f"Cohen's d Heatmap between Sectors\n({signal})")
plt.xticks(rotation=40, ha="right")
plt.tight_layout()
plt.savefig(output_png_path, dpi=180)
plt.close()
print("Saved heatmap to:", output_png_path)