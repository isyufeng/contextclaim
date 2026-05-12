import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 16

# ============================================================
# DATA
# ============================================================
data = [
    ("CT22", "Fine-tuning", "BERT", 0.6829, 0.6844, 0.8671, 0.7645, 0.7020, 0.7081, 0.8483, 0.7718, 0.6948, 0.6987,
     0.8550, 0.7689),
    ("CT22", "Fine-tuning", "RoBERTa", 0.7084, 0.6999, 0.8980, 0.7853, 0.7195, 0.7208, 0.8671, 0.7864, 0.7363, 0.7379,
     0.8644, 0.7954),
    ("CT22", "Fine-tuning", "Llama3", 0.6295, 0.6414, 0.8523, 0.7320, 0.6295, 0.6228, 0.9530, 0.7533, 0.6321, 0.6225,
     0.9664, 0.7572),
    ("CT22", "Fine-tuning", "Mistral", 0.6056, 0.6053, 0.9642, 0.7437, 0.7264, 0.7428, 0.8255, 0.7819, 0.6866, 0.7158,
     0.7830, 0.7478),
    ("CT22", "Zero-shot", "Llama3", 0.6016, 0.5984, 1.0000, 0.7487, 0.6215, 0.6107, 1.0000, 0.7583, 0.6255, 0.6132,
     1.0000, 0.7602),
    ("CT22", "Zero-shot", "Mistral", 0.6773, 0.6518, 0.9799, 0.7828, 0.6534, 0.6348, 0.9799, 0.7704, 0.6534, 0.6348,
     0.9799, 0.7704),
    ("CT22", "Zero-shot", "GPT-4o", 0.7251, 0.7062, 0.9195, 0.7988, 0.7410, 0.7188, 0.9262, 0.8094, 0.7450, 0.7225,
     0.9262, 0.8118),
    ("CT22", "Few-shot", "Llama3", 0.7131, 0.6860, 0.9530, 0.7978, 0.6853, 0.6591, 0.9732, 0.7859, 0.6733, 0.6489,
     0.9799, 0.7807),
    ("CT22", "Few-shot", "Mistral", 0.6375, 0.6218, 0.9933, 0.7649, 0.7052, 0.6943, 0.8993, 0.7836, 0.6932, 0.6748,
     0.9329, 0.7831),
    ("CT22", "Few-shot", "GPT-4o", 0.7928, 0.8489, 0.7919, 0.8194, 0.7928, 0.8540, 0.7852, 0.8182, 0.7769, 0.8298,
     0.7852, 0.8069),
    ("PoliClaim", "Fine-tuning", "BERT", 0.8277, 0.8357, 0.9090, 0.8707, 0.8402, 0.8495, 0.9117, 0.8794, 0.8409, 0.8477,
     0.9175, 0.8806),
    ("PoliClaim", "Fine-tuning", "RoBERTa", 0.8578, 0.8519, 0.9413, 0.8942, 0.8645, 0.8715, 0.9248, 0.8971, 0.8593,
     0.8683, 0.9198, 0.8930),
    ("PoliClaim", "Fine-tuning", "Llama3", 0.6912, 0.7246, 0.8330, 0.7750, 0.7251, 0.7445, 0.8669, 0.8011, 0.7202,
     0.7415, 0.8624, 0.7974),
    ("PoliClaim", "Fine-tuning", "Mistral", 0.8440, 0.8612, 0.9008, 0.8805, 0.8092, 0.8338, 0.8759, 0.8543, 0.8080,
     0.8409, 0.8624, 0.8515),
    ("PoliClaim", "Zero-shot", "Llama3", 0.6924, 0.6885, 0.9463, 0.7971, 0.7096, 0.7094, 0.9232, 0.8023, 0.7181, 0.7137,
     0.9328, 0.8087),
    ("PoliClaim", "Zero-shot", "Mistral", 0.7966, 0.8318, 0.8541, 0.8428, 0.7843, 0.7939, 0.8944, 0.8412, 0.7904, 0.8007,
     0.8944, 0.8450),
    ("PoliClaim", "Zero-shot", "GPT-4o", 0.7623, 0.9739, 0.6449, 0.7760, 0.7782, 0.9722, 0.6718, 0.7946, 0.7782, 0.9620,
     0.6795, 0.7964),
    ("PoliClaim", "Few-shot", "Llama3", 0.7255, 0.7384, 0.8829, 0.8042, 0.7426, 0.7441, 0.9098, 0.8187, 0.7451, 0.7800,
     0.8369, 0.8074),
    ("PoliClaim", "Few-shot", "Mistral", 0.7108, 0.6939, 0.9789, 0.8121, 0.7635, 0.7742, 0.8887, 0.8275, 0.7439, 0.7847,
     0.8253, 0.8045),
    ("PoliClaim", "Few-shot", "GPT-4o", 0.6728, 1.0000, 0.4875, 0.6555, 0.6520, 1.0000, 0.4549, 0.6253, 0.6507, 0.9958,
     0.4549, 0.6245),
]

cols = ["Dataset", "Setting", "Model", "Acc_base", "Prec_base", "Rec_base", "F1_base",
        "Acc_G4o", "Prec_G4o", "Rec_G4o", "F1_G4o", "Acc_M", "Prec_M", "Rec_M", "F1_M"]
df = pd.DataFrame(data, columns=cols)

df["dF1_G4o"] = (df["F1_G4o"] - df["F1_base"]) * 100
df["dF1_M"] = (df["F1_M"] - df["F1_base"]) * 100
df["dAcc_G4o"] = (df["Acc_G4o"] - df["Acc_base"]) * 100
df["dAcc_M"] = (df["Acc_M"] - df["Acc_base"]) * 100
df["dPrec_G4o"] = (df["Prec_G4o"] - df["Prec_base"]) * 100
df["dPrec_M"] = (df["Prec_M"] - df["Prec_base"]) * 100
df["dRec_G4o"] = (df["Rec_G4o"] - df["Rec_base"]) * 100
df["dRec_M"] = (df["Rec_M"] - df["Rec_base"]) * 100


# ============================================================
# FIGURE 1: Heatmap
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ============================================================
# FIGURE 1: Heatmap
# ============================================================
def make_heatmap(ax, method_col, title, vmin=-12, vmax=6):
    models = ["BERT", "RoBERTa", "Llama3", "Mistral", "GPT-4o"]
    settings_order = [
        ("CT22", "Few-shot"),
        ("CT22", "Fine-tuning"),
        ("CT22", "Zero-shot"),
        ("PoliClaim", "Few-shot"),
        ("PoliClaim", "Fine-tuning"),
        ("PoliClaim", "Zero-shot"),
    ]

    labels = ["CT22-FS", "CT22-FT", "CT22-ZS", "PoliClaim-FS", "PoliClaim-FT", "PoliClaim-ZS"]

    matrix = np.full((len(models), len(settings_order)), np.nan)
    for i, m in enumerate(models):
        for j, (ds, se) in enumerate(settings_order):
            row = df[(df["Dataset"] == ds) & (df["Setting"] == se) & (df["Model"] == m)]
            if len(row) == 1:
                matrix[i, j] = row[method_col].values[0]

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap = plt.cm.RdYlGn

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='equal')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=14)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=18)

    for i in range(len(models)):
        for j in range(len(settings_order)):
            v = matrix[i, j]
            if not np.isnan(v):
                color = 'white' if abs(v) > 3 else 'black'
                ax.text(j, i, f"{v:.2f}", ha='center', va='center', fontsize=18, color=color,
                        fontweight='bold')

    ax.set_xlabel("Dataset × Learning Setting", fontsize=20, fontweight='bold')
    ax.set_ylabel("Base Model", fontsize=20, fontweight='bold')
    ax.set_title(title, fontsize=22, fontweight='bold')
    return im


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

im1 = make_heatmap(ax1, "dF1_G4o", "CC-G4o Performance Improvement")
im2 = make_heatmap(ax2, "dF1_M", "CC-M Performance Improvement")

cb1 = fig.colorbar(im1, ax=ax1, fraction=0.035, pad=0.02)
cb1.set_label("F1 Improvement (%)", fontsize=16, fontweight='bold')
cb1.ax.tick_params(labelsize=14)

cb2 = fig.colorbar(im2, ax=ax2, fraction=0.035, pad=0.02)
cb2.set_label("F1 Improvement (%)", fontsize=16, fontweight='bold')
cb2.ax.tick_params(labelsize=14)

plt.subplots_adjust(wspace=0.18, bottom=0.22, top=0.88)
plt.tight_layout()
plt.savefig("output/visualizations/figure1_heatmap.pdf", bbox_inches='tight', dpi=300)
plt.savefig("output/visualizations/figure1_heatmap.png", bbox_inches='tight', dpi=300)
plt.close()
print("Figure 1 done.")

# ============================================================
# FIGURE 2: Model Architecture line plot
# ============================================================
models_order = ["BERT", "RoBERTa", "Mistral", "Llama3", "GPT-4o"]
model_labels = ["BERT\n(~110M)", "RoBERTa\n(~125M)", "Mistral\n(~7B)", "Llama3\n(~8B)", "GPT-4o\n(Large)"]

avg_g4o = []
avg_m = []
for m in models_order:
    sub = df[df["Model"] == m]
    avg_g4o.append(sub["dF1_G4o"].mean())
    avg_m.append(sub["dF1_M"].mean())

fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(models_order))
ax.plot(x, avg_g4o, 'o-', color='#2E86AB', linewidth=2.5, markersize=10, label='CC-G4o',
        markeredgecolor='black', markeredgewidth=1.5, zorder=5)
ax.plot(x, avg_m, 's-', color='#F18F01', linewidth=2.5, markersize=10, label='CC-M',
        markeredgecolor='black', markeredgewidth=1.5, zorder=5)

for i in range(len(models_order)):
    ax.annotate(f"{avg_g4o[i]:+.2f}", (x[i], avg_g4o[i]), textcoords="offset points",
                xytext=(0, 14), ha='center', fontsize=16, color='#2E86AB', fontweight='bold')
    ax.annotate(f"{avg_m[i]:+.2f}", (x[i], avg_m[i]), textcoords="offset points",
                xytext=(0, -18), ha='center', fontsize=16, color='#F18F01', fontweight='bold')

ax.axhline(0, color='black', linewidth=1.5, linestyle='--', alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=18)
ax.set_xlabel("Model Architecture (by parameter size)", fontsize=18, fontweight='bold')
ax.set_ylabel("Average ΔF1 (across both datasets)", fontsize=18, fontweight='bold')
ax.tick_params(axis='y', labelsize=18)
ax.legend(fontsize=12, loc='upper right', framealpha=0.9, edgecolor='black')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("output/visualizations/figure2_model_architecture.pdf", bbox_inches='tight', dpi=300)
plt.savefig("output/visualizations/figure2_model_architecture.png", bbox_inches='tight', dpi=300)
plt.close()
print("Figure 2 done.")

# ============================================================
# FIGURE 3: Learning Settings bar chart — split by dataset (2×2)
# ============================================================
settings = ["Fine-tuning", "Zero-shot", "Few-shot"]
datasets = ["CT22", "PoliClaim"]

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
x = np.arange(len(settings))
width = 0.35

color_g4o = '#2E86AB'
color_m = '#F18F01'


def annotate_bars(ax, bar_group):
    for bar in bar_group:
        h = bar.get_height()
        va = 'bottom' if h >= 0 else 'top'

        offset = 0.01 * ax.get_ylim()[1]

        y = h + offset if h >= 0 else h - offset

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{h:+.2f}",
            ha='center',
            va=va,
            fontsize=16,
            fontweight='bold'
        )


for col, ds in enumerate(datasets):
    sub = df[df["Dataset"] == ds]

    # Row 0: F1 Score Changes
    ax = axes[0, col]
    vals_g4o = [sub[sub["Setting"] == s]["dF1_G4o"].mean() for s in settings]
    vals_m = [sub[sub["Setting"] == s]["dF1_M"].mean() for s in settings]
    b1 = ax.bar(x - width / 2, vals_g4o, width, color=color_g4o, label='CC-G4o',
                edgecolor='black', linewidth=1.2, alpha=0.85)
    b2 = ax.bar(x + width / 2, vals_m, width, color=color_m, label='CC-M',
                edgecolor='black', linewidth=1.2, alpha=0.85)
    ax.axhline(0, color='black', linewidth=1.5, linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(settings, fontsize=18)
    ax.set_ylabel("Average ΔF1", fontsize=18, fontweight='bold')
    ax.set_title(f"(a) F1 Score Changes — {ds}", fontsize=20, fontweight='bold')
    ax.tick_params(axis='y', labelsize=18)
    ax.legend(fontsize=12, loc='upper right', framealpha=0.9, edgecolor='black')
    ax.grid(axis='y', alpha=0.3)
    annotate_bars(ax, b1)
    annotate_bars(ax, b2)

    # Row 1: Accuracy Changes
    ax = axes[1, col]
    vals_g4o = [sub[sub["Setting"] == s]["dAcc_G4o"].mean() for s in settings]
    vals_m = [sub[sub["Setting"] == s]["dAcc_M"].mean() for s in settings]
    b3 = ax.bar(x - width / 2, vals_g4o, width, color=color_g4o, label='CC-G4o',
                edgecolor='black', linewidth=1.2, alpha=0.85)
    b4 = ax.bar(x + width / 2, vals_m, width, color=color_m, label='CC-M',
                edgecolor='black', linewidth=1.2, alpha=0.85)
    ax.axhline(0, color='black', linewidth=1.5, linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(settings, fontsize=18)
    ax.set_ylabel("Average ΔAccuracy", fontsize=18, fontweight='bold')
    ax.set_title(f"(b) Accuracy Changes — {ds}", fontsize=20, fontweight='bold')
    ax.tick_params(axis='y', labelsize=18)
    ax.legend(fontsize=12, loc='upper right', framealpha=0.9, edgecolor='black')
    ax.grid(axis='y', alpha=0.3)
    annotate_bars(ax, b3)
    annotate_bars(ax, b4)

plt.tight_layout()
plt.savefig("output/visualizations/figure3_learning_settings.pdf", bbox_inches='tight', dpi=300)
plt.savefig("output/visualizations/figure3_learning_settings.png", bbox_inches='tight', dpi=300)
plt.close()
print("Figure 3 done.")

# ============================================================
# FIGURE 4: Precision-Recall tradeoff scatter
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

cmap = plt.cm.RdYlGn
vmin, vmax = -5, 5

for dataset, ax, title in [("CT22", ax1, "CT22 Dataset"), ("PoliClaim", ax2, "PoliClaim Dataset")]:
    sub = df[df["Dataset"] == dataset]

    for _, row in sub.iterrows():
        size_g4o = max(abs(row["dF1_G4o"]) * 30, 20)
        norm_val = (row["dF1_G4o"] - vmin) / (vmax - vmin)
        norm_val = np.clip(norm_val, 0, 1)
        c = cmap(norm_val)
        ax.scatter(row["dPrec_G4o"], row["dRec_G4o"], s=size_g4o, c=[c],
                   edgecolors='#2E86AB', linewidths=1.5, zorder=5, alpha=0.85)

        size_m = max(abs(row["dF1_M"]) * 30, 20)
        norm_val_m = (row["dF1_M"] - vmin) / (vmax - vmin)
        norm_val_m = np.clip(norm_val_m, 0, 1)
        c_m = cmap(norm_val_m)
        ax.scatter(row["dPrec_M"], row["dRec_M"], s=size_m, c=[c_m],
                   edgecolors='#F18F01', linewidths=1.5, zorder=5, alpha=0.85)

    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xlabel("Precision Change (%)", fontsize=18, fontweight='bold')
    ax.set_ylabel("Recall Change (%)", fontsize=18, fontweight='bold')
    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', labelsize=18)

    ax.scatter([], [], s=60, c='lightgreen', edgecolors='#2E86AB', linewidths=1.5, label='CC-G4o')
    ax.scatter([], [], s=60, c='lightyellow', edgecolors='#F18F01', linewidths=1.5, label='CC-M')
    ax.legend(fontsize=12, loc='upper right', framealpha=0.9, edgecolor='black')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cb1 = fig.colorbar(sm, ax=ax1, shrink=0.8)
cb1.set_label("F1 Improvement (%)", fontsize=18, fontweight='bold')
cb1.ax.tick_params(labelsize=16)
cb2 = fig.colorbar(sm, ax=ax2, shrink=0.8)
cb2.set_label("F1 Improvement (%)", fontsize=18, fontweight='bold')
cb2.ax.tick_params(labelsize=16)

plt.tight_layout()
plt.savefig("output/visualizations/figure4_prec_rec_tradeoff.pdf", bbox_inches='tight', dpi=300)
plt.savefig("output/visualizations/figure4_prec_rec_tradeoff.png", bbox_inches='tight', dpi=300)
plt.close()
print("Figure 4 done.")

print("\nAll figures generated successfully!")