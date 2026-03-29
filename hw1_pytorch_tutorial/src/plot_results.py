import numpy as np
import matplotlib.pyplot as plt

# Data from 4 seed runs (each: 5 epochs)
data = {
    "FFN": [
        [0.8303, 0.8883, 0.9236, 0.9366, 0.9418],
        [0.8651, 0.9146, 0.9342, 0.9411, 0.9508],
        [0.8427, 0.9088, 0.9320, 0.9335, 0.9457],
        [0.8358, 0.9155, 0.9249, 0.9384, 0.9440],
    ],
    "GLU": [
        [0.8794, 0.9274, 0.9331, 0.9499, 0.9541],
        [0.8543, 0.9110, 0.9308, 0.9436, 0.9473],
        [0.8506, 0.9088, 0.9299, 0.9348, 0.9436],
        [0.8695, 0.9207, 0.9346, 0.9477, 0.9506],
    ],
    "GEGLU": [
        [0.8812, 0.9287, 0.9436, 0.9527, 0.9561],
        [0.8648, 0.9252, 0.9398, 0.9548, 0.9589],
        [0.8502, 0.9095, 0.9338, 0.9430, 0.9517],
        [0.8698, 0.9191, 0.9484, 0.9560, 0.9607],
    ],
    "SwiGLU": [
        [0.8731, 0.9291, 0.9430, 0.9475, 0.9562],
        [0.8600, 0.9260, 0.9393, 0.9519, 0.9533],
        [0.8530, 0.9081, 0.9337, 0.9388, 0.9507],
        [0.8706, 0.9241, 0.9454, 0.9531, 0.9595],
    ],
    "ReGLU": [
        [0.8766, 0.9213, 0.9444, 0.9545, 0.9566],
        [0.8655, 0.9282, 0.9461, 0.9591, 0.9596],
        [0.8413, 0.9077, 0.9305, 0.9477, 0.9486],
        [0.8721, 0.9144, 0.9443, 0.9544, 0.9595],
    ],
}

epochs = np.arange(1, 6)
colors = {
    "FFN": "#1f77b4",
    "GLU": "#ff7f0e",
    "GEGLU": "#2ca02c",
    "SwiGLU": "#d62728",
    "ReGLU": "#9467bd",
}

# --- Figure 1: Test Accuracy with variance bands ---
fig, ax = plt.subplots(figsize=(8, 5))

for name, runs in data.items():
    arr = np.array(runs)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    ax.plot(
        epochs,
        mean,
        marker="o",
        label=f"{name} ({mean[-1]:.4f})",
        color=colors[name],
        linewidth=2,
    )
    ax.fill_between(epochs, mean - std, mean + std, alpha=0.15, color=colors[name])

ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("Test Accuracy", fontsize=13)
ax.set_title("TinyViT on MNIST: FFN vs GLU Variants", fontsize=14)
ax.legend(fontsize=11, loc="lower right")
ax.set_xticks(epochs)
ax.set_ylim(0.82, 0.97)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("src/test_accuracy.png", dpi=150)
plt.show()

# --- Figure 2: Summary bar chart (final accuracy mean +/- std) ---
fig2, ax2 = plt.subplots(figsize=(7, 4))

names = list(data.keys())
means = [np.array(data[n])[:, -1].mean() for n in names]
stds = [np.array(data[n])[:, -1].std() for n in names]
bars = ax2.bar(
    names,
    means,
    yerr=stds,
    capsize=6,
    color=[colors[n] for n in names],
    edgecolor="black",
    linewidth=0.8,
)

ax2.set_ylabel("Final Test Accuracy", fontsize=13)
ax2.set_title("Final Accuracy Comparison (mean +/- std)", fontsize=14)
ax2.set_ylim(0.93, 0.97)
ax2.grid(True, alpha=0.3, axis="y")

for bar, m, s in zip(bars, means, stds):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + s + 0.001,
        f"{m:.4f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig("src/final_accuracy_bar.png", dpi=150)
plt.show()

# --- Print summary table ---
print("\n" + "=" * 55)
print(f"{'Variant':<10} {'Params':>8} {'Mean Acc':>10} {'Std':>8}")
print("-" * 55)
params = {
    "FFN": 105098,
    "GLU": 105010,
    "GEGLU": 105010,
    "SwiGLU": 105010,
    "ReGLU": 105010,
}
for name in names:
    arr = np.array(data[name])[:, -1]
    print(f"{name:<10} {params[name]:>8,} {arr.mean():>10.4f} {arr.std():>8.4f}")
