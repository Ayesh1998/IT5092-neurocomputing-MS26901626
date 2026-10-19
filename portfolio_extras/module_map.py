"""Draw the 'how the module connects' map used at the start of the portfolio.
Run:  python module_map.py   (saves figures/module_map.png)"""
import os
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["axes.unicode_minus"] = False   # plain "-" on axis labels
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BLUE = GREEN = ORANGE = GREY = ("#000000", "#ffffff")   # plain black and white
W, H = 2.15, 1.05
boxes = {
    1: (0, 1, "1. Biology", "brain areas, neurons,\naction potential", "L1, L3", BLUE),
    2: (2.5, 1, "2. Measuring", "EEG sees synchronised\npopulations; spikes", "L2, L3 · Lab 1, 2", BLUE),
    3: (5, 1, "3. Neuron models", "HH → LIF → Izhikevich\n+ Poisson input noise", "L4, L5 · Lab 4, 5", GREEN),
    4: (7.5, 1, "4. Neural code", "rate, timing, population\nencoding ↔ decoding", "L3, L6 · Lab 3, 7", GREEN),
    5: (7.5, -0.6, "5. Information", "entropy and\nmutual information", "L9 · Lab 9", GREEN),
    6: (5, -0.6, "6. Learning", "Hebbian, STDP,\nintrinsic plasticity", "L5 · Lab 6", ORANGE),
    7: (2.5, -0.6, "7. Networks", "MLP + backprop vs\nSNN (Diehl & Cook)", "L7, L8 · Assignment", ORANGE),
    8: (0, -0.6, "The thread I follow", "biology → equation →\nmodel → data → meaning", "every section", GREY),
}

fig, ax = plt.subplots(figsize=(10, 3.6))
for k, (x, y, title, body, tag, (edge, fill)) in boxes.items():
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fill, ec=edge, lw=1.4, ls="--" if k == 8 else "-"))
    ax.text(x + W / 2, y + H - 0.2, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color="#000000")
    ax.text(x + W / 2, y + H / 2 - 0.05, body, ha="center", va="center", fontsize=8.6, color="#000000", linespacing=1.15)
    ax.text(x + W / 2, y + 0.13, tag, ha="center", va="center", fontsize=7.8, color=edge)


def arrow(p, q, rad=0.0, text=None, txy=None):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.4, color="#000000",
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text(*txy, text, fontsize=7.8, color="#000000", ha="center", va="center", style="italic")


mid = 1 + H / 2
low = -0.6 + H / 2
arrow((W + 0.02, mid), (2.48, mid))
arrow((2.5 + W + 0.02, mid), (4.98, mid))
arrow((5 + W + 0.02, mid), (7.48, mid))
arrow((7.5 + W / 2, 0.98), (7.5 + W / 2, -0.6 + H + 0.02))
arrow((7.48, low), (5 + W + 0.02, low))
arrow((4.98, low), (2.5 + W + 0.02, low))
arrow((5 + W / 2, 0.98), (5 + W / 2, -0.6 + H + 0.02), text="spikes drive\nplasticity", txy=(5 + W / 2 + 0.6, 0.71))

ax.set_xlim(-0.1, 9.75)
ax.set_ylim(-0.7, 2.15)
ax.axis("off")
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/module_map.png", dpi=220, bbox_inches="tight", pad_inches=0.05)
print("saved figures/module_map.png")
