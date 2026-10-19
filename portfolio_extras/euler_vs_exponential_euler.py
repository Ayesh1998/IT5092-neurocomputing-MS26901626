"""
Why my assignment SNN only reached 10% accuracy with method='euler'
-------------------------------------------------------------------
One excitatory neuron from the assignment (same equation and parameters as config.py),
with ge = 0 and a burst of k inhibitory spikes arriving together at t = 10 ms.
Each inhibitory spike adds W_INH_TO_EXC = 17 to gi.

  tau * dv/dt = -(v - V_rest) + ge*(E_exc - v) + gi*(E_inh - v)

Plain Euler takes a straight-line step:   v <- v + dt * dv/dt
  This is only stable if dt * (1 + ge + gi) / tau < 2,
  i.e. gi < 2*tau/dt - 1 = 399 with tau = 100 ms and dt = 0.5 ms.
  So if about 24 or more inhibitory neurons fire at the same time, v overshoots and explodes.

Exponential Euler solves the linear equation exactly over one step:
  v_inf = (V_rest + ge*E_exc + gi*E_inh) / (1 + ge + gi)
  v     <- v_inf + (v - v_inf) * exp(-dt * (1 + ge + gi) / tau)
  so v can never jump past v_inf (it stays between -100 mV and 0 mV).

Run:  python euler_vs_exponential_euler.py   (saves figures/euler_vs_exp_euler.png)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False   # plain "-" on axis labels

# values from the assignment config.py
DT = 0.5            # ms
TAU_M = 100.0       # ms
TAU_GI = 2.0        # ms
V_REST = -65.0      # mV
E_EXC = 0.0         # mV
E_INH = -100.0      # mV
V_THRESH = -52.0    # mV (threshold at the start, theta = 20 mV)
W_INH_TO_EXC = 17.0


def simulate(k_inh_spikes, method, t_end=40.0, t_kick=10.0):
    """Return time, v trace and the times where v crossed threshold."""
    t = np.arange(0, t_end + DT, DT)
    v = np.zeros_like(t)
    v[0] = V_REST
    gi = 0.0
    ge = 0.0
    crossings = []
    for i in range(1, len(t)):
        if abs(t[i - 1] - t_kick) < 1e-9:
            gi += k_inh_spikes * W_INH_TO_EXC          # k inhibitory spikes arrive together
        vp = v[i - 1]
        if method == "euler":
            dv = (-(vp - V_REST) + ge * (E_EXC - vp) + gi * (E_INH - vp)) / TAU_M
            v[i] = vp + DT * dv
        else:  # exponential euler
            g_tot = 1.0 + ge + gi
            v_inf = (V_REST + ge * E_EXC + gi * E_INH) / g_tot
            v[i] = v_inf + (vp - v_inf) * np.exp(-DT * g_tot / TAU_M)
        gi = gi * np.exp(-DT / TAU_GI)                  # gi decays (same for both)
        if v[i] > V_THRESH and vp <= V_THRESH:
            crossings.append(t[i])
    return t, v, crossings


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    print(f"Euler is stable only while gi < {2 * TAU_M / DT - 1:.0f}, "
          f"i.e. fewer than {int((2 * TAU_M / DT - 1) // W_INH_TO_EXC) + 1} inhibitory spikes at once")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, k in zip(axes, [10, 60]):
        for method, style in [("euler", "C3-"), ("exponential_euler", "C0-")]:
            t, v, cr = simulate(k, method)
            ax.plot(t, v, style, lw=1.8, label=f"{method}")
            print(f"k = {k:>2} inhibitory spikes | {method:17s} | min v = {v.min():10.1f} mV | "
                  f"max v = {v.max():10.1f} mV | false threshold crossings = {len(cr)}")
        ax.axhline(V_THRESH, color="k", ls="--", lw=1, label="threshold (-52 mV)")
        ax.axhline(E_INH, color="grey", ls=":", lw=1, label="E_inh (-100 mV)")
        ax.set_title(f"{k} inhibitory spikes at once (gi jumps to {k * W_INH_TO_EXC:.0f})")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("v (mV)")
        if k > 30:
            ax.set_yscale("symlog", linthresh=100)
        ax.legend(fontsize=8, loc="lower right" if k <= 30 else "upper right")
    plt.suptitle("Same neuron, same input: plain Euler explodes when inhibition is strong", fontsize=11)
    plt.tight_layout()
    plt.savefig("figures/euler_vs_exp_euler.png", dpi=150, bbox_inches="tight")
    print("saved figures/euler_vs_exp_euler.png")
