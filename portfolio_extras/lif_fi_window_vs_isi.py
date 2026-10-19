"""
Lab 4 follow-up: is the LIF F-I curve really saturating at 140 Hz?
------------------------------------------------------------------
In Lab 4 my F-I curve went flat at 140 Hz for I = 12, 14, 16 and I said it was saturating
because of the 5 ms refractory period (max 1/5 ms = 200 Hz).

Check: the rate was measured by COUNTING spikes in a 50 ms window (10-60 ms).
One extra spike in 50 ms = +20 Hz, so the window rate can only be 0, 20, 40, ... Hz.
Here I compare that with the rate from the inter-spike interval (1/ISI) and with the
exact LIF formula:
    ISI = t_ref + tau * ln( R*I / (R*I - (V_th - V_rest)) )     (only if R*I > V_th - V_rest)

Same model and parameters as Lab 4 (models/lif.py and config.py).
Run:  python lif_fi_window_vs_isi.py   (saves figures/lif_fi_window_vs_isi.png)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False   # plain "-" on axis labels

# Lab 4 config.py
DT, T_TOTAL = 0.01, 80.0
TAU_M, V_REST, V_RESET, V_TH, R_M, REFRAC_MS = 20.0, -65.0, -65.0, -50.0, 10.0, 5.0


def simulate_lif(I, t):
    """Same loop as my Lab 4 models/lif.py."""
    V = np.zeros_like(t)
    V[0] = V_REST
    spike_times, refrac_steps = [], 0
    for i in range(1, len(t)):
        if refrac_steps > 0:
            V[i] = V_RESET
            refrac_steps -= 1
            continue
        V[i] = V[i - 1] + DT * (-(V[i - 1] - V_REST) + R_M * I[i - 1]) / TAU_M
        if V[i] >= V_TH:
            spike_times.append(t[i])
            V[i] = V_RESET
            refrac_steps = int(REFRAC_MS / DT)
    return V, np.array(spike_times)


def step_current(t, t_on=10.0, t_off=60.0, amp=10.0):
    I = np.zeros_like(t)
    I[(t >= t_on) & (t <= t_off)] = amp
    return I


def theory_rate(amp):
    drive = R_M * amp
    gap = V_TH - V_REST
    if drive <= gap:
        return 0.0
    isi = REFRAC_MS + TAU_M * np.log(drive / (drive - gap))
    return 1000.0 / isi


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    t = np.arange(0, T_TOTAL + DT, DT)
    amps = np.arange(2, 41, 2)
    win, isi_rate, th = [], [], []
    for a in amps:
        _, sp = simulate_lif(step_current(t, amp=a), t)
        sp_win = sp[(sp >= 10) & (sp <= 60)]
        win.append(len(sp_win) / 0.05)
        isi_rate.append(1000.0 / np.mean(np.diff(sp_win)) if len(sp_win) > 1 else np.nan)
        th.append(theory_rate(a))
    for a, w, r, x in zip(amps, win, isi_rate, th):
        print(f"I = {a:>2}: window count = {w:5.0f} Hz | 1/ISI = {r:6.1f} Hz | formula = {x:6.1f} Hz")

    plt.figure(figsize=(7, 4))
    plt.step(amps, win, where="mid", color="C1", label="count in 50 ms window (what I did in Lab 4)")
    plt.plot(amps, isi_rate, "o", color="C0", label="1 / mean ISI (from the simulation)")
    plt.plot(amps, th, "k--", lw=1, label="exact LIF formula")
    plt.axhline(1000 / REFRAC_MS, color="r", ls=":", label="refractory limit 1/5 ms = 200 Hz")
    plt.xlabel("Current amplitude (a.u.)")
    plt.ylabel("Firing rate (Hz)")
    plt.title("LIF F-I curve: the 140 Hz 'plateau' was a counting artefact")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/lif_fi_window_vs_isi.png", dpi=150)
    print("saved figures/lif_fi_window_vs_isi.png")
