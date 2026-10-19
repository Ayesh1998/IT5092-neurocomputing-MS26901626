"""
Lab 1 follow-up: eyes OPEN (Run 1) vs eyes CLOSED (Run 2)
---------------------------------------------------------
In Lab 1 I compared two segments of the SAME eyes-open run, so the alpha difference was tiny
(only about half the channels agreed). In my mini-report I said I should compare Run 1 vs Run 2
and cut each run into many short epochs so I can do a statistical test. This script does that.

Pipeline = same as Lab 1: average reference -> 1-45 Hz band-pass -> 50 Hz notch -> Welch PSD
Data: PhysioNet EEG Motor Movement/Imagery dataset (BCI2000), subject S001.
Expect: alpha (8-12 Hz) much stronger with eyes closed, mainly at occipital channels (O1, Oz, O2).

Run:  python eeg_eyes_open_vs_closed.py   (data in ../lab01_eeg_rhythms/files; saves figures/eeg_open_vs_closed.png)
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False   # plain "-" on axis labels
import mne
from scipy import stats

mne.set_log_level("WARNING")
ALPHA = (8, 12)
OCCIPITAL = ["O1..", "Oz..", "O2.."]


def load_and_clean(path):
    raw = mne.io.read_raw_edf(path, preload=True)
    raw.set_eeg_reference("average")
    raw.filter(1.0, 45.0)
    raw.notch_filter([50.0])
    return raw


def epoch_alpha(raw, picks, epoch_s=2.0):
    """Cut the run into 2 s epochs and return the alpha power of each epoch (mean over picks)."""
    epochs = mne.make_fixed_length_epochs(raw, duration=epoch_s, preload=True)
    psd = epochs.compute_psd(method="welch", fmin=1, fmax=45, n_fft=256, picks=picks)
    data, freqs = psd.get_data(return_freqs=True)          # (n_epochs, n_channels, n_freqs)
    idx = (freqs >= ALPHA[0]) & (freqs <= ALPHA[1])
    return data[:, :, idx].mean(axis=(1, 2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../lab01_eeg_rhythms/files", help="folder that contains S001/S001R01.edf")
    args = ap.parse_args()
    os.makedirs("figures", exist_ok=True)

    raw_open = load_and_clean(os.path.join(args.data, "S001", "S001R01.edf"))
    raw_closed = load_and_clean(os.path.join(args.data, "S001", "S001R02.edf"))

    # 1) whole-run PSD at the occipital channels
    psd_o = raw_open.compute_psd(method="welch", fmin=1, fmax=45, n_fft=2048, picks=OCCIPITAL)
    psd_c = raw_closed.compute_psd(method="welch", fmin=1, fmax=45, n_fft=2048, picks=OCCIPITAL)
    po, f = psd_o.get_data(return_freqs=True)
    pc, _ = psd_c.get_data(return_freqs=True)
    po_db, pc_db = 10 * np.log10(po.mean(0)), 10 * np.log10(pc.mean(0))

    # 2) alpha power per channel (all 64) -> which channels change most
    pa_o = raw_open.compute_psd(method="welch", fmin=1, fmax=45, n_fft=2048)
    pa_c = raw_closed.compute_psd(method="welch", fmin=1, fmax=45, n_fft=2048)
    do, fa = pa_o.get_data(return_freqs=True)
    dc, _ = pa_c.get_data(return_freqs=True)
    ia = (fa >= ALPHA[0]) & (fa <= ALPHA[1])
    ratio = dc[:, ia].mean(1) / do[:, ia].mean(1)
    names = raw_open.ch_names
    top = np.argsort(ratio)[::-1][:5]
    print("Channels where alpha (closed / open) is up:", int((ratio > 1).sum()), "of", len(ratio))
    print("Top 5 channels by closed/open alpha ratio:",
          ", ".join(f"{names[i].strip('.')} x{ratio[i]:.1f}" for i in top))
    occ_ratio = dc[[names.index(c) for c in OCCIPITAL]][:, ia].mean() / do[[names.index(c) for c in OCCIPITAL]][:, ia].mean()
    print(f"Occipital (O1, Oz, O2) alpha: closed / open = {occ_ratio:.2f}x")

    # 3) epoch-level statistics (many 2 s epochs per condition)
    a_open = epoch_alpha(raw_open, OCCIPITAL)
    a_closed = epoch_alpha(raw_closed, OCCIPITAL)
    t_stat, p_val = stats.ttest_ind(a_closed, a_open, equal_var=False)
    print(f"Epochs: open n = {len(a_open)}, closed n = {len(a_closed)}")
    print(f"Welch t-test (closed vs open, occipital alpha): t = {t_stat:.2f}, p = {p_val:.2e}")
    print(f"Epochs where closed alpha > median open alpha: "
          f"{(a_closed > np.median(a_open)).mean() * 100:.0f}%")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
    axes[0].plot(f, po_db, color="C0", label="Run 1: eyes open")
    axes[0].plot(f, pc_db, color="C3", label="Run 2: eyes closed")
    axes[0].axvspan(*ALPHA, color="gold", alpha=0.3, label="alpha 8-12 Hz")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Power (dB)")
    axes[0].set_title(f"Occipital PSD (O1, Oz, O2): alpha x{occ_ratio:.1f} with eyes closed")
    axes[0].legend(fontsize=8)
    axes[1].boxplot([a_open * 1e12, a_closed * 1e12])
    axes[1].set_xticks([1, 2], ["eyes open", "eyes closed"])
    axes[1].set_ylabel("Alpha power per 2 s epoch (x1e-12 V²/Hz)")
    axes[1].set_title(f"{len(a_open)} vs {len(a_closed)} epochs, Welch t-test p = {p_val:.1e}")
    plt.tight_layout()
    plt.savefig("figures/eeg_open_vs_closed.png", dpi=150)
    print("saved figures/eeg_open_vs_closed.png")
