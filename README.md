# IT5092 Neuroscience and Neurocomputing - lab work, assignment and portfolio code

**P. K. A. Lakshan · MS26901626 · MSc Year 1 Semester 2**

This repository holds my work for IT5092: the lab notebooks and code, my spiking-neural-network assignment,
and the short follow-up scripts behind the figures in my Individual Learning Portfolio.

## What is where

| Folder | Topic | What is inside |
|---|---|---|
| `lab01_eeg_rhythms/` | Lab 1: EEG rhythms and brain state | Notebook: filtering, PSD and band power on real 64-channel EEG |
| `lab02_eeg_physics/` | Lab 2: the physics of EEG | Notebook, plots and report: synchrony, dipole depth and orientation, overlapping sources |
| `lab03_spike_trains/` | Lab 3: neural coding | Notebook and report: Poisson spikes, refractoriness, coincidence detection, population decoding |
| `lab04_hh_vs_lif/` | Lab 4: Hodgkin-Huxley vs LIF | `models/hodgkin_huxley.py`, `models/lif.py`, notebook, answers |
| `lab05_izhikevich/` | Lab 5: Izhikevich model | `models/izhikevich.py`, notebook |
| `lab06_hebbian_memory/` | Lab 6: Hebbian learning | `models/hebbian.py`, `models/associative_memory.py`, notebook |
| `lab07_encoding_decoding/` | Lab 7 (Parts 1-2): encoding and decoding | Bayesian decoder, Gaussian tuning curve, population vector, answers |
| `lab09_information_theory/` | Lab 9: information theory | Entropy, mutual information, spike-train entropy, coding efficiency, answers |
| `assignment_snn_mnist/` | Assignment | Diehl & Cook (2015) spiking network for MNIST in Brian2 (notebook, `config.py`, `utils/`) |
| `portfolio_extras/` | Portfolio follow-ups | Scripts that re-create the debugging and revisited results in the portfolio |

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook            # open any lab notebook from its own folder
```

Each lab notebook imports its own `config.py`, `models/` and `utils/`, so run it from inside its folder.

### Lab 1 data (not included, ~3 GB)

Lab 1 uses the **EEG Motor Movement/Imagery Dataset** from PhysioNet
(<https://physionet.org/content/eegmmidb/1.0.0/>). Download it and place the subject folders so that
`lab01_eeg_rhythms/files/S001/S001R01.edf` exists. Only `S001R01.edf` and `S001R02.edf` are needed for
the notebook and for `portfolio_extras/eeg_eyes_open_vs_closed.py`.

### Assignment

The MNIST data is downloaded automatically by scikit-learn on first run. Training 1,000 images with the
NumPy backend takes a long time; the saved notebook already contains all outputs.

## Portfolio follow-up scripts (`portfolio_extras/`)

| Script | What it shows | Portfolio section |
|---|---|---|
| `euler_vs_exponential_euler.py` | Why plain Euler made my SNN explode (10% accuracy) and exponential Euler fixed it (64%) | 5.1 |
| `eeg_eyes_open_vs_closed.py` | Lab 1 revisited: eyes open vs closed, 2 s epochs and a t-test | 5.3 |
| `lif_fi_window_vs_isi.py` | Lab 4: the 140 Hz "plateau" in my F-I curve was a counting artefact | 5.4 |
| `lecture_worked_examples.py` | Worked examples from the lectures (Bayes, forward pass, entropy, backprop) | 2.6, 2.9 |
| `module_map.py` | The concept map at the start of the portfolio | 1 |

```bash
cd portfolio_extras
python euler_vs_exponential_euler.py
python lif_fi_window_vs_isi.py
python eeg_eyes_open_vs_closed.py        # needs the Lab 1 data, see above
python lecture_worked_examples.py
```

## Notes

- Lab sheets, lecture slides, the assignment sheet and the Diehl & Cook paper belong to the lecturer and
  the authors, so they are not included here.
- Main reference for the assignment: Diehl, P. U. & Cook, M. (2015). Unsupervised learning of digit
  recognition using spike-timing-dependent plasticity. *Frontiers in Computational Neuroscience*, 9, 99.
