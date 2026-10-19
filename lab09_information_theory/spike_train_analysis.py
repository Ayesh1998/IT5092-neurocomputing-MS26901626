import numpy as np
from entropy_utils import compute_entropy

# -----------------------------------------
# Spike Train Binning
# -----------------------------------------

def bin_spike_train(spike_times, duration, bin_size):
    """
    Convert spike times to binned spike count vector.
    """

    # TODO 1:
    # Create bins from 0 to duration
    bins = np.arange(0, duration + bin_size, bin_size)

    # TODO 2:
    # Count spikes in each bin
    counts, edges = np.histogram(spike_times, bins=bins)

    return counts


def compute_spike_train_entropy(spike_trains, duration, bin_size):
    """
    Compute entropy of spike patterns across trials.
    """

    # TODO 3:
    # Convert each spike train into binned representation
    patterns = []
    for train in spike_trains:
        binned = bin_spike_train(train, duration, bin_size)
        patterns.append(tuple(binned))

    # TODO 4:
    # Treat each unique pattern as symbol
    pattern_counts = {}
    for pattern in patterns:
        if pattern in pattern_counts:
            pattern_counts[pattern] = pattern_counts[pattern] + 1
        else:
            pattern_counts[pattern] = 1

    # Compute entropy
    counts = np.array(list(pattern_counts.values()))
    probabilities = counts / np.sum(counts)
    H = compute_entropy(probabilities)

    return H