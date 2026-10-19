import matplotlib.pyplot as plt

def plot_weight_history(history, title="Weight Evolution"):
    plt.figure(figsize=(8, 4))
    for i in range(history.shape[1]):
        plt.plot(history[:, i], label=f"w{i}")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Weight value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def show_pattern(pattern, title="Pattern"):
    plt.figure(figsize=(3, 3))
    plt.imshow(pattern.reshape(4, 4), cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()