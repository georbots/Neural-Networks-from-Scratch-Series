import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import io
from perceptron import Perceptron


def get_dataset(name):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    if name == "AND":
        y = np.array([0, 0, 0, 1])
    elif name == "OR":
        y = np.array([0, 1, 1, 1])
    elif name == "XOR":
        y = np.array([0, 1, 1, 0])
    elif name == "NAND":
        y = np.array([1, 1, 1, 0])
    elif name == "NOR":
        y = np.array([1, 0, 0, 0])
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return X, y


def create_decision_boundary_gif_in_memory(X, y, snapshots, output_path, interval=0.1):
    images = []

    for epoch, weights, bias in snapshots:
        fig, ax = plt.subplots(figsize=(4, 4))

        xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        z = 1 / (1 + np.exp(-(grid @ weights + bias)))
        Z = z.reshape(xx.shape)

        ax.contour(xx, yy, Z, levels=[0.5], colors='black')
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
        ax.set_title(f"Epoch {epoch}")
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.grid(True)

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        image = imageio.imread(buf)
        images.append(image)

    imageio.mimsave(output_path, images, duration=interval, loop=0)
    print(f"Decision boundary GIF saved as {output_path}")


def main(logic_type="AND"):
    print(f"Training on {logic_type} dataset...\n")
    X, y = get_dataset(logic_type)

    model = Perceptron(shape=(2,))
    snapshots = []
    PLOT_INTERVAL = 100
    EPOCHS = 10000

    def capture_snapshot(epoch, weights, bias):
        if epoch % PLOT_INTERVAL == 0 or epoch == EPOCHS - 1:
            snapshots.append((epoch, weights.copy(), bias))

    model.train(X, y, num_epochs=EPOCHS, callback=capture_snapshot)

    create_decision_boundary_gif_in_memory(
        X, y, snapshots, output_path=f"decision_boundary_gifs/{logic_type.lower()}_boundary.gif", interval=0.1
    )


if __name__ == "__main__":
    modes = ["AND", "OR", "XOR", "NAND", "NOR"]
    for mode in modes:
        main(logic_type=mode)
