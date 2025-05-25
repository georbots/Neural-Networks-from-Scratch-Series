import numpy as np
import matplotlib.pyplot as plt
from data_utils import create_batches
from perceptron import Perceptron
import imageio.v2 as imageio
import io

snapshots = []  # <-- MUST be global or at least before capture_snapshot

def capture_snapshot(epoch, weights, bias):
    if epoch % 100 == 0 or epoch == EPOCHS - 1:
        snapshots.append((epoch, weights.copy(), bias))

def create_decision_boundary_gif(X, y, snapshots, output_path, interval=0.1):
    images = []

    for epoch, weights, bias in snapshots:
        fig, ax = plt.subplots(figsize=(4, 4))

        xx, yy = np.meshgrid(np.linspace(np.min(X[:,0]) - 0.5, np.max(X[:,0]) + 0.5, 200),
                             np.linspace(np.min(X[:,1]) - 0.5, np.max(X[:,1]) + 0.5, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]

        z = 1 / (1 + np.exp(-(grid @ weights + bias)))  # sigmoid activation
        Z = z.reshape(xx.shape)

        ax.contour(xx, yy, Z, levels=[0.5], colors='black')
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
        ax.set_title(f"Epoch {epoch}")
        ax.set_xlim(np.min(X[:,0]) - 0.5, np.max(X[:,0]) + 0.5)
        ax.set_ylim(np.min(X[:,1]) - 0.5, np.max(X[:,1]) + 0.5)
        ax.grid(True)

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        image = imageio.imread(buf)
        images.append(image)

    imageio.mimsave(output_path, images, duration=interval, loop=0)
    print(f"Saved decision boundary GIF to {output_path}")


np.random.seed(42)

# Generate data
class0 = np.random.randn(150, 2) + np.array([1, 1])
class1 = np.random.randn(150, 2) + np.array([4, 4])

X = np.vstack((class0, class1))
y = np.hstack((np.zeros(150), np.ones(150)))

# Create batches
X_batches, y_batches = create_batches(X, y, batch_size=16, shuffle=True)
print(len(X_batches), y_batches[0].shape)

EPOCHS = 10000
p = Perceptron(shape=2, learning_rate=0.0001, activation='sigmoid')

# Train with snapshot callback
loss_history, acc_history = p.train(X_batches, y_batches, num_epochs=EPOCHS, callback=capture_snapshot)

# After training create GIF with snapshots
create_decision_boundary_gif(
    X=np.vstack(X_batches),
    y=np.hstack(y_batches),
    snapshots=snapshots,
    output_path="decision_boundary.gif",
    interval=0.1
)

# Plot training progress
plt.plot(loss_history, label="Loss")
plt.plot(acc_history, label="Accuracy")
plt.legend()
plt.xlabel("Epochs")
plt.title("Training Progress")
plt.show()
