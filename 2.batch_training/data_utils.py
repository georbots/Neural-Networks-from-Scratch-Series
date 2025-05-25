import numpy as np

def create_batches(X, y, batch_size, shuffle=True):
    """Split data into batches, optionally shuffle."""
    N = X.shape[0]
    print(N)

    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)
    x_batches = []
    y_batches = []

    for start in range(0, N, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        X_batch = X[batch_idx]
        y_batch = y[batch_idx]

        x_batches.append(X_batch)
        y_batches.append(y_batch)
    return x_batches, y_batches
