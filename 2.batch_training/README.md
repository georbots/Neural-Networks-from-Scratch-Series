# Batch Training for Perceptron

## Overview

This module builds upon the single-sample perceptron training approach ([Perceptron from Scratch](../1.perceptron/README.md)), by implementing **mini-batch gradient descent**.  

---

## Mini-Batch Gradient Descent Theory

Mini-batch training updates model parameters after processing small subsets (batches) of the training data instead of individual samples or the entire dataset. This approach balances:

- **Efficiency** (faster than full batch)  
- **Stability** (less noisy than pure stochastic updates)

### Key steps:

- Split data into mini-batches of size B.  
- For each batch:  
  - Compute weighted sums and activations for all samples in the batch.  
  - Calculate the average loss over the batch.  
  - Compute gradients averaged across the batch.  
  - Update weights and bias accordingly.

---

## Mathematical Formulation of Batch Training

Let the dataset have N samples, each with d features:  
X = [x_1, x_2, ..., x_N], where each x_i is a d-dimensional vector

and corresponding labels:  
y = [y_1, y_2, ..., y_N], where each y_i is 0 or 1

1. Partition dataset into mini-batches

Split X and y into M batches of size B:

X = [X^(1), X^(2), ..., X^(M)]  
y = [y^(1), y^(2), ..., y^(M)]

where each batch X^(m) has shape (B, d)

2. Forward pass for mini-batch m

Calculate weighted sums for all samples in batch m:

z^(m) = X^(m) dot w + b

where  
- w is the weights vector of shape (d,)  
- b is the bias scalar  
- z^(m) is a vector of length B

3. Apply activation (sigmoid)

For each element of z^(m):

y_hat^(m) = 1 / (1 + exp(-z^(m)))

This gives predicted probabilities for batch m.

4. Compute average binary cross-entropy loss over batch

Loss^(m) = -(1/B) * sum_{i=1 to B} [ y_i^(m) * log(y_hat_i^(m)) + (1 - y_i^(m)) * log(1 - y_hat_i^(m)) ]

5. Compute gradients (average over batch)

Gradient w.r.t weights:

grad_w^(m) = (1/B) * X^(m).T dot (y_hat^(m) - y^(m))

Gradient w.r.t bias:

grad_b^(m) = (1/B) * sum_{i=1 to B} (y_hat_i^(m) - y_i^(m))

6. Update parameters with learning rate eta

w = w - eta * grad_w^(m)  
b = b - eta * grad_b^(m)

Repeat steps 2-6 for all batches and epochs.

---

## Synthetic Dataset Description

To test batch training, we generate a simple **linearly separable 2D dataset** with:

- 300 samples (150 per class)  
- Class 0: Gaussian centered at (1,1)  
- Class 1: Gaussian centered at (4,4)  
- Added Gaussian noise to make the task realistic  

This dataset enables visualization of decision boundaries and training progress in 2D space.

---

## References

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.  
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.  

---
