# Batch Training for Perceptron

## Overview

This module builds upon the single-sample perceptron training approach ([Perceptron from Scratch](../1.perceptron/README.md)), by implementing **mini-batch gradient descent**.  

---

## Mini-Batch Gradient Descent Theory

Mini-batch training updates model parameters after processing small subsets (batches) of the training data instead of individual samples or the entire dataset. This approach balances:

- **Efficiency** (faster than full batch)  
- **Stability** (less noisy than pure stochastic updates)

### Key steps:

- Split data into mini-batches of size \( B \).  
- For each batch:  
  - Compute weighted sums and activations for all samples in the batch.  
  - Calculate the average loss over the batch.  
  - Compute gradients averaged across the batch.  
  - Update weights and bias accordingly.

---

## Mathematical Formulation of Batch Training

Let the dataset have \( N \) samples, each with \( d \) features:  
\[
\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_N], \quad \mathbf{x}_i \in \mathbb{R}^d
\]

and corresponding labels:  
\[
\mathbf{y} = [y_1, y_2, ..., y_N], \quad y_i \in \{0, 1\}
\]

1. Partition dataset into mini-batches

Split \(\mathbf{X}\) and \(\mathbf{y}\) into \( M = \lceil \frac{N}{B} \rceil \) batches of size \( B \):

\[
\mathbf{X} = [\mathbf{X}^{(1)}, \mathbf{X}^{(2)}, ..., \mathbf{X}^{(M)}]
\]

\[
\mathbf{y} = [\mathbf{y}^{(1)}, \mathbf{y}^{(2)}, ..., \mathbf{y}^{(M)}]
\]

where each batch \(\mathbf{X}^{(m)}\) has shape \((B, d)\).

---

2. Forward pass for mini-batch \( m \)

Compute the linear combination for all batch samples:

\[
\mathbf{z}^{(m)} = \mathbf{X}^{(m)} \mathbf{w} + b
\]

where

- \(\mathbf{w} \in \mathbb{R}^d\) is the weight vector,  
- \(b \in \mathbb{R}\) is the bias scalar,  
- \(\mathbf{z}^{(m)} \in \mathbb{R}^B\) is the pre-activation output for batch \(m\).

---

3. Apply activation function (sigmoid)

\[
\hat{\mathbf{y}}^{(m)} = \sigma(\mathbf{z}^{(m)}) = \frac{1}{1 + e^{-\mathbf{z}^{(m)}}}
\]

\(\hat{\mathbf{y}}^{(m)} \in \mathbb{R}^B\) contains predicted probabilities for the batch.

---

4. Compute binary cross-entropy loss (average over batch)

\[
\mathcal{L}^{(m)} = - \frac{1}{B} \sum_{i=1}^B \left[ y_i^{(m)} \log \hat{y}_i^{(m)} + (1 - y_i^{(m)}) \log (1 - \hat{y}_i^{(m)}) \right]
\]

---

5. Calculate gradients (average over batch)

Weight gradient:

\[
\nabla_{\mathbf{w}}^{(m)} = \frac{1}{B} \mathbf{X}^{(m)^\top} \left( \hat{\mathbf{y}}^{(m)} - \mathbf{y}^{(m)} \right)
\]

Bias gradient:

\[
\nabla_{b}^{(m)} = \frac{1}{B} \sum_{i=1}^B \left( \hat{y}_i^{(m)} - y_i^{(m)} \right)
\]

---

6. Update parameters

With learning rate \(\eta\):

\[
\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla_{\mathbf{w}}^{(m)}
\]

\[
b \leftarrow b - \eta \nabla_{b}^{(m)}
\]

---

Repeat steps 2-6 for each mini-batch \( m = 1, 2, ..., M \) and over all epochs until convergence.

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
