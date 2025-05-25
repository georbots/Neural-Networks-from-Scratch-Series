# Batch Training for Perceptron

## Overview

This module builds upon the single-sample perceptron training approach ([Perceptron from Scratch](../1.perceptron/README.md)), by implementing **mini-batch gradient descent**.  

---

## Mini-Batch Gradient Descent Theory

Mini-batch training updates model parameters after processing small subsets (batches) of the training data instead of individual samples or the entire dataset. This approach balances:

- **Efficiency** (faster than full batch)  
- **Stability** (less noisy than pure stochastic updates)

### Key steps:

1. Split data into mini-batches of size $B$  
2. For each batch:  
   - Compute weighted sums and activations for all samples  
   - Calculate the average loss over the batch  
   - Compute gradients averaged across the batch  
   - Update weights and bias  

---

## Mathematical Formulation

### 1. Dataset Notation

Let the dataset have $N$ samples, each with $d$ features:

$$
\mathbf{X} = \begin{bmatrix} \mathbf{x}_1^\top \\ \mathbf{x}_2^\top \\ \vdots \\ \mathbf{x}_N^\top \end{bmatrix} \in \mathbb{R}^{N \times d}, \quad 
\mathbf{y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_N \end{bmatrix} \in \{0,1\}^N
$$

### 2. Mini-Batch Partitioning

Split into $M = \lceil N/B \rceil$ batches:

$$
\mathbf{X} = \begin{bmatrix} \mathbf{X}^{(1)} \\ \mathbf{X}^{(2)} \\ \vdots \\ \mathbf{X}^{(M)} \end{bmatrix}, \quad
\mathbf{y} = \begin{bmatrix} \mathbf{y}^{(1)} \\ \mathbf{y}^{(2)} \\ \vdots \\ \mathbf{y}^{(M)} \end{bmatrix}
$$

Each $\mathbf{X}^{(m)} \in \mathbb{R}^{B \times d}$, $\mathbf{y}^{(m)} \in \{0,1\}^B$

### 3. Forward Pass (Batch $m$)

Pre-activation:

$$
\mathbf{z}^{(m)} = \mathbf{X}^{(m)} \mathbf{w} + b\mathbf{1}_B \in \mathbb{R}^B
$$

Sigmoid activation:

$$
\hat{\mathbf{y}}^{(m)} = \sigma(\mathbf{z}^{(m)}) = \frac{1}{1 + \exp(-\mathbf{z}^{(m)})}
$$

### 4. Binary Cross-Entropy Loss

Average loss over batch:

$$
\mathcal{L}^{(m)} = -\frac{1}{B} \sum_{i=1}^B \left[ y_i^{(m)} \log(\hat{y}_i^{(m)}) + (1-y_i^{(m)}) \log(1-\hat{y}_i^{(m)}) \right]
$$

### 5. Gradient Computation

Weight gradient:

$$
\nabla_{\mathbf{w}}^{(m)} = \frac{1}{B} (\mathbf{X}^{(m)})^\top (\hat{\mathbf{y}}^{(m)} - \mathbf{y}^{(m)}) \in \mathbb{R}^d
$$

Bias gradient:

$$
\nabla_b^{(m)} = \frac{1}{B} \mathbf{1}_B^\top (\hat{\mathbf{y}}^{(m)} - \mathbf{y}^{(m)}) \in \mathbb{R}
$$

### 6. Parameter Update

With learning rate $\eta$:

$$
\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla_{\mathbf{w}}^{(m)}
$$

$$
b \leftarrow b - \eta \nabla_b^{(m)}
$$

---

## Synthetic Dataset

Generated 2D linearly separable data:
- 300 samples (150 per class)
- Class 0: $\mathcal{N}([1,1]^\top, 0.5\mathbf{I})$
- Class 1: $\mathcal{N}([4,4]^\top, 0.5\mathbf{I})$
- Visual decision boundaries possible

---

## References

- Goodfellow et al. (2016) *Deep Learning*  
- Bishop (2006) *Pattern Recognition and Machine Learning*