# Perceptron from Scratch — AND Gate Example

## What is a Perceptron?

A Perceptron is the simplest type of artificial neuron used for binary classification.  
It takes an input vector, computes a weighted sum plus bias, applies an activation function, and outputs a prediction between 0 and 1.

---

## Implementation Details

This implementation uses:

- **Sigmoid activation function:**
  $$
  \[
  \hat{y} = \frac{1}{1 + e^{-z}}
  \]
  $$
  which outputs probabilities instead of hard binary values.

- **Cross-entropy loss:**  
  Used to measure how well the predictions match the true labels.

- **Gradient descent:**  
  To update weights and bias by computing gradients of the loss.

---

## How It Works

1. **Input:**  
   A vector \( x = [x_1, x_2, \ldots, x_n] \), for example for the AND gate:  
   \[
   [0, 0], [0, 1], [1, 0], [1, 1]
   \]

2. **Weighted sum:**  
   \[
   z = \mathbf{w} \cdot \mathbf{x} + b
   \]

3. **Prediction:**  
   Sigmoid applied to \( z \) produces \( \hat{y} \in (0, 1) \).

4. **Loss calculation:**  
   Cross-entropy loss compares \( \hat{y} \) with true label \( y \).

5. **Gradient computation:**  
   Calculate derivatives of loss with respect to weights and bias.

6. **Update:**  
   Adjust weights and bias to minimize the loss.

7. **Repeat:**  
   Iterate over the training data for multiple epochs.

---

## Training on the AND Gate

The perceptron is trained on the 4 input-output pairs of the logical AND function:

| Input  | Output |
|--------|--------|
| [0, 0] | 0      |
| [0, 1] | 0      |
| [1, 0] | 0      |
| [1, 1] | 1      |

The goal is to learn to output 1 only when both inputs are 1.

---

## Usage

- The `Perceptron` class has `predict` and `train` methods.  
- The `train` method prints accuracy and loss after each epoch.  
- The number of epochs and learning rate can be adjusted.

---

## References

- Rosenblatt, F. (1958). *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain.*  
- Nielsen, M. (Online Book). *Neural Networks and Deep Learning.*  
