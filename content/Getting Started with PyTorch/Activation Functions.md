In the previous example, our delivery model used a single neuron to learn the relationship between **distance** and **delivery time**. This worked when the data roughly followed a straight line, but it fails when the company expanded its delivery service to longer routes using cars (the blue dots on the graph below).

![[non_linear_pattern.png|514]]

The reason is that ==a linear model assumes the relationship stays linear.== In reality, the delivery time behaves differently not following a linear function. ==We need then a model that can learn nonlinear patterns.==
# Why More Neurons Aren't Enough

You might think that adding more neurons would solve the problem. However, simply stacking linear neurons does **not** make the model nonlinear.

![[multiperceptron.png|444]]

Each neuron performs a linear transformation, and so even though we now have multiple neurons and layers, ==combining linear operations still results in a linear function==. The model would still only be able to produce a straight-line relationship.

Therefore, to learn curves and more complex patterns, we need to introduce **nonlinearity**.
# Activation

==An activation function is a function applied to the output of a neuron that introduces nonlinearity.== A neuron first performs its linear transformation; then the result is passed through an activation function.

![[activation_function_nn.png|415]]

This small additional step allows neural networks to learn much more complex relationships.
Activation functions are usually applied **element-wise**, meaning each neuron's output is passed through the function independently.
## ReLU

One of the most commonly used activation functions is **ReLU (Rectified Linear Unit)**.
It is defined as:

```text
ReLU(x) = max(0, x)
```

In other words:
- If the input is negative → output `0`
- If the input is positive → output the input unchanged

```text
x < 0  →  0
x ≥ 0  →  x
```

In PyTorch:
```python
# Define the model
model = nn.Sequential(
	nn.Linear(1, 1),
	nn.ReLU()
)
```

The ReLU activation function is commonly used in hidden layers in real word applications.

---

With ReLU:
```text
y = ReLU(wx + b)
```

the output becomes zero whenever `wx + b` is negative and follows the original line when it is positive.
This creates a **bend** in the function, allowing the model to represent a nonlinear relationship.

The bend occurs where:
```text
wx + b = 0
```
so:
```text
x = -b / w
```

This point determines where the neuron switches from producing `0` to responding to the input.

One ReLU neuron can create one basic bend, but real-world relationships can be much more complicated.
Each neuron has its own **weight and bias**, so each neuron can activate at a different point and contribute differently to the final prediction.

![[activation_function_result.png|483]]

### Non Linear Model

A simple network for this problem could look like:

```python
model = nn.Sequential(
    nn.Linear(1, 3),
    nn.ReLU(),
    nn.Linear(3, 1)
)
```

The first linear layer:
```python
nn.Linear(1, 3)
```
means:
- `1` input feature → distance
- `3` neurons → three different learned transformations

The three outputs then pass through:
```python
nn.ReLU()
```

The second linear layer:
```python
nn.Linear(3, 1)
```

takes those three transformed values and combines them into **one final output**: the predicted delivery time.

## Sigmoid

- **Sigmoid** → maps values between `0` and `1`, making it useful when an output represents a probability.

## Tanh

- **Tanh** → maps values between `-1` and `1`.