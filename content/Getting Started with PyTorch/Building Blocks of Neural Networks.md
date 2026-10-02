Neural networks are useful for solving problems where we have data containing patterns that we want the model to learn. While they are inspired by biology, **neurons in this context are just mathematical units**. ==They're tools with [[Hyperparameter Optimization|adjustable parameters]] that shift to match the patterns in your data.==

Let's tackle a delivery problem where a delivery driver is assigned an order based on the distance they need to travel. Here we can see a pattern in our historical delivery data:

![[delivery_data.png|645]]

We can see that as the distance increases, the delivery time also tends to increase, and the points roughly follow a straight line. A good predictive model for this data would therefore be a **line**. If you know the equation for that line, you can use it to predict new values.
## Neuron
==A single neuron can be represented as a linear equation with two parameters: the weight and the bias.==

![[neuron.png|399]]

==The weight controls the slope of the line, while the bias shifts the line up or down.==

So, all the neuron needs to do is find the right values for `w` and `b` to create the best-fitting line through the data. The process of finding these values is essentially the **learning** part of machine learning.
## How Does a Neuron Learn?

To find good values for the parameters, PyTorch starts randomly initialized values for the weight and bias. The neuron then uses these parameters to make predictions and compares those predictions with the actual values in the training data. The difference between the predictions and the actual values is measured using a **[[Loss Function|loss function]]**. The further the predictions are from the actual values, the larger the loss will generally be.

The network then uses **[[Backpropagation|calculus]]**, specifically gradients, to determine how the parameters should change to reduce the loss. It is essentially asking:
 
> If I increase the weight slightly, does the error go up or down?

Based on this information, the model takes a small step in the direction that reduces the error, calculates the loss again, and repeats the process. This happens many times during training. Over time, the weight and bias are adjusted so that the model's predictions become increasingly accurate.

```text
Initial parameters
       ↓
Make predictions
       ↓
Calculate loss
       ↓
Calculate gradients
       ↓
Update parameters
       ↓
Repeat
```

In PyTorch, this entire process can be implemented with relatively few lines of code.
## From One Neuron to a Neural Network

But what happens when you connect thousands or even millions of neurons together? Does the mathematics become impossibly complex?
The basic calculation performed by each neuron remains the same. A neuron can also accept **multiple inputs**, with each input having its own weight.

For example, instead of predicting delivery time using only distance, we could use:
- distance
- time of day
- weather

The equation would then look like:
```text
y = w₁x₁ + w₂x₂ + w₃x₃ + b
```

Each input gets its own weight, the weighted inputs are added together, and then the single bias is added. ==So even with multiple inputs, the neuron is still performing the same basic linear calculation.==

![[neural_network.png|294]]
## Layers

==A layer is simply a group of neurons that all receive the same inputs.==
When we connect the outputs of one layer to the inputs of another layer, we create a **neural network**.

A simple neural network can be thought of as:
```text
Input layer → Hidden layers → Output layer
```

The **input layer** receives the raw data, such as distance, time of day, and weather.

The **hidden layers** contain neurons that transform the information as it passes through the network. They are called "hidden" because their values are not directly part of the input or final output that we specify.

The **output layer** produces the final prediction, such as the estimated delivery time.

The important thing to understand is that a neural network is built from relatively simple mathematical operations.
By connecting many neurons and layers together, neural networks can learn much more complex relationships from data.

Look next: [[Machine Learning Pipelines]]