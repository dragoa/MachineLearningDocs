Once the model has been trained, we can use it to make predictions about data it has never seen before. This process is called [[Inference]].
A typical PyTorch / machine learning project follows **6 main stages**:

![[ml_pipeline.png]]
## 1. Data Ingestion

Before training a model, we need to gather our raw information and organize it so that PyTorch can work with each data point efficiently. 
But real-world data is often messy. For example, older records might store delivery times as text such as `"22 minutes"`, while newer records might store them as numerical values such as `22.0`. We might also have missing values, negative delivery times, duplicate records, or impossible values such as a delivery driver supposedly traveling at 200 miles per hour.

This kind of messy data is the **norm rather than the exception** in real-world machine learning projects.
## 2. Data Preparation

Collecting the data is only the beginning. We then need to **clean, transform, and organize the data into a form that our model can learn from**.

For the delivery data, this could include removing impossible delivery times, removing duplicate entries, and handling missing values when information such as timestamps wasn't recorded.

We might also need to perform **feature engineering**, where we ==create useful features from the raw data.==

> **Many machine learning problems are caused by poor or messy data rather than incorrect model mathematics.**
## 3. Model Building

Once the data is cleaned and ready, we can design the model that will learn from it.
This stage is about choosing the appropriate model architecture for the problem.

[[Architectural Hyperparameters|Architecture]] refers to the structure of a neural network, including things such as:

- How many neurons it contains
- How those neurons are connected
- How many layers it has
- What types of layers it uses

The architecture is essentially the **blueprint** for the model, but defining the architecture doesn't mean that the model has learned anything yet.

A simple neuron with one input and one output, is built in PyTorch using a linear layer:
```python
model = nn.Sequential(nn.Linear(1, 1))
```
## 4. Training
After building the model, we need to **train** it.
==Training is where the model actually learns the patterns in the data.==

The model makes predictions based on its current parameters and compares those predictions with the actual values.
During training, we need several components to control the learning process:

- A way to **measure prediction errors** → the [[Loss Function]] 
- A way to determine **how the model should improve** → [[Gradient|gradients]]
- A method for **updating the model's parameters** → the [[Hyperparameter Optimization|Optimizer]]
- Training settings that control how the model learns, such as the [[Learning Rate]]

These components work together inside the **training loop**:
```text
Input → Prediction → Loss → Gradients → Parameter update → Repeat
```

PyTorch handles the mathematical computations involved in this process, allowing the model's parameters to gradually improve as it sees more training examples. 
## 5. Evaluation & Debugging

After training, we need to check how well the model performs on **unseen data**.
==A model could perform very well on the data it trained on but perform poorly when given new data.== Therefore, we need to evaluate the model using **unseen data**.

A common approach is to split our dataset into different parts, such as a **training set** and a **test set**. The training set is used to teach the model, while the test set is held back and used later to measure how well the trained model performs on data it hasn't seen before.

> Does your model work well enough for you to trust it?
## 6. Deployment

The final stage is **deployment**. This means taking the trained model and putting it into a real-world environment where people or other software can actually use it.

For example, our delivery model could eventually be integrated into a company's delivery system. When a new order comes in, the system could provide the distance to the model, and the model could return an estimated delivery time.

# Building a Simple Neural Network

Now we can map the machine learning pipeline to actual PyTorch code. In this simple example, **data ingestion and data preparation are combined** because the data is already clean and ready to use.

First, we import the libraries we need:
```python
import torch
import torch.nn as nn
import torch.optim as optim
```

- `torch` → PyTorch's core functionality, including tensors.
- `torch.nn` → tools for building neural networks and layers. 
- `torch.optim` → optimization algorithms used during training.

### 1. Define the data

```python
# Distance in miles
distances = torch.tensor([[1.0],[2.0],[3.0],[4.0]], dtype=torch.float32)
# Delivery times in minutes
times = torch.tensor([[6.96],[12.11],[16.77],[22.21]], dtype=torch.float32)
```

These are [[Tensor|tensors]]. Tensors are PyTorch's main data structure for storing and performing mathematical operations on data, and they are optimized for the computations used in neural networks.

The data has the shape: [4, 1]
This means we have **4 samples**, with **1 feature per sample**.

```text
[
    [1.0],  ← sample 1
    [2.0],  ← sample 2
    [3.0],  ← sample 3
    [4.0]   ← sample 4
]
```

==The outer brackets represent the collection of samples, while each inner set of brackets represents an individual sample.==

If we had multiple features, for example distance, time of day, and temperature, one sample could look like:

```text
[7.0, 18.0, 25.0]
```

So the brackets help PyTorch understand the difference between **samples** and **features**.

`dtype=torch.float32` tells PyTorch to store the values as **32-bit floating-point numbers**, which are commonly used for neural network computations.

### 2. Build the Model

Now we define our neural network:
```python
model = nn.Sequential(nn.Linear(1, 1))
```

`nn.Sequential` is a container that passes data through a sequence of layers in order. In this example we have only one layer (one neuron).
The 2 numbers means 1 input and 1 output, where `nn.Linear` performs a linear transformation:

```text
output = weight × input + bias
```

The weight and bias are automatically created as the model's **parameters**. During training, PyTorch will adjust these parameters so that the model learns the relationship between distance and delivery time.

### 3. Loss Function and Optimizer

The model needs two important tools to learn:
```python
# Define the loss function and optimizer
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

`nn.MSELoss()` stands for **Mean Squared Error Loss**.
The loss function measures how far the model's predictions are from the actual values.
==The goal during training is to minimize the loss.==

`optim.SGD` stands for **[[SGD|Stochastic Gradient Descent]]**.
The optimizer uses the gradients calculated during [[Backpropagation|backpropagation]] to determine how the model's parameters should be adjusted to reduce the loss.

`model.parameters()` gives the optimizer access to the model's learnable parameters.

The `lr` parameter is the **[[Learning Rate|learning rate]]** and it controls how large each parameter update is.

- Smaller learning rate → smaller updates.    
- Larger learning rate → larger updates.

The learning rate needs to be chosen carefully because updates that are too small can make training slow, while updates that are too large can make training unstable.

## 4. Training Loop

This is where the actual learning happens.
```python
# Training loop
for epoch in range(500):
	# 0. Reset the optimizer
	optimizer.zero_grad()
	# 1. Make predictions
	outputs = model(distances)
	# 2. Calculate the loss - how bad was this guess?
	loss = loss_function(outputs, times)
	# 3. Calculate adjustments
	loss.backward()
	# 4. Update the model
	optimizer.step()
```

==An epoch is one complete pass through the training data.== In this case we look at the data 500 times.

`optimizer.zero_grad()`: Gradients are accumulated by default in PyTorch, so we need to clear the gradients from the previous training step before calculating new ones.

`model(distances)`: The model receives the distances as input and produces its predictions.

`loss_function(outputs, times)`: Then the predictions are compared with the actual delivery times using a loss function. This gives us a numerical measure of how wrong the model currently is.

`loss.backward()`: ==This calculates the gradients of the loss with respect to the model's parameters. This process is called backpropagation.==

The gradients tell the optimizer which direction the weight and bias should move to reduce the loss.

`optimizer.step()`: The optimizer uses those gradients to update the model's parameters.
After many epochs, the weight and bias should have moved toward values that produce better predictions.

### 5. Inference

Once training is finished, we can use the model to make predictions on new data.

```python
with torch.no_grad():
	test_distance = torch.tensor([[25.0]], dtype=torch.float32)
	predicted_time = model(test_distance)
	print(f"Predicted time for 25 miles: {predicted_time.item():.1f} minutes")
```

`torch.no_grad()` tells PyTorch that we're **not training the model**, we're performing [[Inference]].

During training, PyTorch needs to track operations so that it can calculate gradients. During inference, we don't need those gradients, so `torch.no_grad()` avoids that extra work and makes inference more efficient.

