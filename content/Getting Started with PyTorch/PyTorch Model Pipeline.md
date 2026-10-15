![[ml_pipeline.png]]

Previously we have seen the [[PyTorch Data Pipeline]] where you load, prepare, and serve data efficiently for then giving it to a ML model.
Now it's time to look at the rest of the Machine Learning pipeline: **model building, training, and evaluation.**

In PyTorch, you can build a model using `nn.Sequential`, which allows you to stack layers in order:
```python
model = nn.Sequential(
    nn.Linear(1, 20), # Linear Layer with 1 input and 20 outputs
    nn.ReLU(),
    nn.Linear(20, 1) # Linear Layer with 20 inputs and 1 output
)
```

This works great for simple models, but there's another pattern you'll see everywhere in PyTorch: creating a model using the [[PyTorch Module|nn.Module]].

```python
class ExampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(1, 20)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        return x
```

This model does exactly the same thing as the `nn.Sequential` version, but gives you more control over how the model works.

==Every custom PyTorch module has two important parts:==

- `__init__()` defines the layers that exist in the model.
- `forward()` describes how data flows through those layers.

In `__init__()`, you're defining which tools your model will use. The actual order comes in `forward()`, where you describe the path the data takes through the network.

==`__init__()` defines the layers, while `forward()` defines how the data moves through them.==
# Calling the Model

Once you've defined your model, you can create an instance and pass data through it:

```python
model = ExampleModel()
output = model(data)
```

You might think you should call `model.forward(data)` directly, but that's not the recommended way.

```python
# Don't call forward directly
output = model.forward(data)

# Call the model instead
output = model(data)
```

Why? When you call `model(data)`, PyTorch does more than just run your `forward()` method. ==It also performs internal checks, tracks the necessary mathematical operations, and sets things up for updating the model during training.==

==Always call the model directly with the input. PyTorch will handle the `forward()` call for you.==

This applies to both custom models built with `nn.Module` and models created using `nn.Sequential`, since `nn.Sequential` is also a subclass of `nn.Module`.
# Why `super().__init__()` Matters

Inside your custom model, you'll often see this line:

```python
super().__init__()
```

It might look like standard Python boilerplate, but it's essential.

PyTorch needs to set up a system to track the model's layers and its learnable parameters, including **weights and biases**. These are the values that will be updated during training.

Calling `super().__init__()` initializes the underlying `nn.Module` system so that PyTorch can register and track these components.
Without it, you'll encounter an error when trying to assign layers:

```python
class ExampleModel(nn.Module):
    def __init__(self):
        # Missing super().__init__()
        self.fc1 = nn.Linear(784, 128)
```

This can raise an error such as:

```text
AttributeError: cannot assign module before Module.__init__() call
```

==Always call `super().__init__()` before defining your layers==, otherwise, PyTorch cannot properly register the model's components.
# Model Training

Once you've built your model and prepared your data, you can start training.
A typical PyTorch training loop follows this pattern:

```python
for epoch in range(epochs):
    for X, y in train_loader:
        optimizer.zero_grad()

        outputs = model(X)
        loss = loss_function(outputs, y)

        loss.backward()
        optimizer.step()
```

Each step has a specific purpose:

- `optimizer.zero_grad()` clears the gradients left over from previous batches.
- `model(X)` passes the input through the model and produces predictions.
- `loss_function(outputs, y)` measures how far the predictions are from the correct answers.
- `loss.backward()` calculates the gradients needed to update the model's parameters.
- `optimizer.step()` updates the parameters using those gradients.

The order of these operations is important.
==The standard training sequence is: clear gradients, calculate the loss, compute gradients, and update the model.==

If you change this order, your model might not learn properly even if the code runs without errors.

For example:

- If you call `optimizer.step()` before `loss.backward()`, the optimizer won't have the current batch's gradients available.
- If you call `optimizer.zero_grad()` after `loss.backward()`, you erase the gradients before the optimizer can use them.
- If you call `optimizer.zero_grad()` only once outside the loop, gradients accumulate across batches instead of being cleared at each iteration.

This is why the same sequence is repeated for every batch.
# Evaluation

After training, you need to know whether your model has actually learned something useful.
[[Model Evaluation|Evaluation]] means testing the model on new data that it hasn't been trained on.

Testing only on the training data isn't enough. A model might memorize the examples it has already seen without learning patterns that generalize to new data.
For image classification, one simple way to measure performance is [[Accuracy|accuracy]].
## `model.eval()`

Before evaluating your model, call:

```python
model.eval()
```

Be careful: `model.eval()` does not evaluate the model's performance. ==It switches the model into evaluation mode.==

Some layers, such as dropout and batch normalization, behave differently during training and evaluation. Calling `model.eval()` tells these layers to use their evaluation behavior.
## `torch.no_grad()`

During evaluation, you usually don't need to calculate gradients because you're not updating the model's parameters.
You can disable gradient tracking using:

```python
with torch.no_grad():
    outputs = model(images)
```

This reduces memory usage and avoids unnecessary gradient calculations.
Without it, PyTorch may continue tracking operations that aren't needed for evaluation.

==`model.eval()` changes how certain layers behave, while `torch.no_grad()` disables gradient tracking. They do different things, and you often use them together.==
## Calculating Accuracy

For image classification, accuracy measures the percentage of predictions that are correct.
The formula is:
$$Accuracy=Correct PredictionsTotal Predictions×100$$
For example, if your model correctly classifies 9,500 digits out of 10,000, its accuracy is 95%.

```python
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f'Accuracy: {accuracy}%')
```

Here's what happens:

- `model(images)` generates predictions for the images.
- `torch.max(outputs, 1)` selects the class with the highest score for each image.
- `(predicted == labels)` checks which predictions are correct.
- `correct` counts the correct predictions.
- `total` counts the number of evaluated samples.

Finally, the accuracy is calculated by dividing the number of correct predictions by the total number of samples and multiplying by 100.
==Evaluation tells you how well your model performs on data it hasn't seen during training.==

After evaluation, if you want to continue training, switch the model back to training mode:

```
model.train()
```

This restores the training behavior of layers such as dropout and batch normalization.

Next, you'll take a closer look at [[Loss Function]] and [[Omptimizer & Gradients]] to understand how neural networks measure errors and update their parameters.