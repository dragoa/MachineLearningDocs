![[ml_pipeline.png]]

When working with machine learning, datasets can contain thousands or even millions of samples. ==Loading all this data into memory at once can use too much RAM.==

For example, imagine loading 100,000 delivery records:
```python
all_distances = []
all_times = []

for i in range(100000):
    distance, time = load_delivery_record(i)
    all_distances.append(distance)
    all_times.append(time)
```

This approach stores every sample in memory before the model can use it.

==The practical solution is to work with batches, smaller manageable chunks of data.==

![[batch_samples.png]]

PyTorch provides three important tools to make this possible:

- [[PyTorch Transforms|Transforms]]: prepare each sample for the model.
- [[PyTorch Dataset|Dataset]]: retrieves individual samples.
- [[PyTorch DataLoader|DataLoader]]: serves the data in batches.

The principle is simple: ==Load only what you need, when you need it.==
# Transforms

[[PyTorch Transforms|Transforms]] are operations that run on each data point as it's loaded, preparing the data for the model.

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])
```

`Compose` simply means applying the following transforms in order.

The two most common transforms are `ToTensor()` and `Normalize()`.
## ToTensor()

Neural networks are surprisingly picky. They generally train better when their inputs are small numbers, often between zero and one for image data.

`ToTensor()` converts image data into [[Tensor|PyTorch tensors]] and scales typical 8-bit image pixel values from 0–255 to 0–1.

For example, consider these values:
```python
[2, 4, 6, 8, 10]
```

Divide each value by the maximum value, 10:
```python
tensor([0.2, 0.4, 0.6, 0.8, 1.0])
```

The values are now between zero and one.
## Normalize()

`Normalize()` adjusts these values even further, ==centering them around zero and scaling them using the standard deviation.==

The formula is:
$$
x_{\text{normalized}} = \frac{x - \mu}{\sigma}
$$
Where:
- $x$ is the original value.
- $\mu$ is the [[Mean|mean]].
- $\sigma$ is the [[Standard Deviation|standard deviation]].

For example, normalizing the values using their mean and standard deviation gives approximately:
```python
tensor([-1.41, -0.71, 0.00, 0.71, 1.41])
```

These transformations help the model train more effectively. The mean and standard deviation should be chosen based on the dataset, usually calculated separately for each image channel.

> [!note] Avoid data leakage
> Calculate normalization statistics from the training set only, then reuse those same values for validation, test, and inference data. Computing them from the full dataset gives the model information about held-out data.
# Dataset

A [[PyTorch Dataset|Dataset]] object handles the data and knows how to retrieve individual samples.

It takes care of:
- Where the data lives on disk.
- How to load a specific sample.
- How many samples are available.
- How to apply transforms as each sample is loaded.

PyTorch provides many pre-built datasets, so you don't always need to create your own.

```python
dataset = SomeDataset(
    './data',
    train=True,
    download=True,
    transform=transform
)
```

- `./data` indicates where the data is stored.
- `train=True` selects the training set, when supported by the dataset.
- `download=True` downloads the data if it isn't already available.
- `transform=transform` applies the transforms to each sample.

You can retrieve a sample simply by indexing the dataset:

```python
first_item = dataset[0]
```

The dataset loads that specific sample and applies the transforms.
==Instead of loading everything into memory at once, the dataset retrieves samples when they're requested.== This makes it possible to work with much larger datasets.
# DataLoader

Once you've defined your dataset, you can use a [[PyTorch DataLoader|DataLoader]] to serve the data in batches.

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)
```

Two important parameters are:

- `batch_size=32` tells the loader how many samples to serve at a time.
- `shuffle=True` shuffles the sample order, helping the model learn more effectively during training.

==Instead of sending the entire dataset to the model, the `DataLoader` requests one batch at a time from the dataset.==
This makes training on large datasets much more manageable.
# Complete PyTorch Data Pipeline

==The complete pipeline combines transforms, a dataset and a data loader.==

```python
# 1. Define the transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])

# 2. Create the dataset
train_dataset = YourDataset(
    './data',
    train=True,
    transform=transform
)

# 3. Create the DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)
```

Once the pipeline is ready, you can start training the model.

```python
for batch_idx, (data, labels) in enumerate(train_loader):

    # Data arrives in batches, already transformed
    output = model(data)
```

==Each batch contains the input data and its corresponding labels. The model processes these samples together rather than processing the entire dataset at once.==

During training, you would also calculate the loss, compute gradients and update the model's parameters.

==This pipeline makes it possible to train on datasets that would otherwise overwhelm your computer's memory.== Whether you're working with thousands of delivery records or millions of images, the principle stays the same: load only what you need, when you need it.

Getting data into the model is just the beginning. Next, you'll look at the [[PyTorch Model Pipeline]], where you can see how to create a model in PyTorch, in order to create the base to then train it.