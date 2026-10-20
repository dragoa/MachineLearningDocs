TorchVision offers ready-made datasets, utilities for loading custom datasets, and synthetic datasets for testing:

- **Pre-built datasets**: usually come with labels and plug directly into PyTorch’s training pipeline. They let us compare models with established benchmarks like MNIST and ImageNet, and they save the time of collecting and labeling data.
- **Custom datasets**: utilities such as `ImageFolder` support loading our own labeled image folders; other data layouts may require a custom `Dataset`
- **Fake datasets**: synthetic data to debug pipelines and check that the shape of the data is consistent with the architecture
### Pre-built Datasets

The official documentation lists the datasets organized by the type of problem they solve:

- **Image classification** → assigns labels to images
- **Object detection** → identifies and localizes objects within an image
- **Segmentation** → assigns a class to each pixel; instance segmentation also distinguishes separate objects of the same class
- and many more

Some popular ones:

| Dataset           | Description                                                                                                              |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Fashion MNIST** | clothing items like shirts, shoes and bags                                                                               |
| **SVHN**          | house numbers from Google Street View                                                                                    |
| **CelebA**        | celebrity portraits annotated with 40 attributes (smiling, wearing glasses...), plus bounding boxes and facial landmarks |
| **COCO**          | annotations for over 300,000 images for object detection, segmentation and image captioning                              |
| **PascalVOC**     | about 3,000 images for object detection and segmentation, over 20 classes                                                |
| **ImageNet**      | over 1 million high-resolution images across 1,000 categories                                                            |
> [!warning] ImageNet
> ImageNet has long been the gold standard for training deep networks and testing state-of-the-art models, but it requires **manual download** from their website because of its licensing policy.
#### Loading CIFAR-10

CIFAR contains `32 × 32` color images. CIFAR-10 and CIFAR-100 have 10 and 100 classes respectively.

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

cifar_dataset = datasets.CIFAR10(root='./data',
                                 train=False,
                                 download=True,
                                 transform=transform)

cifar_loader = DataLoader(cifar_dataset, batch_size=8, shuffle=True)
```

- `root` → directory where the images are downloaded
- `train=False` → downloads the **test set** (otherwise it uses the training set)
- `download=True` → downloads from the internet, unless it has already been downloaded
- `transform` → the transforms to apply, if we already know which ones
- `DataLoader` → handles **batching, shuffling and parallel data loading**

Here the batches have `8` images and are shuffled. Test sets are normally **not shuffled**: it is done here only so that a new image is displayed each time.
#### Loading EMNIST

Each dataset is unique and the loading method may vary, so it is crucial to **read the documentation** of each dataset.

The EMNIST `digits` split contains `280,000` 28 × 28 grayscale images of handwritten digits (240,000 for training and 40,000 for testing). It requires an extra parameter compared with CIFAR-10: EMNIST comprises **six different splits**, each with different images and labels, so we must specify which one to load.

```python
emnist_transformation = transforms.Compose([
	transforms.RandomRotation(degrees=(90, 90)),
	transforms.RandomVerticalFlip(p=1.0),
	transforms.ToTensor(),
	transforms.Normalize((0.5,), (0.5,)) # The mean and std must be in a tuple
])

emnist_dataset = datasets.EMNIST(root='./data',
                                 split='digits',
                                 train=False,
                                 download=True,
                                 transform=emnist_transformation)

emnist_loader = DataLoader(emnist_dataset, batch_size=8, shuffle=True)
```

The transformation pipeline rotates the image by 90 degrees, applies a vertical flip, and normalizes with mean and std of `0.5`. These details come from the **dataset documentation**.
### Custom Datasets

Before loading, the data must be organized in **folders by class**:

```text
root/
├── class_a/
│   ├── img1.jpg
│   └── img2.jpg
└── class_b/
    ├── img3.jpg
    └── img4.jpg
```

1. main **root folder**
2. one **folder per class**
3. images placed in their respective folders

Then we use `ImageFolder`, a generic dataset that **infers the class labels from the folder names**. This makes it easy to load custom datasets and apply transformations.

```python
from torchvision.datasets import ImageFolder

custom_dataset = ImageFolder(root='./my_data', transform=base_transform)
custom_loader = DataLoader(custom_dataset, batch_size=8, shuffle=True)
```

Once loaded, it is a good idea to **display the images** to validate them.
#### Train / test split

There are two main ways to split a custom dataset:

- load the entire dataset and **split it programmatically**, with functions like `random_split()`
- organize the data into **two separate root folders** (train and test) and load them separately

```python
from torch.utils.data import random_split

train_set, test_set = random_split(custom_dataset, [0.8, 0.2])
```
### Fake Datasets

Fake datasets are useful to quickly test or debug a pipeline, for example to check that the images have the **right size** for our model.

`FakeData` generates **random images and labels**, mimicking the shape and size of a real dataset. It is ideal for testing without worrying about data availability.

```python
fake_dataset = datasets.FakeData(size=100,
                                 image_size=(3, 224, 224),
                                 num_classes=10,
                                 transform=transforms.ToTensor())
```
