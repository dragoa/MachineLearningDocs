Over the years the research community has developed many deep learning architectures that excel at image tasks, such as VGG, ResNet and GoogLeNet. TorchVision offers a comprehensive selection of them through its **Models** module. They can be:

- **trained from scratch**
- loaded with **pre-trained weights** (many are trained on large datasets like ImageNet), to adapt them to our task with **transfer learning** or **fine-tuning**
### Image Classification with ResNet50

Suppose we want to classify an image with **ResNet50**, pre-trained on ImageNet (`1000` classes).
#### 1. Load the model with its weights

```python
import torch
from PIL import Image
from torchvision import models

weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)

model.eval()
```

- The `weights` argument loads the weights trained on ImageNet. `pretrained=True` is the **old API** (see the section on class names below)
- `.eval()` switches the model to **evaluation mode**, so it is ready for inference

> [!note] Evaluation mode and gradient tracking
> `.eval()` changes the behavior of layers such as Dropout and BatchNorm; it does not disable gradient tracking. Use `torch.no_grad()` (or `torch.inference_mode()`) when gradients are not needed.
#### 2. Class labels

The model outputs **numeric indices** for the 1000 classes. For ImageNet models they can be mapped to names with a **JSON file**: loading it gives a dictionary `index → label`. Details in the next section.
#### 3. Preprocess the input

Use the preprocessing transform associated with the selected weights. It applies the expected resizing, cropping, conversion, and normalization, which can vary between weight versions.

```python
preprocess = weights.transforms()

img = Image.open('./images/dog.jpg').convert("RGB")
input_tensor = preprocess(img).unsqueeze(0)
```

`unsqueeze(0)` adds the batch dimension, so the image is processed as a **batch of one**, as the model requires: `[3, 224, 224]` → `[1, 3, 224, 224]`.
#### 4. Inference

The model returns raw scores called **logits**, one for each of the 1000 classes. **Softmax** converts them into probabilities: each value is between `0` and `1` and they all **sum to 1**.

To get the **top 5 guesses**:
1. get the top 5 class indices and their probabilities
2. map each index to its name with the ImageNet classes dictionary

```python
with torch.no_grad():
    logits = model(input_tensor)

probs = torch.softmax(logits, dim=1)
top5_prob, top5_idx = torch.topk(probs, 5)

categories = weights.meta["categories"]
for p, idx in zip(top5_prob[0], top5_idx[0]):
    print(f"{categories[idx.item()]}: {p.item():.2%}")
```

In the example the model correctly identified a **golden retriever** with high confidence.

> [!note] Softmax is not calibration
> Softmax converts logits into scores that sum to `1`, but those scores are not necessarily well-calibrated probabilities. Treat them as relative model scores unless calibration has been evaluated.
### Getting the Class Names

When working with pre-trained models, it is crucial to know **which classes they recognize**. For example, before using the model for a task we must confirm that `golden_retriever` is a recognizable category.

There are two ways:
- **Manual approach** → usually for older models
- **Modern approach** → class names retrieved from the **weights metadata**
#### Manual approach

Used for older models loaded with the `pretrained=True` flag, which loads a default set of ImageNet weights.

To find the **number of classes** we can:
- print the whole model and look at the final classification layer (`1000` outputs)
- access it directly in the code

```python
print(model.fc)               # Linear(in_features=2048, out_features=1000, ...)
print(model.fc.out_features)  # 1000
```

This tells us there are 1000 classes, but **not their names**. For those we need external sources such as the PyTorch documentation or helper files. One solution is to download a **JSON file** that maps indices to labels. For example, index `207` corresponds to `golden retriever`.
#### Modern approach: weights metadata

A more direct approach is to inspect the **metadata of the weights object**, which includes the category names for supported classification weights.

Example with **DeepLabV3_ResNet50**, a model for **image segmentation**:

1. select the architecture's weights enum
2. read the category names from the selected weights' metadata

```python
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

weights = DeepLabV3_ResNet50_Weights.DEFAULT
categories = weights.meta["categories"]

print(len(categories))   # 21
print(categories)
```

These weights correspond to **21 segmentation classes**, including a **`dog`** category, which will be used in the lab for segmentation.
