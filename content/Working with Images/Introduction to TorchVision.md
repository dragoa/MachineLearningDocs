## Why TorchVision?

Deep learning for computer vision comes with a lot of moving parts:

- Image loading
- Data augmentation
- Normalization
- Batching
- Model definitions
- Visual debugging

Writing all of this from scratch is time-consuming, error-prone and hard to maintain. TorchVision packages these common functionalities into a single library.

What we gain by using it:
- **Consistency**: it integrates with PyTorch's `Dataset` and `DataLoader`
- **Efficiency**: many transforms are optimized in C or TorchScript
- **Reliability**: standardized code means fewer bugs and better reproducibility

In this way we can focus on our own computer vision problem instead of reinventing the fundamental tools.

TorchVision provides tools in four key areas:
![[torchvision.png]]
## Transforms

==Transforms are preprocessing and augmentation operations applied to images before feeding them into a model. They include resizing, cropping, rotating and many more.==

TorchVision provides ready-made functions for all of these, and also allows us to create **custom transformations**.

![[transforms.png|300]]

### ToTensor()

Converts an image from **PIL format** to a **PyTorch tensor**. PIL stands for Pillow, a common Python library for image handling.
<p align="center">
  <img src="toTensor.png" width="500">
</p>
It also **scales pixel values from the `0-255` range to the `0-1` range**, which helps stabilize training by keeping the inputs consistent.

The resulting tensor is in **CHW** format (channels → height → width). NumPy and PIL use **HWC** (height → width → channels).

```python
import torchvision.transforms as transforms
from PIL import Image

# Load an image
image = Image.open('./images/mangoes.jpg')
# Dimensions of the original PIL image
print("Original PIL Image dimensions:", image.size)

# Convert the PIL Image to a PyTorch Tensor
img_tensor = transforms.ToTensor()(image)

# Dimensions (shape) of the tensor
print(f"Dimensions after converting to a Tensor: {img_tensor.shape}")
```

```text
Original PIL Image dimensions: (2048, 2048)
Dimensions after converting to a Tensor: torch.Size([3, 2048, 2048])
```

- The original image is `2048 × 2048`, and each pixel has an RGB tuple
- The tensor has `3` channels (red, green, blue), followed by height and width
### ToPILImage()

If we have manipulated the tensor and want to go back to a PIL image, we use `ToPILImage()`.

```python
# Converting back to PIL
img_pil = transforms.ToPILImage()(img_tensor)
# Dimensions of the converted back PIL image
print("Dimensions after converting back to PIL:", img_pil.size)
```

The size remains `2048 × 2048`. Comparing the original image with the transformed one, there are **no differences**, so no deterioration.
### Resize()

`Resize()` adjusts the image to a specific size, ensuring **consistent input dimensions** for the model.

```python
# Define the resize transformation (50x50 square)
resize_transform = transforms.Resize(size=50)
# Apply the transformation
resized_image = resize_transform(image)
```

- ==If the image is not square, the **smaller edge** is resized to `50` first, and the other edge is resized proportionally (the aspect ratio is kept)==
- The resized image looks **pixelated** because of the reduced resolution
### CenterCrop()

`CenterCrop()` keeps the **central portion** of the image, removing irrelevant edges.

```python
# Define the center crop transformation (256x256)
center_crop_transform = transforms.CenterCrop(size=256)
# Apply the transformation
cropped_image = center_crop_transform(image)
```

Both `Resize()` and `CenterCrop()` take the **desired output size** as an argument.
<p align="center">
  <img src="centerCrop.png" width="500">
</p>

### Normalize()

After `ToTensor()` has brought the values into the `0-1` range, `Normalize()` normalizes each channel to have **mean 0** and **standard deviation 1**. For each channel:

```text
normalized = (x - mean) / std
```

This is essentially computing the **z-score**.

```python
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

img_normalized = normalize(img_tensor)
```
<p align="center">
  <img src="pixel_distribution.png" width="500">
</p>
Comparing the pixel histograms of each RGB channel before and after:

- **Before** → the distributions may be skewed, so the model may receive inputs that are not balanced across channels
- **After** → the shape of the distribution stays similar, but values are centered around `0` with std `1`

Centering improves learning efficiency, especially with [[Activation Functions|activation functions]] like ReLU or Tanh, and leads to smoother gradient flow, stable optimization steps and faster convergence.
#### Which mean and std to use?

- Ideally, compute them **directly from our own dataset**, as long as it is big enough
- For convenience, developers often use the **pre-computed ImageNet values**, which are the ones in the snippet above
## Compose and Data Augmentation

Suppose we want every image to follow a specific sequence of steps: resize, center crop, convert to tensor, normalize. ==`Compose()` takes a **list of operations and applies them in sequence**.==

```python
# Simple transform to get a clean, unaugmented version of the images
base_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

Any image passed through this pipeline will have the correct **size, format and scale** for models like [[ResNet]]. 
When we pass a pipeline to a TorchVision dataset, the transformation is **automatically applied** to every image entering a batch. Each time the `DataLoader` receives a sample, the pipeline transforms the original image.

```python
# Load the OxfordIIITPet dataset, applying the simple transform to each image
my_dataset = datasets.OxfordIIITPet(root=oxf_pet_data_path,
                                    split='test',
                                    download=oxf_pet_download,
                                    transform=base_transform)
```
### Image Augmentation

A common challenge in computer vision is having a **small dataset**. Even with a good architecture, the data may be too limited to train it effectively.

==One solution is to create **variations of each image**: color jitter, rotation, noise, random crops, random resizing... This is called **image augmentation**.== Each new version effectively becomes a new sample, increasing the **size and variety** of the dataset without collecting new data, and improving **model generalization**.

- **`RandomResizedCrop`** → randomly crops and resizes different parts of the image, encouraging the model to focus on all object areas, not just the center. In the example, the crops were resized to `224 × 224`
- **`RandomHorizontalFlip`** → flips the image horizontally with a default probability of `0.5`. Helps the model learn that objects can appear in either direction
- **`ColorJitter`** → randomly adjusts brightness, contrast and saturation, making the model more resilient to lighting variations. Each parameter is changed by a factor randomly selected in the given range (e.g. `0.5` → between `0` and `0.5`)

```python
# The full augmentation pipeline with all random transformations
full_augmentation_pipeline = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
    transforms.ToTensor(),
    # Using 'mean' and 'std' values as calculated on the 100x100 images
    transforms.Normalize(mean=dataset_mean,
                         std=dataset_std)
])
```

The order matters: the random transforms work on the PIL image, and then we convert to a tensor and apply normalization.
### Custom Transforms

Sometimes we need a transformation that is **not built in**. An example is **salt and pepper noise**, which randomly places white and black pixels in an image. This simulates a grainy photo and helps the model adapt to imperfect real-life images.

To create one, we define a **class** with two methods:

- `__init__` → takes two float parameters:
	- `salt_vs_pepper`: proportion of white vs black pixels (`0.5` = equal amounts)
    - `amount`: total noise amount, i.e. how much of the image is affected
- `__call__` → receives the image and returns the modified one:
    1. copy the original image
    2. **salt**: select random coordinates based on `amount` and set those pixels to white (`255`)
    3. **pepper**: select random coordinates and set those pixels to black (`0`)
    4. return the modified image

```python
class SaltAndPepperNoise:
    def __init__(self, salt_vs_pepper=0.5, amount=0.04):
        self.s_vs_p = salt_vs_pepper
        self.amount = amount

    def __call__(self, image):
        output = np.copy(np.array(image))

        # Salt: white pixels
        num_salt = np.ceil(self.amount * image.size[0] * image.size[1] *                            self.s_vs_p)
        # su riga e colonna tra 0 e img.size, escono num_salt punti 
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.size]
        output[coords[1], coords[0]] = 255

        # Pepper: black pixels
        num_pepper = np.ceil(self.amount * image.size[0] * image.size[1] * (1.0                       - self.s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in                           image.size]
        output[coords[1], coords[0]] = 0

        return Image.fromarray(output)
```

Once the class is defined, we add it to the pipeline with **a single line** inside `Compose()`:

```python
full_augmentation_pipeline = transforms.Compose([
	transforms.andomesizedCrop(size=224),
	transforms.andomorizontallip(p=0.5),
	transforms.Coloritter(brightness=0.05, contrast=0.05),
	SaltndPepperNoise(amount=0.00),
	transforms.ToTensor(),
	# sing 'mean' and 'std' values as calculated on the 00x00 images
	transforms.Normalize(mean=dataset_mean,
	std=dataset_std)
])
```
<p align="center">
  <img src="salt_pepper_noise.png" width="500">
</p>
Salt and pepper is just one noise type. Others to explore:

- **Gaussian noise**: applies random variations to some pixels, commonly simulating degradation found in some datasets
- **Poisson noise**: simulates degradation under low light or low signal conditions

==The noise type should **align with the conditions** the model will face.==
## Utility Functions

Utility functions simplify image management. The first ones we will use are:
#### decode_image()

Loads images such as JPEGs or PNGs and converts them into **[[Tensor|tensors]]**, the format required by deep learning models.
Once loaded, we can inspect the tensor's dimensions and data type:

- ==TorchVision orders dimensions as **channels → height → width** (`C × H × W`). This is different from libraries like Pillow or OpenCV.==
- The data type is `uint8`, since pixel values range from `0` to `255`.

```python
from torchvision.io import decode_image

img = decode_image("image.jpg")
print(img.shape)   # torch.Size([C, H, W])
print(img.dtype)   # torch.uint8
```
#### make_grid()

Arranges a batch of image tensors into a **single grid**. It is very useful for quickly visualizing the content of a batch.

```python
from torchvision import utils

grid = utils.make_grid(images, nrow=3, padding=2, normalize=True)
```

- `nrow=3` → three images per row
- `padding` → adds space between images
- `normalize=True` → scales pixel values to the `0–1` range, which is essential when visualizing raw tensors

![[make_grid.png|700]]
#### save_image()

Exports tensors back into image files such as JPEG or PNG.

```python
utils.save_image(grid, "grid.png")
```

Look also at: [[TorchVision Utility Functions]]
## Datasets

TorchVision includes **ready-to-use datasets** for various computer vision tasks:

- classification
- detection
- segmentation
- video analysis

They have a standardized format and are compatible with PyTorch's `DataLoader`. This removes the need for manual downloads, extraction or writing custom parsers.

Look more at [[TorchVision Datasets]]
## Models

TorchVision also includes **built-in models**: pre-configured architectures that can be:

- trained from scratch on our own data
- loaded with **pre-trained weights**, for transfer learning or fine-tuning

Some well-known examples:

- **VGG16** → deep CNN known for its simplicity and uniform layer design
- **ResNet50** → introduces deep residual connections, enabling very deep networks
- **GoogLeNet** → uses inception modules to process multiple filter sizes in parallel

Loading ResNet50 with pre-trained weights takes a single line:

```python
from torchvision.models import resnet50, ResNet50_Weights

model = resnet50(weights=ResNet50_Weights.DEFAULT)
```

Look more at [[TorchVision Models]]