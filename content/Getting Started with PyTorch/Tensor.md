==Tensors are PyTorch's main data structure for storing and manipulating data.== Many PyTorch errors come from tensors having the wrong **shape**, **data type**, or **device**, so being comfortable with tensors is essential.

## Tensor Shapes

```python
# Your delivery data
distances = torch.tensor([[3.0], [7.0], [12.0], [18.0], [22.0], [28.0]])
print(distances.shape)
```

```text
torch.Size([6, 1])
```

The shape tells us how the data is organized:

- `6` → the **batch size**, how many samples per batch, meaning there are 6 samples.
- `1` → the number of **features per sample**. In this case, each sample has one feature: delivery distance.

The **first dimension is usually the batch dimension**. It tells the model how many samples it is processing at once.
The remaining dimensions describe what each individual sample looks like.

```python
simple_model = # Expects 1 feature per sample
nn.Linear(1, 1) output = simple_model(distances)
# Works!
```

If we provide multiple features, an error will be raised as the model expects **1 feature**, but each sample contains **3 features**.
```python
# Your delivery data
# distance, hour, weather
distances = torch.tensor([[3.0, 7.0, 1.0], [18.0, 22.0, 2.0]])
print(distances.shape)
simple_model = nn.Linear(1, 1)
output = simple_model(distances) # RuntimeError
```

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x3 and 1x1)
```

PyTorch will tell you what went wrong, but not always how to fix it.

## Tensor Data Types (`dtype`)

```python
int_tensor = torch.tensor([1, 2, 3])
print(int_tensor.dtype) # torch.int64
float_tensor = torch.tensor([1.0, 2.0, 3.0])
print(float_tensor.dtype) # torch.float32

float_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
```

Every tensor has a data type. PyTorch can infer the dtype from the values we provide, or we can set it explicitly specifing the `dtype` or using the `.float()` method.

For neural networks, **`float32` is commonly used** because it provides a good balance between precision, memory usage, and computational speed.
Other common dtypes include:

- `float64` → higher precision, but uses more memory.
- `float32` → common choice for neural networks.
- `int64` → common integer dtype in PyTorch.
- `int8` → uses less memory, but has a much smaller range.

### Mixing Types

What happens when you mix types? It was the case that PyTorch would throw an error but that's no longer true. ==PyTorch uses type promotion when operations involve compatible tensors with different dtypes.==

```python
float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)

mixed_tensor = float_tensor + int_tensor
print(f"mixed type: {mixed_tensor.dtype}")
```
```text
mixed type: torch.float32
```

PyTorch promotes the integer tensor to a compatible floating-point type so that the operation can be performed.

> **Important:** Type promotion doesn't mean that every combination of dtypes will work automatically. Some operations still require compatible dtypes.

## Creating Tensors

The simplest way to create a tensor is from a Python list:
```python
my_distances = [[3.0], [7.0], [12.0], [18.0]]
tensor_from_list = torch.tensor(my_distances)
```

You can convert a NumPy array into a PyTorch tensor:
```python
import numpy as np
numpy_array = np.array([1.0, 2.0, 3.0])
tensor_from_numpy = torch.from_numpy(numpy_array)
```
`torch.from_numpy()` **shares memory** with the NumPy array, therefore, modifying one can affect the other.

PyTorch also provides convenient functions for creating tensors with specific patterns:
```python
zeros = torch.zeros(3, 3) # 3x3 tensor of zeros
ones = torch.ones(2, 4) # 2x4 tensor of ones
random = torch.rand(5, 5) # 5x5 tensor with random values
```
These are especially useful when creating test data or initializing tensors.
## Reshape Tensors

Models often expect tensors to have a specific shape:
```python
# Your delivery data
single_distance = torch.tensor(25.0)
with_batch = single_distance.unsqueeze(0)
# torch.Size([1]) —> added batch dim
ready_for_model = with_batch.unsqueeze(1)
# torch.Size([1, 1]) —> now ready!
```

`unsqueeze(dim)` **adds a dimension of size 1** at the specified position.
`squeeze()` removes dimensions whose size is `1`.

```python
x = torch.tensor([[1.0]])
print(x.shape)
# [1, 1]

x = x.squeeze()
print(x.shape)
# []
```

Be careful: ==`squeeze()` removes all dimensions of size 1 by default.==
You can specify which dimension to remove using `x = x.squeeze(0)`

Another useful operation is `reshape()`:
```python
x = torch.tensor([1, 2, 3, 4, 5, 6])
x = x.reshape(2, 3)
print(x)
```

```text
tensor([
    [1, 2, 3],
    [4, 5, 6]
])
```

This is useful when you need to reorganize a tensor into a different shape without changing its data. ==The number of elements must stay the same.==
## Indexing and Slicing

Tensor indexing and slicing work similarly to Python lists.

```python
# Your predictions from the delivery model
predictions = torch.tensor([[14.9], [24.1], [35.6], [45.2]])

# Get the first prediction
first = predictions[0]
print(first)

# Get first three predictions
first_three = predictions[:3] # it's still a tensor
print(first_three)

# Use .item() to get a Python number
value = predictions[0].item() 
# 14.899999618530273, <class 'float'>
```

`.item()` only works when the tensor contains exactly one element.

For tensors with multiple dimensions, you can index each dimension separately.

```python
# Multiple features per sample
data = torch.tensor([[3.0, 8.0, 1.0], # distance, hour, weather
	[7.0, 17.0, 2.0],
	[12.0, 12.0, 1.0]])

distances = data[:, 0] # tensor([ 3., 7., 12.])
```

Here:
- `:` means **all rows**
- `0` means the **first column**

Remember that Python uses **zero-based indexing**, so the first row/column has index `0`.
## Tensor Math

Let's start with a computation from a single neuron model:

$$
\text{output} = \text{weight} \times \text{distance} + \text{bias}
$$

```python
distances = torch.tensor([[3.0], [7.0], [12.0]])
weight = 2.3
bias = 8.0

output = weight * distances + bias
print(output)
```

```
tensor([[14.9000],
        [24.1000],
        [35.6000]])
```

==PyTorch's tensor math works element-wise, so each element is operated on independently.==

The computation applies the same weight to each distance and then adds the same bias to each result. ==The math looks just like regular Python, but PyTorch runs these operations efficiently on all elements at once.==
### Scalars

A **scalar** is a single value. When we multiply a tensor by a scalar, the operation is applied to every element.

```python
distances = torch.tensor([[3.0], [7.0], [12.0]])

result = 2.3 * distances
print(result)
```

```
tensor([[ 6.9000],
        [16.1000],
        [27.6000]])
```

Mathematically:

```
2.3 * [[3.0], [7.0], [12.0]]

= [[2.3 * 3.0],
   [2.3 * 7.0],
   [2.3 * 12.0]]
```

The scalar `2.3` is applied to every element in the tensor.
### Tensors with the Same Shape

==When two tensors have the same shape, PyTorch performs operations element by element.==

```python
a = torch.tensor([[1.0], [2.0], [3.0]])
b = torch.tensor([[3.0], [7.0], [12.0]])

result = a * b
print(result)
```

```
tensor([[ 3.],
        [14.],
        [36.]])
```

Mathematically:

```
[[1.0],      [[3.0],       [[1.0 * 3.0],
 [2.0],   *   [7.0],   =    [2.0 * 7.0],
 [3.0]]       [12.0]]       [3.0 * 12.0]]
```

==Each element is multiplied by the element at the same position.==

The same principle applies to addition, subtraction, and division.

```
a + b  # Element-wise addition
a - b  # Element-wise subtraction
a * b  # Element-wise multiplication
a / b  # Element-wise division
```

**Important:** `*` performs element-wise multiplication, not matrix multiplication. For matrix multiplication, PyTorch provides `@` and `torch.matmul()`.
## Broadcasting

When working with tensors, we don't always want to create tensors with exactly the same shape.
Imagine three deliveries where each has three features: distance, hour, and weather.

```python
# distance, hour, weather
deliveries = torch.tensor([
    [3.0, 8.0, 1.0],
    [7.0, 17.0, 2.0],
    [12.0, 12.0, 1.0]
])
```

We want to apply an adjustment factor to each feature:

- `1.1` → increase the distance by 10%.
- `1.0` → leave the hour unchanged.
- `5.0` → apply a penalty to the weather feature.

Instead of repeating these values for every delivery, we can create a single tensor containing the adjustment factors.

```python
adjustments = torch.tensor([1.1, 1.0, 5.0])

result = deliveries * adjustments
print(result)
```

```
tensor([[ 3.3000,  8.0000,  5.0000],
        [ 7.7000, 17.0000, 10.0000],
        [13.2000, 12.0000,  5.0000]])
```

The shapes are:

```
deliveries.shape  = torch.Size([3, 3])
adjustments.shape = torch.Size([3])
result.shape      = torch.Size([3, 3])
```

==PyTorch automatically applies the adjustment factors to every row. This automatic expansion is called broadcasting.==

Broadcasting allows PyTorch to perform operations on tensors with different shapes, as long as their dimensions are compatible. It avoids the need for manual repetition or loops.
### Broadcasting with a Scalar

Remember how a scalar could update every value in our distance tensor?

```python
values = torch.tensor([[1.0, 2.0, 3.0]])

result = values + 5.0
print(result)
```

```
tensor([[6., 7., 8.]])
```

Mathematically:

```
[[1.0, 2.0, 3.0]] + 5.0
= [[1.0 + 5.0, 2.0 + 5.0, 3.0 + 5.0]]
= [[6.0, 7.0, 8.0]]
```

The scalar `5.0` is automatically applied to every element.
### Broadcasting with Different Shapes

What happens when we combine a tensor with shape `[1, 3]` and a tensor with shape `[1, 1]`?

```python
a = torch.tensor([[1.0, 2.0, 3.0]])
b = torch.tensor([[5.0]])

print(a.shape)  # torch.Size([1, 3])
print(b.shape)  # torch.Size([1, 1])

result = a + b
print(result)
```

```
tensor([[6., 7., 8.]])
```

The shapes are:

```
a: [[1.0, 2.0, 3.0]]    shape = [1, 3]
b: [[5.0]]              shape = [1, 1]
```

The second tensor contains only one value. PyTorch automatically expands it to match the first tensor's shape:

```
[[1.0, 2.0, 3.0]]    shape = [1, 3]
[[5.0, 5.0, 5.0]]    shape = [1, 3]
```

Now both tensors have the same shape, so they can be added element by element.

**Broadcasting rule:** ==When comparing two tensor shapes, PyTorch checks their dimensions from right to left. Two dimensions are compatible when they are equal or when at least one of them is `1`. Missing dimensions are treated as if they were `1`.==

If the dimensions are compatible, PyTorch expands the dimensions of size `1` as needed. Otherwise, the operation raises an error.
### Broadcasting a Vertical Array with a Horizontal Array

Let's look at a more complex example: combining a tensor with shape `[3, 1]` and a tensor with shape `[1, 3]`.

```python
vertical = torch.tensor([
    [1.0],
    [2.0],
    [3.0]
])
horizontal = torch.tensor([[10.0, 20.0, 30.0]])

print(vertical.shape)    # torch.Size([3, 1])
print(horizontal.shape)  # torch.Size([1, 3])

result = vertical + horizontal
print(result)
```

```
tensor([[11., 21., 31.],
        [12., 22., 32.],
        [13., 23., 33.]])
```

Let's see how broadcasting works.

**Step 1: Compare the shapes.**

```
vertical:   [3, 1]
horizontal: [1, 3]
```

**Step 2: Expand the dimensions of size** **`1`**.

The vertical tensor expands across the columns, and the horizontal tensor expands across the rows.
Both tensors now behave as if they had shape `[3, 3]`.

**Step 3: Perform the element-wise addition.**

```python
[[ 1.0,  1.0,  1.0],    [[10.0, 20.0, 30.0],
 [ 2.0,  2.0,  2.0],  +  [10.0, 20.0, 30.0],
 [ 3.0,  3.0,  3.0]]    [10.0, 20.0, 30.0]]

= [[11.0, 21.0, 31.0],
   [12.0, 22.0, 32.0],
   [13.0, 23.0, 33.0]]
```

The resulting shape is `[3, 3]`.

This is one of the most useful features of broadcasting: we can combine data along different dimensions without manually repeating values.
### When Broadcasting Fails

Not every combination of shapes is compatible.

```python
a = torch.tensor([[1.0, 2.0, 3.0]])  # [1, 3]
b = torch.tensor([[1.0, 2.0]])       # [1, 2]

result = a + b  # RuntimeError
```

The last dimensions are `3` and `2`. They are not equal, and neither is `1`, so PyTorch cannot broadcast these tensors together.

**Remember:**

- Same shapes → operations work element by element.
- Scalars → can be applied to every element of a tensor.
- Different shapes → broadcasting works when each pair of dimensions is equal or at least one dimension is `1`.
- Incompatible shapes → PyTorch raises a `RuntimeError`.

Broadcasting appears everywhere in deep learning, including adjusting multiple features across batches, combining different dimensions of data, and applying transformations efficiently.