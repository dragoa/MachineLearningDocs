At its core optimization is about **finding the best possible value of a function**, a maximum or a minimum.

In machine learning this usually means adjusting hyperparameters or architecture to improve some objective metric: accuracy, speed, or memory efficiency. 

![[optimization.png|350]]

# 1. Evaluation Metrics

Different evaluation metrics tell you different things about your model. The metric you optimize determines what your model gets good at.
### Confusion Matrix

All classification metrics are based on **four fundamental outcomes**:

```
                              REAL
                       ┌─────────────┬─────────────┐
                       │  Positive   │  Negative   │
          ┌────────────┼─────────────┼─────────────┤
          │ Positive   │     TP      │     FP      │
PREDICTED │            │ True        │ False       │
          │            │ Positive    │ Positive    │
          ├────────────┼─────────────┼─────────────┤
          │ Negative   │    FN       │     TN      │
          │            │ False       │ True        │
          │            │ Negative    │ Negative    │
          └────────────┴─────────────┴─────────────┘

```

- TP (True Positive): The model predicted positive, and the real class is positive ✅

- FP (False Positive): The model predicted positive, but the real class is negative ❌
	*Example: The model incorrectly diagnoses a healthy plant as diseased.*
	
- FN (False Negative): The model predicted negative, but the real class is positive ❌
	*Example: The model fails to detect an infected plant.*

- TN (True Negative): The model predicted negative, and the real class is negative ✅

#### Binary vs Multiclass
  
**Binary:** One "positive" class you want to detect vs "negative" (everything else)

- Examples: Spam/Not spam, Disease/Healthy, Fraud/Legitimate

**Multiclass:** Multiple distinct classes (no inherent "positive")

- Examples: Dog/Cat/Bird, Digits 0-9
- Calculate metrics per class (treating each as "positive" vs rest), then combine using either macro / weighted / micro averaging

### Accuracy
![[Accuracy#Definition]]
### Precision
![[Precision#Definition]]

### Recall
![[Recall#Definition]]

### F1 Score
![[F1 Score#Definition]]

---
# 2. Computing Metrics in PyTorch

PyTorch's `torchmetrics` library computes these metrics efficiently during evaluation.
  
- **Binary:** `task="binary"`
- **Multiclass:** `task="multiclass", num_classes=X` choosing an averaging strategy

### Averaging Strategies Explained

To understand the difference, imagine a **3-class animal classifier** evaluated on 150 images:  

```
Dataset:
├─ Dog: 100 images (large class)
├─ Cat: 40 images (medium class)
└─ Bird: 10 images (small class, rare)
```

After evaluation, per-class F1 scores:

```
Dog F1 = 0.90 (well represented in dataset)
Cat F1 = 0.70 (less data, harder)
Bird F1 = 0.40 (very few samples, model struggles)
```

Now we need **one number** to describe overall performance. The three strategies give very different answers:
  
==**Macro-average** → Simple mean, all classes treated equally==

```
Macro F1 = (0.90 + 0.70 + 0.40) / 3 = 0.67

Dog: ██████████ 0.90 weight = 1/3
Cat: ███████ 0.70 weight = 1/3
Bird: ████ 0.40 weight = 1/3
─────────────

average = 0.67
```

==✅ Use when: Every class matters equally regardless of frequency==
⚠️ Sensitive to poor performance on rare classes (Bird drags the score down)

==**Weighted-average** → Mean weighted by number of samples per class==

```
Weighted F1 = (100×0.90 + 40×0.70 + 10×0.40) / 150
= (90 + 28 + 4) / 150
= 0.81

Dog: ██████████ 0.90 weight = 100/150 = 67%
Cat: ███████ 0.70 weight = 40/150 = 27%
Bird: ████ 0.40 weight = 10/150 = 7%
─────────────

average = 0.81
```

==✅ Use when: Class frequency in the dataset reflects real-world distribution==
⚠️ Can hide poor performance on rare but important classes (Bird barely matters)

==**Micro-average** → Pool all TP, FP, FN across classes, then compute once==

```
Imagine totals across all classes:

Total TP = 90 + 28 + 4 = 122
Total FP = 10 + 12 + 6 = 28
Total FN = 10 + 12 + 6 = 28

Micro Precision = 122 / (122 + 28) = 0.81
Micro Recall = 122 / (122 + 28) = 0.81
Micro F1 = 0.81

(In multiclass, micro F1 ≈ accuracy)
```

==✅ Use when: You care about total correct predictions across everything==
==⚠️ Dominated by large classes, Bird is nearly invisible==

`average="macro"` is the default, a safe choice that doesn't hide poor performance on any single class.

**Typical workflow:**

```python
import torchmetrics

# Create metrics
f1 = torchmetrics.F1Score(task="multiclass", num_classes=10, average="macro")

# During evaluation loop
for images, labels in val_loader:
	predictions = model(images)
	f1.update(predictions, labels) # Accumulate

# Get final score
print(f"F1 Score: {f1.compute():.3f}")
```

The metric state is updated on each batch. At the end of the epoch it is computed to evaluate the overall performance.

For an example of how to choose a metric in a real scenario, take a look at [[Real World Examples]]

---
# 3. Learning Rate Schedulers

### What is Optimization?
  
**Optimization** in ML means tuning your model to achieve the best possible performance according to a specific metric like precision, recall, f1, etc.

==An **hyperparameter** is a setting that controls how a model is trained== and is chosen before or during training rather than learned directly from the data. The **learning rate** is one of the most fundamental hyperparameters, controlling how much the model's parameters are updated at each training step.
### The Learning Rate Effect
  
The learning rate determines the step size the optimizer takes when updating weights at each iteration. Its relationship with accuracy follows a clear pattern:

![Learning Rate vs Accuracy](images/lr_vs_accuracy.png)
  
- **Too small** → training is slow and the model may get stuck at a suboptimal point
- **Too large** → the model overshoots and bounces around, never converging
- **Just right** → highest accuracy

This forms an **inverted U-shape**: performance is low at both extremes and peaks in the middle. The goal of optimization is to find that peak.

### Before Tuning: How Can You Improve These Metrics?

==Before touching any hyperparameter, rule out **data problems**. The model can only be as good as what you feed it.==

```
EXTERNAL FACTORS (check these first)
──────────────────────────────────────────────────────────
Data quantity      Not enough examples of a class?
                   → Model won't learn to recognize it
                   → Gather more data from open datasets

Data quality       Noisy labels?
                   → Model learns wrong patterns
                   → "Garbage in, garbage out"

Feature quality    Blurry or low-resolution images?
                   → Model can't find useful patterns
                   → Apply preprocessing: resize, crop, normalize
```
  
```
INTERNAL FACTORS
──────────────────────────────────────────────────────────
Architecture       Number of layers, neurons per layer,
                   activation functions

Regularization     Dropout rate, weight decay

Training           Learning rate, batch size,
                   number of epochs, optimizer choice
```

## Learning Rate Schedulers

A fixed learning rate always forces a trade-off between speed and precision:

![High vs Low Learning Rate|630](images/high_vs_low_lr.png)

- ==**High LR:** Accuracy rises fast in early epochs, then flatlines==
- ==**Low LR:** Climbs slowly but eventually surpasses the high LR, at the cost of many more epochs==

**The question:** Can we get the fast start of a high LR *and* the precision of a low LR?
**The answer:** Yes — with a **learning rate scheduler**.

![[Learning Rate Schedulers]]

---
# 4. Tuning Hyperparameters

Several hyperparameters can be used to enhance the performance of a model.

```
┌───────────────────────┬───────────────────────┬───────────────────────┐
│     Architectural     │       Training        │    Regularization     │
├───────────────────────┼───────────────────────┼───────────────────────┤
│ • Number of Layers    │ • Learning Rate &     │ • Weight Decay        │
│ • Neurons/Filters     │   Schedulers          │ • Dropout             │
│ • Activations         │ • Optimizer           │ • Early Stopping      │
│                       │ • Batch Size          │ • BatchNorm           │
└───────────────────────┴───────────────────────┴───────────────────────┘
```
##### Architectural
![[Architectural#Definition]]

##### Training
![[Training#Definition]]

##### Regularization

![[Regularization#Definition]]

##### Where to start

With so many hyperparameters, fine-tuning can feel overwhelming.

1. **Establish a simple baseline model** — small, low-complexity, with minimal tuning. Even if it's not ideal, it offers a sanity check and a performance floor that more complex models should exceed. It also shows whether your dataset is even learnable in the first place.
2. **Use PyTorch's default hyperparameters first.** Run a model with all defaults to gauge its performance — this provides a stable foundation for measuring future improvements.
3. **Look for reference points in the literature.** Has anyone else tackled a similar problem? Reviewing prior work gives insight into effective architectures, learning rates, and regularization strategies. _Example: for a botanical classification app, researchers may have built CNNs for similar datasets — like insects. These published configurations are valuable starting points to replicate and iterate on._

##### Iterating

Even with baselines, hyperparameter tuning is inherently iterative; ==it's not about guessing the perfect combination right away==. Improve the model progressively through structured trial and error. Initially, focus on the hyperparameters most likely to have substantial impact — ==learning rate, batch size, and dropout== — and observe how each one influences your objective.

---
# 5. Flexible Architecture Design

Have a look at [[Flexible CNN Architecture]], where we will create a flexible CNN architecture and use **Optuna** to optimize its hyperparameters.

---

# 6. Optimize Model Efficiency

In resource-constrained environments, such as edge devices like smartphones, drones, smartwatches or other low-power embedded systems, you need to consider additional factors.

These include model size (particularly its memory footprint), inference time (how long it takes to make a prediction), power consumption, and latency and throughput requirements.
These factors involve trade-offs. Improving accuracy with a deeper network may lead to a slower and larger model, which may not be suitable for real-time applications.

As an example, we will compare two models, an optimized CNN and a ResNet34, using three metrics: accuracy, model size, and inference time.

First, let's have a look to the **memory footprint** of a model:
```python
def get_model_size(model):

	# Model parameters
	param_size = 0
	for param in model.parameters(): # all trainable params
		param_size += param.nelement() * param.element_size()
	
	# Model buffers, non-trainable parameters
	buffer_size = 0
	for buffer in model.buffers():
		buffer_size += buffer.nelement() * buffer.element_size()
	
	# Convert bytes to megabytes
	size_in_mb = (param_size + buffer_size) / 1024**2
	return size_in_mb # model memory footprint
```

==Model buffers are non-trainable tensors==, so they are not updated by the optimizer during training. Examples include the running statistics of batch normalization, pre-computed constants, and fixed embeddings. Although they are not trained, buffers are important for inference and model behavior.

**Inference time** is defined as how long the model takes to make a prediction (crucial for real-time applications).
```python
def measure_inference_time(model, input_data, num_iterations=100):
	
	model.eval() # Set to evaluation mode
	# Move input to the same device as the model
	device = next(model.parameters()).device
	input_data = input_data.to(device)
	
	# Warmup
	with torch.no_grad():
		for _ in range(10):
			_= model(input_data)
	
	# Measure
	start_time = time.time()
	with torch.no_grad():
		for _ in range(num_iterations):
			_ = model(input_data)
			
	end_time = time.time()
	avg_time = (end_time - start_time) / num_iterations
	return avg_time * 1000 # Convert to ms
```

The warmup loop is important because of PyTorch's lazy initialisation and GPU startup costs. The average inference time is computed over several iterations of the same input.

After computing these metrics, you can combine them into a comprehensive model summary, which can be used to determine the best model: 
```
{
	'accuracy': 0.80,
	'model_size_mb': 9.2,
	'inference_time_ms': 0.51
}
```

![[model_efficiency.png]]

But real-world choices aren't always so clear-cut. When no single model outperforms the other on every metric, you need a smarter selection process.

### Constraint Based Selection

Use this approach when your deployment targets resource-constrained environments (mobile phones or edge devices). ==It filters out models that exceed the memory or speed limits==, then selects the most accurate model among the remaining ones.

```python
def select_best_model_constraint_based(results, max_size_mb, max_inference_ms):
	
	results = results.to_dict(orient="index")
	# Filter models that meet both size and inference time constraints
	viable_models = {
		name: metrics for name, metrics in results.items()
		if metrics["model_size_mb"] <= max_size_mb and
			metrics["inference_time_ms"] <= max_inference_ms
	}
	if not viable_models:
		# If no models satisfy the constraints, inform the user
		print("No models meet all constraints. Consider relaxing constraints.")
		return None
		
	# Among viable models, select the one with the highest accuracy
	best_model = max(viable_models.items(), key=lambda x: x[1]["accuracy"])
	return best_model[0], viable_models # Return best model name and all viable      models
```

### Weighted Scoring

Use this when the requirements are more flexible. Instead of fixed rules, you ==assign a weight to each metric, reflecting its relative importance to you.== The function then returns the model with the best weighted score.

```python
def select_best_model_weighted(results, weights=None):
	results = results.to_dict(orient="index")
	
	if weights is None:
		# Default weights: prioritize accuracy more than efficiency
		weights = {"accuracy": 0.5, "model_size_mb": 0.2, "inference_time_ms":                      0.3}
		
	metrics = list(weights.keys()) # List of metrics to consider
	normalized = {name: {} for name in results} # Initialize normalized results
```

The weighted score of each model is computed on the **normalized** metrics (not the raw values), where $acc'$, $size'$ and $time'$ are all on a 0 to 1 scale and higher is always better
$$\text{score} = 0.5 \cdot acc' + 0.2 \cdot size' + 0.3 \cdot time'$$

This is essential as these metrics are on different scales and also differ in the desired direction (higher is better for accuracy, lower is better for model size and inference time)

```python
# Normalize each metric across all models
for metric in metrics:
	
	values = [res[metric] for res in results.values()] # Get all values
	min_val, max_val = min(values), max(values)
	# Avoid division by 0
	range_val = max_val - min_val if max_val != min_val else 1.0

	for name, res in results.items():
	
		value = res[metric]
		if metric == "accuracy":
			# For accuracy: higher is better -> normalize directly
			norm_value = (value - min_val) / range_val
		else:
			# For model size and inference time: lower is better
			norm_value = 1 - (value - min_val) / range_val
			normalized[name][metric] = norm_value
```

For accuracy, where higher is better, you apply min-max normalization. For inference time and model size, where lower is better, you apply the inverse:
$$x' = \frac{x - \min(x)}{\max(x) - \min(x)} \qquad \text{(higher is better)}$$ $$x' = 1 - \frac{x - \min(x)}{\max(x) - \min(x)} \qquad \text{(lower is better)}$$
Now that all metrics are on a 0 to 1 scale, you can multiply each normalized metric by its weight and sum the results:

```python
# Compute weighted score for each model
scores = {
    name: sum(weights[metric] * normalized[name][metric] for metric in metrics)
        for name in results
}
    
# Select model with highest weighted score
best_model = max(scores.items(), key=lambda x: x[1])

return best_model[0], scores  # Return best model name and all scores
```
