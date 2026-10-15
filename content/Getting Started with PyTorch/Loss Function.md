Now we are in the **Training** part of the ML pipeline.

![[ml_pipeline.png]]

```python
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_function(outputs, y)
    loss.backward()  # FOCUS IS HERE
    optimizer.step()
```

These three lines come up again and again during training:

```python
loss = loss_function(outputs, targets)  # Measure
loss.backward()                         # Diagnose
optimizer.step()                        # Update
```

- **Measure:** Measure how wrong the model's predictions are by comparing them with the true answers. This produces a single number called the **loss**.
- **Diagnose:** `loss.backward()` calculates the gradients, showing how each weight contributes to the loss and how changing it would affect the error.
- **Update:** `optimizer.step()` uses those gradients to adjust the model's weights to reduce the loss.

The goal is to **minimize the loss**.
## 1. Loss Functions

==A loss function compares the model's predictions with the true answers (`targets`) and returns a number representing how wrong the predictions are.==

- Higher loss → worse predictions.
- Lower loss → better predictions.
- Loss = 0 → perfect predictions according to that loss function.

![[loss_comparison.png]]

Different tasks require different loss functions:
```python
loss_function = nn.MSELoss()
loss_function = nn.CrossEntropyLoss()
```
## 2. Mean Squared Error (MSE)

MSE is commonly used for **regression**, where the model predicts continuous numerical values, such as delivery time, distance, temperature, or price.
### Example: The delivery guy

Imagine a delivery company wants to predict delivery times.

| Delivery | Predicted time | Actual time | Error (prediction − actual) |
| -------- | -------------- | ----------- | --------------------------- |
| 1        | 6 min          | 4 min       | +2                          |
| 2        | 3 min          | 5 min       | −2                          |

To calculate the average error, we subtract the actual value from the predicted value and average the results.
$$Average Error=2+(−2)2=0$$

The result is zero, which incorrectly suggests that the model made no mistakes!
Both predictions were actually wrong. The positive and negative errors simply cancel each other out.
### The solution: Square the errors

Instead of averaging the raw errors, we square each error before averaging them.

**Mean Squared Error formula:**
$$
\mathrm{MSE} = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)^2
$$
Where:
- $N$ = number of predictions.
- $y^i$ = predicted value.
- $y_i$ = actual value.

For our example:
$$
\mathrm{MSE}=\frac{(6-4)^2+(3-5)^2}{2}=\frac{4+4}{2}=4
$$
Now both mistakes count, and they cannot cancel each other out.
### Why use MSE?

1. ==It prevents errors from cancelling each other out.== Squaring removes negative signs, so all errors contribute positively to the loss.
2. ==It penalizes larger errors more heavily.== Being off by 10 minutes is much worse than being off by 1 minute because the squared errors are 100 and 1, respectively.

The lower the MSE, the closer the predictions are to the actual values.
An MSE of zero means every prediction exactly matches its target.

==Use MSE when predicting numerical values, such as delivery times, distances, temperatures, or prices ([[Regression Problem]]).==
## 3. Cross-Entropy Loss

==Cross-entropy loss is commonly used for [[Classification Problem|classification]], where the model predicts a category rather than a continuous numerical value.==

Examples:
- Recognizing a digit from 0 to 9.
- Classifying an image as a cat or a dog.
- Predicting which category a text belongs to.
### How does it work?

Take the MNIST dataset. The model must classify an image as one of the digits from 0 to 9.

The model produces a score for each possible class. With PyTorch's `nn.CrossEntropyLoss()`, these scores are called **logits**.
==Logits are raw scores, not probabilities.== Cross-entropy internally applies the appropriate mathematical operations to calculate the loss.

For example, suppose the actual digit is `3`.

- **Prediction A:** The model assigns 95% probability to `7` and only 2% to `3`. ==This produces a high loss because the model is confidently wrong.==
- **Prediction B:** The model assigns 55% probability to `7` and 30% to `3`. ==This produces a lower loss because the model assigns more probability to the correct class.==

![[Pasted image 20261015174832.png]]

==🟠The key idea is that cross-entropy penalizes the model when it assigns a low probability to the correct answer. The more confident the model is in an incorrect answer, the more severe the penalty can be.==
### Cross-Entropy formula

For one example with one correct class:
$$
L = -\log(p_{\text{correct}})
$$
Where $p_{\text{correct}}$ is the probability assigned to the true class.

For $N$ examples:
$$
L = -\frac{1}{N}\sum_{i=1}^{N}\log(p_{i,y_i})
$$
Where:

- $N$ = number of examples.
- $p_{i,y_i}$ = predicted probability assigned to the correct class for example i.
- $log⁡$ = natural logarithm.

- ==If the model assigns a probability close to 1 to the correct class, the loss approaches zero.==
- ==If the model assigns a probability close to 0 to the correct class, the loss becomes very large.==

The goal is to make the model assign high probability to the correct answers.
### PyTorch example

```python
loss_function = nn.CrossEntropyLoss()
loss = loss_function(outputs, labels)
```

Important details:

- `outputs` contains raw logits, usually shaped `[batch_size, num_classes]`.
- `labels` contains the correct class indices, usually shaped `[batch_size]`.
- **Do not apply** **`softmax`** to the outputs before passing them to `nn.CrossEntropyLoss()` PyTorch handles the necessary calculations internally.
### Important warnings

- **MSE for classification can work, but often performs less effectively.** Cross-entropy is generally the standard choice for multi-class classification because it directly penalizes low probability assigned to the correct class.
- **Cross-entropy for regression is generally inappropriate.** It expects classification-style logits and class targets, not ordinary continuous numerical targets.
- **Do not compare raw loss values from different loss functions.** An MSE of `0.08` is not automatically better than a cross-entropy loss of `2.3`. They measure different things on different scales.
- **Monitor the loss during training.** We generally want the training loss to decrease. Validation loss is also important to check whether the model generalizes to unseen data.

The goal is always to minimize the loss, but the correct loss function depends on the task.
## 5. Other Common Loss Functions in PyTorch

| Loss function            | Typical use                          | What it measures                                                                |
| ------------------------ | ------------------------------------ | ------------------------------------------------------------------------------- |
| `nn.MSELoss()`           | Regression                           | Average squared difference between predictions and targets                      |
| `nn.CrossEntropyLoss()`  | Multi-class classification           | Penalizes low probability assigned to the correct class                         |
| `nn.L1Loss()`            | Regression                           | Average absolute difference; less sensitive to outliers than MSE                |
| `nn.BCEWithLogitsLoss()` | Binary or multi-label classification | Binary cross-entropy combined with a sigmoid operation                          |
| `nn.NLLLoss()`           | Multi-class classification           | Negative log-likelihood; expects log-probabilities, typically from `LogSoftmax` |
| `nn.SmoothL1Loss()`      | Regression                           | Combines L1 and squared-error behavior; less sensitive to outliers than MSE     |
| `nn.KLDivLoss()`         | Comparing probability distributions  | Measures a form of divergence between distributions                             |
### Quick rule

- Predicting a number → **MSE** or another regression loss.
- Predicting one category out of many → **CrossEntropyLoss**.
- Predicting a binary outcome or independent labels → **BCEWithLogitsLoss**.

Remember the training sequence:

**Measure → Diagnose → Update**

`loss_function()` measures the error, `loss.backward()` calculates the gradients, and `optimizer.step()` updates the model's weights.

Next see 
[[Omptimizer & Gradients]]
