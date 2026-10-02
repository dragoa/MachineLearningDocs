Starting from the CIFAR-10 dataset, we build a flexible CNN (Convolutional Neural Network) that lets us easily test different architectural configurations.

![[flexible_cnn.png]]

##### CNN block

A CNN block consist of:

- a **[[Convolutional Layer|convolutional layer]]** that computes the convolution operation using a given number of filters and a kernel size
- a **ReLU [[Activation Functions|activation function]]**
- a **max [[Pooling|pooling]] layer** which a form of regularization that takes the maximum value from a pool of values

In a flexible architecture, we want to easily change the values of these architectural hyperparameters.

```python
class FlexibleCNN(nn.Module):

    def __init__(self, n_layers, n_filters, kernel_sizes, dropout_rate, fc_size):

        super(FlexibleCNN, self).__init__()
        blocks = []       # list of CNN blocks
        in_channels = 3   # RGB input

        for i in range(n_layers):
            out_channels = n_filters[i]        # i-th filter count
            kernel_size = kernel_sizes[i]      # i-th kernel size
            padding = (kernel_size - 1) // 2   # 'same' padding

            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            blocks.append(block)

            # the next block's input channels = this block's output channels
            in_channels = out_channels
```

==`n_layers` controls how many convolutional blocks are stacked==, while `n_filters` and `kernel_sizes` are lists. For example, to make the first block use 64 filters with a 3×3 kernel and the second use 32 filters with a 5×5 kernel, you'd set `n_filters = [64, 32]` and `kernel_sizes = [3, 5]`.

Padding is computed as `(kernel_size - 1) // 2`, which ==keeps the input and output spatial dimensions the same, this is known as **"same" padding**==, and it makes it much easier to track the feature map size across layers, especially with small inputs like CIFAR-10's.

##### Tracking spatial dimensions

==Spatial dimensions describe how the size of the data changes as it flows through the network.== Each image in CIFAR-10 is 32×32 pixels with three color channels. Since every block ends with a max pooling layer (kernel size 2, stride 2), the spatial size is halved after each block:

![[spatial_dimension.png|396]]

After three blocks, a 32×32 input shrinks all the way down to 4×4.

Because the final flattened size depends on the input size, the number of blocks, the kernel sizes, and the number of filters none of which are fixed in a flexible architecture, ==the classifier can't be built until the first forward pass, once the actual flattened size is known.== Therefore, the classifier is initialized dynamically inside the forward pass rather than being defined a priori.

```python
def forward(self, x):

    # Extract the device from the input tensor
    device = x.device

    # Apply feature extractor
    x = self.features(x)

    # Flatten for the classifier
    flattened = torch.flatten(x, 1)
    flattened_size = flattened.size(1)

    # Dynamically initialize the classifier on the first forward pass
    if self.classifier is None:
        self._create_classifier(flattened_size, device)

    return self.classifier(flattened)
```

This ensures the first fully connected layer always gets the correct number of input units, regardless of how many convolutional layers you choose or how they're configured.

##### Classification head

The classification layers and the dropout layer applied before each of them are parameterized too.

```python
class FlexibleCNN(nn.Module):

    def __init__(self, n_layers, n_filters, kernel_sizes, dropout_rate, fc_size):
        ...

    def _create_classifier(self, flattened_size, device):

        # Dynamically create the classifier based on the flattened size
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(flattened_size, self.fc_size),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.fc_size, 10)  # CIFAR-10: 10 classes
        ).to(device)
```

`fc_size` controls the number of neurons in the hidden fully connected layer, the output layer is fixed at 10 neurons since CIFAR-10 has 10 classes. ==A larger `fc_size` gives the model more capacity to represent complex decision boundaries==, but it also increases the risk of [[Overfitting|overfitting]] and requires more data to train effectively. Parameterizing it lets you explore that trade-off directly.

Put together, this architecture exposes five hyperparameters you can tune: `n_layers`, `n_filters`, `kernel_sizes`, `dropout_rate`, and `fc_size`. Even with manual tuning, a flexible design like this saves time and lets you test ideas faster than hardcoding a fixed architecture for every experiment.

# Hyperparameter Optimization with Optuna

Which combination of the hyperparameters described above is the best choice? Optuna can search the hyperparameter space efficiently. 

If we have two hyperparameters, each with five possible values, the search space contains 5 × 5 = 25 possible combinations. Grid search automatically tests every possible combination until the whole space is covered, and then we choose the best one. This process is very time-consuming, because the number of combinations grows exponentially with the number of hyperparameters.
In random search, only a random sample of combinations is tested. It is faster than grid search, but it doesn't learn from previous results and may waste time searching unpromising regions of the space.

A third approach is Bayesian optimization, in particular the Tree-structured Parzen Estimator (TPE), which is the default algorithm in Optuna. ==It begins by sampling random combinations and, as more results are collected, it refines its guesses by focusing on regions that are likely to yield better results.==

- Uses results from previous trials
- Focuses on regions with high potential
- Powerful for large search spaces

The 5 hyperparameters we are gonna use are the one previously defined: `n_layers`, `n_filters`, `kernel_sizes`, `dropout_rate`, and `fc_size`

To define the search space, we declare the hyperparameters inside the objective function. Using the `trial` object, you can specify, for example, an integer range from 1 to 3. Each time the objective function runs, a new value within this range is selected.

```python
def objective(trial, device):

    # Feature extractor parameters
    n_layers = trial.suggest_int("n_layers", 1, 3)

    # list of filter counts between 16 and 128, one per layer
    n_filters = [
        trial.suggest_int(f"n_filters_{i}", 16, 128)
        for i in range(n_layers)
    ]

    # only 3 or 5 for each layer
    kernel_sizes = [
        trial.suggest_categorical(f"kernel_size_{i}", [3, 5])
        for i in range(n_layers)
    ]

    # Classifier parameters
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    fc_size = trial.suggest_int("fc_size", 64, 256)
```

By defining these parameters with the `trial.suggest_*` methods, you create a dynamic search space. Each time Optuna calls the objective function it will make smarter selections based on prior results.

In this case the learning rate is fixed, but if we wanted to search for its value too, we could make it optimizable:
```python
learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
```

`log=True` samples the values on a logarithmic scale, which suits the learning rate because it spans several orders of magnitude.

In a standard (single-objective) study, Optuna optimizes one metric, which in this case is validation accuracy.

With your objective function ready, the next step is to create an Optuna study and initiate the optimization process. An Optuna study records all tried hyperparameter combinations, the outcome of each trial, and identifies the best configuration.

```python
# Create a study object and optimize the objective function
study = optuna.create_study(direction='maximize') # maximize accuracy

# Start the optimization process (it takes about 8 minutes for 20 trials)
n_trials = 20
study.optimize(lambda trial: objective(trial, device), n_trials=n_trials) 
```

The `study.optimize` function executes the objective function 20 times, each time with a different set of hyperparameters proposed by Optuna.

A lambda function wraps the call to `objective`: `study.optimize` only passes the `trial` argument, so the lambda lets us also pass the `device` (CPU or CUDA).

One of the most informative visualizations is the optimization history plot which shows the accuracy obtained in each trial and its progression across trials.

![[optimization_history.png|559]]

Another powerful Optuna feature is its ability to determine and visualize each hyperparameter's relative importance.

![[hyperpar_importance.png|558]]

A practical strategy is to optimize one hyperparameter at a time: find its best value and fix it, which reduces the search space for future studies. Repeat this process, fixing a new hyperparameter each time (the importance plot helps you decide which one to start with).

Optuna offers also a **parallel coordinate** plot, which helps to visualize how combinations of hyperparameters relate to the objective value. The hyperparameters are on the x-axis, and the color bar represents accuracy (darker means higher).

![[pararrel_plot.png|546]]

Each line corresponds to one trial. The best way to read this graph is to look for clusters of dark lines converging on specific values on each vertical axis (hyperparameter). These clusters point to the hyperparameter configurations that tend to produce the best accuracy. (For `n_layers`, you can see the best value is 3.00.)

```python
# Extract the dataframe with the results
df = study.trials_dataframe()
df

# Extract and print the best trial
trial = study.best_trial
print("Best trial:")
print(f" Value (Accuracy): {trial.value:.4f}")
print(" Hyperparameters:")

for key, value in trial.params.items():
	print(f" {key}: {value}")

# Print the best hyperparameters
print("Best hyperparameters:")
print(study.best_params)
```

Once the Optuna study finishes, you can extract the values that maximize the objective using the `best_trial` attribute.

```python
Best trial:
Value (Accuracy): 0.5560
Hyperparameters:
{'dropout_rate': 0.441682492142502,
'fc_size': 216,
'kernel_size_0': 3,
'kernel_size_1': 3,
'kernel_size_2': 5,
'n_filters_0': 75,
'n_filters_1': 124,
'n_filters_2': 98,
'n_layers': 3}
```

You can then use these parameters to instantiate the flexible CNN and retrain it on the complete training and validation sets.