# Machine Learning & PyTorch Notes

A practical collection of notes on machine learning and deep learning with PyTorch. Start with the fundamentals, follow the model-building workflow, then explore how to evaluate and improve models.

## Suggested learning path

1. Understand [[Getting Started with PyTorch/Why PyTorch?|why PyTorch]] and the [[Getting Started with PyTorch/Building Blocks of Neural Networks|building blocks of neural networks]].
2. Learn how to work with [[Getting Started with PyTorch/Tensor|tensors]], [[Getting Started with PyTorch/Activation Functions|activation functions]], and [[Getting Started with PyTorch/Loss Function|loss functions]].
3. Follow the [[Getting Started with PyTorch/PyTorch Data Pipeline|data pipeline]] and [[Getting Started with PyTorch/PyTorch Model Pipeline|model pipeline]] to prepare data, train a model, and evaluate it.
4. Explore [[Hyperparameter Optimization/Hyperparameter Optimization|hyperparameter optimization]], its [[Hyperparameter Optimization/Metrics/Accuracy|evaluation metrics]], and strategies for tuning models.

## Getting started with PyTorch

Foundational notes on PyTorch, neural networks, data, and training:

- [[Getting Started with PyTorch/Why PyTorch?|Why PyTorch?]] — What makes PyTorch's dynamic, Python-like approach useful for building and debugging deep learning models.
- [[Getting Started with PyTorch/Tensor|Tensor]] — Tensor shapes and data types, creating and reshaping tensors, indexing, math, and broadcasting.
- [[Getting Started with PyTorch/Building Blocks of Neural Networks|Building Blocks of Neural Networks]] — Neurons, how they learn, and how they combine into layers and networks.
- [[Getting Started with PyTorch/Activation Functions|Activation Functions]] — ReLU, sigmoid, and tanh, and how nonlinear activations help neural networks learn complex patterns.
- [[Getting Started with PyTorch/Machine Learning Pipelines|Machine Learning Pipelines]] — The stages from data ingestion through deployment, illustrated with a simple neural-network example.
- [[Getting Started with PyTorch/PyTorch Data Pipeline|PyTorch Data Pipeline]] — Using transforms, datasets, and data loaders to prepare and serve data in batches.
- [[Getting Started with PyTorch/PyTorch Model Pipeline|PyTorch Model Pipeline]] — Defining models with `nn.Module`, writing training loops, and evaluating model performance.
- [[Getting Started with PyTorch/Loss Function|Loss Function]] — How loss measures prediction error, with examples including mean squared error and cross-entropy.
- [[Getting Started with PyTorch/Omptimizer & Gradients|Optimizer & Gradients]] — A starting point for optimizer and gradient notes, with links to Adam, SGD, and optimization topics.

## Hyperparameter optimization

Notes on choosing model and training settings, measuring results, and improving performance:

- [[Hyperparameter Optimization/Hyperparameter Optimization|Hyperparameter Optimization]] — An overview of evaluation metrics, schedulers, tuning, flexible architectures, and model efficiency.
- [[Hyperparameter Optimization/Flexible CNN Architecture|Flexible CNN Architecture]] — Building a configurable CNN and using Optuna to search for useful hyperparameter settings.
- [[Hyperparameter Optimization/Learning Rate Schedulers|Learning Rate Schedulers]] — StepLR, ReduceLROnPlateau, and CosineAnnealingLR, with guidance on when to step each scheduler.
- [[Hyperparameter Optimization/Real World Examples|Real World Examples]] — Applying classification metrics and model-selection considerations to medical diagnosis, spam, fraud, and image classification.

### Hyperparameters

- [[Hyperparameter Optimization/Hyperparameters/Architectural Hyperparameters|Architectural Hyperparameters]] — Tuning network depth, layer width, filters, and activation functions.
- [[Hyperparameter Optimization/Hyperparameters/Training Hyperparameters|Training Hyperparameters]] — Optimizers and batch size, including their effects on training and convergence.
- [[Hyperparameter Optimization/Hyperparameters/Regularization|Regularization]] — Techniques for reducing overfitting, including weight decay, dropout, early stopping, and batch normalization.
- [[Hyperparameter Optimization/Hyperparameters/Weight decay|Weight decay]] — How an L2 penalty discourages large model weights and how to set it in a PyTorch optimizer.

### Evaluation metrics

- [[Hyperparameter Optimization/Metrics/Accuracy|Accuracy]] — The proportion of predictions that are correct.
- [[Hyperparameter Optimization/Metrics/Precision|Precision]] — How many positive predictions are correct; useful when false positives are costly.
- [[Hyperparameter Optimization/Metrics/Recall|Recall]] — How many actual positive cases are found; useful when false negatives are costly.
- [[Hyperparameter Optimization/Metrics/F1 Score|F1 Score]] — A harmonic-mean balance of precision and recall, especially useful for imbalanced datasets.

## PyTorch reference notes

Short notes and entry points for core PyTorch building blocks:

- [[PyTorch Docs/PyTorch Docs|PyTorch Docs]] — An entry point to the notes on PyTorch modules, datasets, transforms, and data loaders.
- [[PyTorch Docs/PyTorch Module|PyTorch Module]] — Reference note for PyTorch's `nn.Module` model-building abstraction.
- [[PyTorch Docs/PyTorch Dataset|PyTorch Dataset]] — Reference note for representing and retrieving individual data samples.
- [[PyTorch Docs/PyTorch DataLoader|PyTorch DataLoader]] — Reference note for batching and iterating over dataset samples.
- [[PyTorch Docs/PyTorch Transforms|PyTorch Transforms]] — Reference note for preprocessing and transforming data.
