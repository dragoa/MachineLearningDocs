##### Definition

**Optimizer** An optimizer updates the model's weights and biases using gradients calculated by backpropagation, with the goal of minimizing the loss function.

```
Optimizer
├─ Gradient Descent
│   ├─ Stochastic Gradient Descent (SGD)
│   ├─ Mini-Batch Gradient Descent
│   └─ SGD with Momentum
└─ Adaptive Gradient Descent
    ├─ AdaDelta
    ├─ RMSProp
    └─ Adam
```

- **Gradient Descent**: adjusts weights and biases based on a gradient to minimize the loss function.
- **RMSProp** (Root Mean Squared Propagation): adjusts the learning rate for each parameter individually, using a moving average of recent gradients.
- **Adam** (Adaptive Moment Estimation): one of the most widely used optimizers. It uses estimates of the first and second moments of gradients to adapt updates for each parameter. It is straightforward to use, but its moment estimates require more memory than plain SGD.

**Batch size** The number of samples processed before updating the model's internal parameters. Hardware limitations like RAM and GPU capacity can dictate your choice here.

- **Smaller batch size** → uses less memory and produces noisier gradient estimates; it may take longer to process the full dataset.
- **Larger batch size** → produces less noisy gradient estimates and can improve hardware throughput, but requires more memory and may affect generalization and convergence.

==Starting with a batch size of 32 or 64== can be a useful experiment, but the best choice depends on the dataset, model, and available hardware.

Look also: 
[[Omptimizer & Gradients]]
[[Learning Rate Schedulers]]
