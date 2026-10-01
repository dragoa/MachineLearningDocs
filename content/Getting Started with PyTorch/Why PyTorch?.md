[[PyTorch]] is a deep learning framework where writing and running code feels a lot like regular Python. Adding two numbers can be very simple:

```python
a = torch.tensor(2.0)
b = torch.tensor(3.0)

result = a + b
print(result)
```

[[Tensor|Tensors]] are PyTorch's way of storing numbers and data, but the important idea is that, in today's world, ==machine learning and deep learning can feel a lot like regular programming==, but this wasn't always the case.

[[Machine Learning]] works very differently from traditional programming.
Let's say you want to build a **[[Recomanding System|recommendation system]]** that suggests products to customers.
In traditional programming, you explicitly write the rules:

> _If the customer buys a camera, recommend some lenses._

But what if the camera was a gift?
What if the customer already has five lenses?

You would need to create thousands of rules to handle different situations and exceptions, and you would probably still miss some. This is one of the limitations of traditional programming:
==You write the rules that transform inputs into outputs.==

```text
Input → Rules → Output
```

In [[Machine Learning]], instead, you provide the system with examples of inputs and outputs and the system then **learns the rules/patterns from the examples**.

```text
Input + Output examples → Machine Learning → Learned rules/model
```

[[Deep Learning]] takes this further by using [[Neural Network|neural networks]]. For examples, if you give a neural network customer purchase histories, it can learn patterns in what people tend to buy.

But there's a catch. To learn from thousands or even millions of examples, neural networks have to perform a **massive amount of mathematical computation**. Every example requires many calculations throughout the neural network. These calculations happen repeatedly while the network learns patterns. This can build up to **millions or even billions of mathematical operations** during training. 

Early deep learning frameworks were heavily focused on handling this large amount of computation, even simple operations could become unnecessarily complicated.
To do that ==they required a static computational graph.== 
You had to:

1. Define every operation ahead of time.
2. Build the computational graph.
3. Compile it.
4. Then run your data through it.

The problem was that once the graph was built, changing it wasn't straightforward.
For example, if you made a mistake or wanted to experiment with a different operation, you couldn't simply modify your Python code and continue.
You also couldn't easily test individual parts. You often had to construct the whole computational graph before running it.
Some frameworks also required specialized operators instead of normal Python:

- `if` statements
- `for` loops
- other control-flow logic

This made the code more complex and less intuitive.
Another limitation was that computational graphs could be restrictive about the **size and shape of inputs**.
Debugging was also difficult. When something went wrong, error messages pointed directly to internal framework code rather than where the problem originated in your own code.

As a result, developers could spend more time **fighting the framework** than working on the actual machine learning problem.
# PyTorch Approach

PyTorch was built around a simple idea: deep learning should feel like normal Python.
Instead of forcing you to work with a complicated static computational graph, ==PyTorch uses a dynamic approach to computation.==
==You can write normal Python code and PyTorch handles the complex mathematical underlying computations.==

This means you can use:
- Normal Python `if` statements
- Normal `for` loops
- Normal Python functions
- Regular debugging tools
- Variables that you can inspect while your program is running

You can also change your code and experiment much more naturally.
When something breaks, the error is generally much easier to trace back to the code you actually wrote.

Look next: [[Building Blocks of Neural Networks]]