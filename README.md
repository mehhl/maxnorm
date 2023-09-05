### maxnorm.py

[This](https://github.com/mehlsson/maxnorm/blob/main/mnwrapper.py) is a tiny wrapper adding a max norm constraint as discussed in [Hinton et al. (2012)](https://arxiv.org/pdf/1207.0580.pdf) and [Srivastava et al. (2014)](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf) for PyTorch (I've used it on PyTorch 1.13.1) that wraps it around a PyTorch optimizer, in this case `torch.optim.SGD`. 

##### Usage

You can use `MNSGD` as a drop-in replacement for `SGD`, but you do need to pass it params as an iterable of dicts, and you need to pass `'max_norm'` as keyword for each parameter group to which you want to apply the max-norm constraint, like this:

```python
model = NeuralNetwork(dropout_p = 0.5).to(device)

loss_fn = nn.CrossEntropyLoss()
model = NeuralNetwork().to(device)
optimizer = MNSGD(
    [
        {'params': model.hidden.parameters(), 'max_norm': 2 }, 
        {'params': model.output.parameters() }
    ],
    lr=0.5,
    momentum=0.5
)
```

See [PyTorch's doc for `torch.optim.Optimizer`](https://pytorch.org/docs/stable/optim.html#per-parameter-options) for more. The result of the above is that parameters matrices contained in `model.hidden` will get clipped, and parameter matrix in `model.output` won't be. Note that this is something of a hack that relies on `torch.optim.Optimizer` accepting keyword arguments that don't match what it internally uses.

Here is an [example](https://github.com/mehlsson/maxnorm/blob/main/example.ipynb) of use that includes a test of correctness.

##### Rationale

The max-norm constraint is essentially an additional normalization constraint applied after each optimization steps; the optimizer checks for each unit if its incoming weight vector $ {\boldsymbol w} $ has some $p$-norm $ \lVert {\boldsymbol w} \rVert_p $ greater than or equal than some scalar max norm, and normalizes it if it does not. From what I've seen, its use is mostly associated with the use of the dropout technique. This constraint can be used out-of-the-box in Keras (as `tf.keras.constraint.MaxNorm`), but I couldn't find any out-of-the-box implementation for PyTorch.

##### To-do in near future

Add other optimizers and support for non-iterable `params` argument to the optimizer constructor. Run more tests, including with CUDA.
