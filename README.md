This is the repository for the paper "Automated Augmented Conjugate Inference for Non-Conjugate Gaussian Process Models"

You need [Julia 1.1](https://julialang.org/downloads/oldreleases.html) to run the experiments as well as [GPFlow](https://github.com/GPflow/GPflow) and [Tensorflow](https://www.tensorflow.org/).

All the source code is implemented in the package [AugmentedGaussianProcesses.jl](https://github.com/theogf/AugmentedGaussianProcesses.jl) which this experiment package relies on.

To install all the required packaged, go to the directory and run julia:
```
julia>] activate .
AutoConjugate> instantiate
```
This will install all the Julia packages needed for the experiments
You can then run one of the experiments from the paper:

For the sampling experiment run `include("scripts/sampling_exp.jl")`
For the hyperparameter experiment run `include("scripts/hyperparam_exp.jl")`
And for the convergene the code is described in `script/_conv_exp.jl`

There is a WIP to create a smart macro to create augmented likelihoods from a formula like `p(y|f,β) = 1/2β * exp(-sqrt(y^2-2*y*f+y^2)/β)`, it should arrive in the next weeks
