# ScatteringTransform.jl
An implementation of the generalized scattering transform in Julia. Documentation in progress, as are updates to 1.0

## Installation
Basic installation: at the moment, this is not an official package. Because of this, to install it, you must first clone it, then make it accessible to your project.

### Dependencies
This uses a modified version of Wavelets.jl that can be found [here](https://github.com/dsweber2/Wavelets.jl), and added via `Pkg.develop("https://github.com/dsweber2/Wavelets.jl.git")`.

### Installation

 In 0.7 onwards, use either `Pkg.add("https://github.com/dsweber2/ScatteringTransform.jl.git")`, `Pkg.develop("https://github.com/dsweber2/ScatteringTransform.jl.git")`, or `] add https://github.com/dsweber2/ScatteringTransform.jl.git` in a REPL. Then from a Repl
```
  (v0.7) pkg> activate .
  (ScatteringTransform) pkg> instantiate
```

## Basic Usage

This implementation works strictly on 1 dimensional data and uses Morlet wavelets. If you give it data that has more dimensions, it will transform along the last dimension. There are 2 steps to applying a scattering transform. The first is constructing the transform, done with `layeredTransform(m,example)`. Then you need to actually transform the data; if you are investigating a single example, use `st(f, layers)` to get a `scattered`, a type containing both the intermediate results and the output of each layer. On the other hand, if you're transforming a large set, and only want the outputs, possibly highly subsampled, use `thinSt`.

For more detailed description, see the file [basicUsage.jl](basicUsage.jl)
