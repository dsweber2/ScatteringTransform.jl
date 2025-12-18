# ScatteringTransform
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://BoundaryValueProblems.github.io/ScatteringTransform.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://BoundaryValueProblems.github.io/ScatteringTransform.jl/dev)
[![Build Status](https://github.com/BoundaryValueProblems/ScatteringTransform.jl/actions/workflows/Documenter.yml/badge.svg)](https://github.com/BoundaryValueProblems/ScatteringTransform.jl/actions/workflows/Documenter.yml)
[![Codecov](https://codecov.io/gh/BoundaryValueProblems/ScatteringTransform.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/BoundaryValueProblems/ScatteringTransform.jl)
[![Coveralls](https://coveralls.io/repos/github/BoundaryValueProblems/ScatteringTransform.jl/badge.svg?branch=master)](https://coveralls.io/github/BoundaryValueProblems/ScatteringTransform.jl?branch=master)

An implementation of the generalized scattering transform in Julia. Documentation in progress.

## Installation
Basic installation: at the moment, this is not an official package. Because of this, to install it, you must first clone it, then make it accessible to your project.

### Dependencies
This uses Wai Ho Chak's `rotated_monogenic.jl` package that can be found [here](https://github.com/UCD4IDS/rotated_monogenic.jl), and added via `Pkg.develop("https://github.com/UCD4IDS/rotated_monogenic.jl.git")`.

### Installation

In 0.7 onwards, use either `Pkg.add("https://github.com/BoundaryValueProblems/ScatteringTransform.jl.git")`, `Pkg.develop("https://github.com/BoundaryValueProblems/ScatteringTransform.jl.git")`, or `] add https://github.com/BoundaryValueProblems/ScatteringTransform.jl.git` in a REPL. Then from a Repl
```julia
  (v0.7) pkg> activate .
  (ScatteringTransform) pkg> instantiate
```
either way, make sure you don't have more than a single thread when building by
setting `JULIA_NUM_THREADS = 1` before calling `Pkg.build`.

## Basic Usage

This implementation works strictly on 1 dimensional data and uses Morlet wavelets by default. If you supply data that has more dimensions, it will transform along the last dimension. For more detailed description, see the documentation (docs/basicUsage.jl)
