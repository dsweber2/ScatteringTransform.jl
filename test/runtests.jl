# tests for the various forms of stParallel for the ScatteringTransform
using Distributed
addprocs(min((Sys.CPU_THREADS)-2-nprocs(), 2))
@everywhere using Interpolations, ContinuousWavelets, LinearAlgebra
@everywhere using FFTW
@everywhere using ScatteringTransform
@everywhere using SharedArrays
using Test
using Flux, FourierFilterFlux, CUDA, AbstractFFTs, LinearAlgebra, Statistics
using Zygote

include("planTests.jl")
include("cwtTests.jl")
include("pathTests.jl")
include("fluxtest.jl")
