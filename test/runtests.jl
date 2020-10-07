# tests for the various forms of stParallel for the ScatteringTransform
using Revise; using Distributed
addprocs(min((Sys.CPU_THREADS)-2-nprocs(), 2))
@everywhere using Revise
@everywhere using Interpolations, ContinuousWavelets, LinearAlgebra
@everywhere using FFTW
@everywhere using ScatteringTransform
@everywhere using SharedArrays
using Test


include("planTests.jl")
include("cwtTests.jl")
include("pathTests.jl")
