# tests for the various forms of layeredTransform for the ScatteringTransform
using Distributed
addprocs((Sys.CPU_THREADS)-2-nprocs())
@everywhere using Interpolations, Wavelets, LinearAlgebra
@everywhere using FFTW
@everywhere using ScatteringTransform
@everywhere using SharedArrays
using Test


include("planTests.jl")
include("cwtTests.jl")
include("pathTests.jl")
