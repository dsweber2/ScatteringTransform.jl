# tests for the various forms of stParallel for the ScatteringTransform
# exit()
# using Revise
using ScatteringTransform
using ContinuousWavelets
using AbstractFFTs, FFTW
using Test, LinearAlgebra, Statistics
using Flux, FourierFilterFlux, CUDA
using Zygote

include("pathTests.jl")
include("fluxtests.jl")
