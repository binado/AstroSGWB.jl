using Test
using AstroSGWB
using AstroSGWBInference

include(joinpath(@__DIR__, "..", "..", "AstroSGWB", "test", "parity_test_cache.jl"))
include("test_chain_io.jl")
include("test_config.jl")
include("test_hyperparameters.jl")
include("test_run_mcmc_script.jl")
include("test_turing.jl")
