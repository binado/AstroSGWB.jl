module AstroSGWBInference

include("InferenceImpl.jl")
using .InferenceImpl:
                      build_turing_model,
                      condition_turing_model,
                      logposterior,
                      validate_hyperprior

export build_turing_model,
       condition_turing_model,
       logposterior,
       validate_hyperprior,
       atomic_save_chain,
       MCMCConfig,
       SamplerConfig,
       load_config,
       save_config,
       validate_fiducials

include("chain_io.jl")
include("config.jl")
include("cli/stack_partial_chains.jl")
using .ChainIO: atomic_save_chain
using .Config: MCMCConfig, SamplerConfig, load_config, save_config, validate_fiducials

end
