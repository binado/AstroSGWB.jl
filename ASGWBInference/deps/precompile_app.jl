using ASGWBInference

ENV["ASGWB_REPO_ROOT"] = normpath(joinpath(@__DIR__, "..", ".."))
ENV["MCMC_CONFIG_FILEPATH"] = "config/run_inference_smoke_h0.toml"

ASGWBInference.run_inference_from_env()
