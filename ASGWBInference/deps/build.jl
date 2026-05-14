using PackageCompiler

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const PACKAGE_DIR = joinpath(REPO_ROOT, "ASGWBInference")
const APP_DIR = joinpath(PACKAGE_DIR, "build", "asgwb")
const PRECOMPILE_FILE = joinpath(PACKAGE_DIR, "deps", "precompile_app.jl")
const CPU_TARGET = get(ENV, "ASGWB_CPU_TARGET", "x86-64")

create_app(
    PACKAGE_DIR,
    APP_DIR;
    executables = ["asgwb" => "julia_main"],
    precompile_execution_file = PRECOMPILE_FILE,
    force = true,
    cpu_target = CPU_TARGET
)
