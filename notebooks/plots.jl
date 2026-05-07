# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Julia 1.12.6
#     language: julia
#     name: julia-1.12
# ---

# %% [markdown]
# # MCMC post-processing

# %%
begin
    import Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()

    using ArviZ
    using CairoMakie
    using MCMCChains
    using MCMCDiagnosticTools
    using PairPlots
    using StatsPlots
    using NCDatasets
    using Statistics
    using DataFrames
end

# %%
StatsPlots.default(fmt = :svg, dpi = 300)

# %% [markdown]
# ## Loading chains

# %%
filepath = "chains-H0-γ-κ-zpeak.nc"

function abspath_from_root_dir(filename::AbstractString)
    return (realpath ∘ joinpath)(@__DIR__, "..", filename)
end
netcdf_path = abspath_from_root_dir(filepath)

# %%
begin
    isfile(netcdf_path) ||
        throw(ArgumentError("NetCDF file not found: $(repr(netcdf_path))"))
    idata = from_netcdf(netcdf_path)
end

# %%
begin
    function to_mcmcchains(
            idata;
            internals_name_map::AbstractDict{Symbol, Symbol} = Dict{Symbol, Symbol}()
    )
        # Build MCMCChains.Chains from posterior + compatible sample_stats arrays.
        # MCMCChains expects (draws, parameters, chains), while each InferenceData
        # variable is stored as (draw, chain).
        hasproperty(idata, :posterior) ||
            throw(ArgumentError("InferenceData is missing `posterior` group"))

        post = idata.posterior
        post_syms = collect(propertynames(post))
        isempty(post_syms) &&
            throw(ArgumentError("`posterior` group has no variables"))

        first_arr = Array(getproperty(post, first(post_syms)))
        ndims(first_arr) == 2 ||
            throw(ArgumentError("Posterior variables must be 2D (draw, chain)"))
        n_draw, n_chain = size(first_arr)

        param_vals = zeros(Float64, n_draw, length(post_syms), n_chain)
        for (i, s) in enumerate(post_syms)
            arr = Array(getproperty(post, s))
            size(arr) == (n_draw, n_chain) ||
                throw(ArgumentError("Posterior variable $(s) has shape $(size(arr)); expected ($(n_draw), $(n_chain))"))
            param_vals[:, i, :] = Float64.(arr)
        end

        internal_syms = Symbol[]
        internal_blocks = Array{Float64, 3}[]
        if hasproperty(idata, :sample_stats)
            stats = idata.sample_stats
            for s in propertynames(stats)
                arr = Array(getproperty(stats, s))
                if ndims(arr) == 2 && size(arr) == (n_draw, n_chain)
                    mapped = get(internals_name_map, s, s)
                    if mapped in post_syms || mapped in internal_syms
                        throw(ArgumentError("Mapped internal name $(mapped) conflicts with an existing parameter/internal name"))
                    end
                    push!(internal_syms, mapped)
                    push!(internal_blocks, reshape(Float64.(arr), n_draw, 1, n_chain))
                end
            end
        end

        if isempty(internal_syms)
            return Chains(param_vals, post_syms, (parameters = post_syms,))
        end

        internal_vals = cat(internal_blocks...; dims = 2)
        full_vals = cat(param_vals, internal_vals; dims = 2)
        parameter_names = vcat(post_syms, internal_syms)
        section_map = (parameters = post_syms, internals = internal_syms)
        return Chains(full_vals, parameter_names, section_map)
    end

    internals_name_map = Dict(
        :energy => :hamiltonian_energy,
        :energy_error => :hamiltonian_energy_error
    )
    chain = to_mcmcchains(idata; internals_name_map = internals_name_map)
end

# %%
chain_params = names(chain, :parameters)

# %%
begin
    function has_var(dataset, name::Symbol)
        return name in propertynames(dataset)
    end

    function variable_array(dataset, name::Symbol)
        return Array(getproperty(dataset, name))
    end

    function numeric_summary(x)
        values = collect(skipmissing(vec(Float64.(Array(x)))))
        isempty(values) &&
            return (mean = missing, min = missing, median = missing, max = missing)
        return (
            mean = mean(values),
            min = minimum(values),
            median = median(values),
            max = maximum(values)
        )
    end

    function sample_stats_diagnostics(idata)
        hasproperty(idata, :sample_stats) || return "sample_stats group not present"
        stats = idata.sample_stats
        return (
            variables = propertynames(stats),
            divergence_count = has_var(stats, :diverging) ?
                               count(!iszero, vec(variable_array(stats, :diverging))) :
                               missing,
            acceptance_rate = has_var(stats, :acceptance_rate) ?
                              numeric_summary(variable_array(stats, :acceptance_rate)) :
                              missing,
            tree_depth = has_var(stats, :tree_depth) ?
                         numeric_summary(variable_array(stats, :tree_depth)) : missing,
            n_steps = has_var(stats, :n_steps) ?
                      numeric_summary(variable_array(stats, :n_steps)) : missing,
            bfmi = has_var(stats, :energy) ?
                   ArviZ.bfmi(variable_array(stats, :energy); dims = 1) : missing
        )
    end

    function pairplot_truth_namedtuple(idata, chain_params)
        hasproperty(idata, :constant_data) || return nothing
        cd = idata.constant_data
        cols = Symbol[p for p in chain_params if p in propertynames(cd)]
        isempty(cols) && return nothing
        return (; (p => Float64(only(Array(getproperty(cd, p)))) for p in cols)...)
    end

    nothing
end

# %% [markdown]
# ## Data

# %%
chain_params

# %% [markdown]
# ## Diagnostics

# %%
describe(chain)

# %% [markdown]
# ## Trace and autocorrelation plots

# %%
traceplot(chain)

# %%
autocorplot(chain; maxlag = 100)

# %%
meanplot(chain)

# %%
function _ensure_stats_array(stats, name::Symbol)
    hasproperty(stats, name) || return nothing
    A = Array(getproperty(stats, name))
    ndims(A) == 1 && return reshape(A, length(A), 1)
    return A
end

function _moving_average(y::AbstractVector, window::Int)
    window <= 1 && return Float64.(y)

    values = Float64.(y)
    prefix = cumsum(vcat(0.0, values))
    half_window = window ÷ 2

    return map(eachindex(values)) do i
        lo = max(firstindex(values), i - half_window)
        hi = min(lastindex(values), lo + window - 1)
        lo = max(firstindex(values), hi - window + 1)
        (prefix[hi + 1] - prefix[lo]) / (hi - lo + 1)
    end
end

function _plot_divergences!(ax, A)
    for ch in axes(A, 2)
        draws = findall(!iszero, A[:, ch])
        isempty(draws) && continue
        Makie.scatter!(
            ax, draws, fill(ch, length(draws)); color = (:red, 0.8), markersize = 4)
    end

    Makie.ylims!(ax, 0.5, size(A, 2) + 0.5)
    ax.ylabel = "chain"
    return nothing
end

function _plot_traces!(ax, A, sym::Symbol; colors, smooth_window::Int, draw_stride::Int)
    stride = clamp(draw_stride, 1, size(A, 1))
    plot_min, plot_max = Inf, -Inf

    for ch in axes(A, 2)
        x = collect(axes(A, 1))
        y = Float64.(A[:, ch])

        if sym == :step_size
            keep = y .> 0
            x, y = x[keep], y[keep]
            isempty(y) && continue
            plot_min = min(plot_min, minimum(y))
            plot_max = max(plot_max, maximum(y))
        end

        color = colors[mod1(ch, length(colors))]
        sym == :step_size || Makie.lines!(ax, x[1:stride:end], y[1:stride:end];
            color = RGBA(color, 0.25), linewidth = 0.8)
        Makie.lines!(ax, x, _moving_average(y, smooth_window);
            color = RGBA(color, 1.0), linewidth = 2, label = "chain $(ch)")
    end

    size(A, 2) > 1 && axislegend(ax; position = :rb, framevisible = false)

    if sym == :step_size && isfinite(plot_min) && isfinite(plot_max)
        lo = max(plot_min / sqrt(10), floatmin(Float64))
        hi = max(plot_max * sqrt(10), lo * 10)
        Makie.ylims!(ax, lo, hi)
        ax.yscale = log10
    end

    return nothing
end

function plot_sampler_diagnostics(
        idata; stats_syms = [:step_size, :acceptance_rate, :tree_depth, :diverging],
        figsize = (1000, 800), smooth_window::Int = 25, draw_stride::Int = 5)
    hasproperty(idata, :sample_stats) ||
        throw(ArgumentError("`idata` has no `sample_stats` group"))

    cols = 2
    fig = Figure(; size = figsize)
    colors = Makie.to_colormap(:Set1_9)

    for (i, sym) in enumerate(stats_syms)
        ax = Axis(fig[cld(i, cols), mod1(i, cols)]; title = string(sym), xlabel = "draw")
        A = _ensure_stats_array(idata.sample_stats, sym)
        if A === nothing
            Makie.text!(ax, 0.5, 0.5, "missing", align = (:center, :center))
        elseif sym == :diverging
            _plot_divergences!(ax, A)
        else
            _plot_traces!(ax, A, sym; colors, smooth_window, draw_stride)
        end
    end

    return fig
end

plot_sampler_diagnostics(idata)


# %%
energyplot(chain)

# %% [markdown]
# ## Posterior distributions

# %%
begin
    truth_nt = pairplot_truth_namedtuple(idata, chain_params)
    if length(chain_params) >= 2
        if truth_nt !== nothing
            pairplot(chain, PairPlots.Truth(truth_nt; label = "Truth"))
        else
            pairplot(chain)
        end
    else
        StatsPlots.density(chain)
    end
end
