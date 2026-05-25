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

    using CairoMakie
    using JLD2
    using PairPlots
    using Statistics
    using Turing
    using FlexiChains
    using FlexiChains: Extra, FlexiChain
    using ASGWB
end

# %%
output_dir = joinpath(@__DIR__, "..", get(ENV, "ASGWB_FIGURES_DIR", "output-test-figures"))
const FIGURE_DPI = 300

# Makie figure sizes are CSS pixels; 1 in = 96 CSS px (see Makie docs).
const MAKIE_CSS_PX_PER_INCH = 96
const MAKIE_DEFAULT_PT_PER_UNIT = 0.75

function _makie_save_kwargs(dpi::Int)
    scale = dpi / MAKIE_CSS_PX_PER_INCH
    return (; px_per_unit = scale, pt_per_unit = MAKIE_DEFAULT_PT_PER_UNIT * scale)
end

function _save_plot_object!(obj, path::AbstractString; dpi::Int)
    return save(path, obj; _makie_save_kwargs(dpi)...)
end

function save_figure(
        fig,
        name::AbstractString;
        output_dir::Union{Nothing, AbstractString} = output_dir,
        dpi::Int = FIGURE_DPI
)
    output_dir === nothing && return fig
    mkpath(output_dir)
    stem = joinpath(output_dir, name)
    try
        _save_plot_object!(fig, stem * ".pdf"; dpi)
    catch err
        @warn "PDF export failed; saving PNG instead" name exception = err
        _save_plot_object!(fig, stem * ".png"; dpi)
    end
    return fig
end

# %%
Base.@kwdef struct PlotConfig
    palette::Vector{Makie.RGBAf} = Makie.wong_colors()
    primary_color::Makie.RGBAf = Makie.wong_colors()[1]
    alpha::Float64 = 0.5
    linewidth::Float64 = 1.2
    strokewidth::Float64 = 2.0
    smooth_window::Int = 25
    draw_stride::Int = 5
end

const PLOT_CONFIG = PlotConfig()

function chain_colors(cfg::PlotConfig, n::Integer; alpha::Real = cfg.alpha)
    return [(cfg.palette[mod1(i, length(cfg.palette))], alpha) for i in 1:n]
end

# %% [markdown]
# ## Loading chains

# %%
filepath = get(ENV, "ASGWB_CHAIN_FILE", "chains/chains-H0-seed13-20260508-183716-slim-flexi.jld2")

chain_path = (realpath ∘ joinpath)(@__DIR__, "..", filepath)

# %%
function _load_chain(path::AbstractString)
    isfile(path) || throw(ArgumentError("JLD2 file not found: $(repr(path))"))
    data = load(path)
    if haskey(data, "chain")
        return data["chain"]
    elseif haskey(data, "snapshot")
        return data["snapshot"]
    else
        throw(ArgumentError(
            "JLD2 file contains neither 'chain' nor 'snapshot' key: $(repr(path))",
        ))
    end
end

# %%
chain = _load_chain(chain_path)

# %%
chain_params = FlexiChains.parameters(chain)

# %% [markdown]
# ## Data

# %%
chain_params

# %% [markdown]
# ## Diagnostics

# %%
summarystats(chain)

# %% [markdown]
# ## Trace and autocorrelation plots

# %%
begin
    fig = FlexiChains.mtraceplot(chain; color = chain_colors(PLOT_CONFIG, size(chain, 2)))
    save_figure(fig, "traceplot")
    fig
end

# %%
begin
    n_draws = size(chain, 1)
    autocor_maxlag = min(100, max(1, n_draws - 1))
    fig = FlexiChains.mautocorplot(
        chain; lags = 1:autocor_maxlag, color = chain_colors(PLOT_CONFIG, size(chain, 2)))
    save_figure(fig, "autocorplot")
    fig
end

# %%
begin
    fig = FlexiChains.mmeanplot(chain; color = chain_colors(PLOT_CONFIG, size(chain, 2)))
    save_figure(fig, "meanplot")
    fig
end

# %%
function _ensure_internal_array(chain::FlexiChain, name::Symbol)
    key = Extra(name)
    key in keys(chain) || return nothing
    return Array(chain[key])
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
            color = Makie.RGBAf(color, 0.25), linewidth = 0.8)
        Makie.lines!(ax, x, _moving_average(y, smooth_window);
            color = Makie.RGBAf(color, 1.0), linewidth = 2, label = "chain $(ch)")
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
        chain::FlexiChain;
        cfg::PlotConfig = PLOT_CONFIG,
        stats_syms = [:step_size, :acceptance_rate, :tree_depth, :numerical_error],
        figsize = (1000, 800))
    cols = 2
    fig = Figure(; size = figsize)
    colors = cfg.palette

    for (i, sym) in enumerate(stats_syms)
        ax = Axis(fig[cld(i, cols), mod1(i, cols)]; title = string(sym), xlabel = "draw")
        A = _ensure_internal_array(chain, sym)
        if A === nothing
            Makie.text!(ax, 0.5, 0.5, "missing", align = (:center, :center))
        elseif sym in (:diverging, :numerical_error)
            _plot_divergences!(ax, A)
        else
            _plot_traces!(ax, A, sym;
                colors, smooth_window = cfg.smooth_window, draw_stride = cfg.draw_stride)
        end
    end

    return fig
end

begin
    fig = plot_sampler_diagnostics(chain)
    save_figure(fig, "sampler_diagnostics")
    fig
end

# %%
begin
    energy_key = Extra(:hamiltonian_energy)
    if energy_key in keys(chain)
        fig = FlexiChains.mtraceplot(
            chain, energy_key; color = chain_colors(PLOT_CONFIG, size(chain, 2)))
        save_figure(fig, "energyplot")
        fig
    else
        @info "skipping energyplot: $(energy_key) not present in chain"
        nothing
    end
end

# %% [markdown]
# ## Posterior distributions

# %%
begin
    fig = if length(chain_params) >= 2
        pairplot(
            PairPlots.Series(chain; label = "posterior",
                color = PLOT_CONFIG.primary_color) => (
                PairPlots.Contour(color = PLOT_CONFIG.primary_color),
                PairPlots.Contourf(color = (PLOT_CONFIG.primary_color, PLOT_CONFIG.alpha)),
                PairPlots.MarginDensity(
                    color = (PLOT_CONFIG.primary_color, PLOT_CONFIG.alpha),
                    strokecolor = PLOT_CONFIG.primary_color,
                    strokewidth = PLOT_CONFIG.strokewidth
                )
            );
            pool_chains = true
        )
    else
        Makie.density(chain;
            pool_chains = true,
            legend_position = :none,
            color = (PLOT_CONFIG.primary_color, PLOT_CONFIG.alpha),
            strokecolor = PLOT_CONFIG.primary_color,
            strokewidth = PLOT_CONFIG.strokewidth
        )
    end
    save_figure(fig, length(chain_params) >= 2 ? "pairplot" : "posterior_density")
    fig
end
