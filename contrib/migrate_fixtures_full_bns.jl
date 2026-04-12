#!/usr/bin/env julia
# Convert redshift-only Julia HDF5 caches to full-BNS layout; refresh deterministic_parity posterior_case.
using ASGWB
using HDF5

function _as_string(x)
    x isa AbstractString && return String(x)
    x isa AbstractVector{UInt8} && return String(copy(x))
    return string(x)
end

function _float_scalar(group, key)
    haskey(group, key) || return nothing
    return Float64(read(group[key]))
end

function fid_from_src(src, spec::RedshiftPriorSpec)::ProposalFiducialParameters
    hg, sg = src["hyperparameters"], src["redshift_prior_spec"]
    H0 = Float64(read(hg["H0"]))
    Omega_m = Float64(read(hg["Omega_m"]))
    chi0 = Float64(read(hg["chi0"]))
    chin = Float64(read(hg["chin"]))
    if spec.family == MadauDickinson
        γ = something(
            something(_float_scalar(hg, "gamma"), _float_scalar(sg, "gamma")),
            2.7,
        )
        κ = something(
            something(_float_scalar(hg, "kappa"), _float_scalar(sg, "kappa")),
            3.0,
        )
        zp = something(
            something(_float_scalar(hg, "z_peak"), _float_scalar(sg, "z_peak")),
            2.5,
        )
        return ProposalFiducialParameters(; H0, Omega_m, chi0, chin, gamma=γ, kappa=κ, z_peak=zp)
    end
    λ = something(_float_scalar(hg, "lamb"), _float_scalar(sg, "lamb"))
    λ === nothing && throw(ArgumentError("PowerLaw cache needs lamb in hyperparameters or redshift_prior_spec"))
    return ProposalFiducialParameters(; H0, Omega_m, chi0, chin, lamb=λ)
end

function _flux_matrix(src)
    return Matrix{Float64}(permutedims(Array{Float64}(read(src["cached_flux_over_dgw2"]))))
end

function migrate_cache_file(src_path::AbstractString, dst_path::AbstractString)
    h5open(src_path, "r") do src
        attrs = attributes(src)
        z = vec(Float64.(read(src["proposal_samples/redshift"])))
        n = length(z)
        m1 = fill(1.4, n)
        m2 = fill(1.2, n)
        chi1 = zeros(n)
        chi2 = zeros(n)
        lam1 = fill(100.0, n)
        lam2 = fill(100.0, n)

        intrinsic_mat = hcat(m1, m2, z, chi1, chi2, lam1, lam2)
        flux = _flux_matrix(src)
        dgw = vec(Float64.(read(src["dgw_fid_sq"])))

        sg = src["redshift_prior_spec"]
        spec = RedshiftPriorSpec(
            parse_redshift_prior_family(_as_string(read(sg["family"]))),
            Float64(read(sg["z_min"])),
            Float64(read(sg["z_max"])),
            Int(read(sg["num_interp"])),
            haskey(sg, "time_delay_model") ? (let t = read(sg["time_delay_model"]); isempty(_as_string(t)) ? nothing : _as_string(t) end) : nothing,
        )
        fid = fid_from_src(src, spec)
        samples = FullBNSSamples(
            Vector(m1), Vector(m2), Vector(z), Vector(chi1), Vector(chi2), Vector(lam1), Vector(lam2),
        )
        lp = reconstruct_proposal_log_prob(samples, spec, fid)

        has_cov = haskey(src, "covariance") && haskey(src, "sgwb_scale")

        h5open(dst_path, "w") do dst
            a = attributes(dst)
            for name in (
                "format_name",
                "format_version",
                "local_merger_rate",
                "observation_time_sec",
                "observation_time_yr",
                "redshift_integral_fiducial",
            )
                haskey(attrs, name) || continue
                a[name] = read(attrs[name])
            end

            write(dst, "intrinsic_site_order", FULL_BNS_INTRINSIC_ORDER)
            write(dst, "proposal_intrinsic_vector", Matrix(permutedims(intrinsic_mat)))
            write(dst, "frequencies", vec(Float64.(read(src["frequencies"]))))
            write(dst, "in_band_mask", Vector{Bool}(vec(Bool.(read(src["in_band_mask"])))))
            if has_cov
                write(dst, "covariance", vec(Float64.(read(src["covariance"]))))
                write(dst, "sgwb_scale", vec(Float64.(read(src["sgwb_scale"]))))
            end
            write(dst, "cached_flux_over_dgw2", Matrix(permutedims(flux)))
            write(dst, "proposal_log_prob", Vector(lp))
            write(dst, "dgw_fid_sq", dgw)
            write(dst, "fiducial_spectral_density", vec(Float64.(read(src["fiducial_spectral_density"]))))

            g = create_group(dst, "proposal_samples")
            write(g, "mass_1_source", m1)
            write(g, "mass_2_source", m2)
            write(g, "redshift", z)
            write(g, "chi_1", chi1)
            write(g, "chi_2", chi2)
            write(g, "lambda_1", lam1)
            write(g, "lambda_2", lam2)
            attributes(g)[PROPOSAL_SAMPLES_SOURCE_TYPE_ATTR] = PROPOSAL_SAMPLES_SOURCE_TYPE_BNS

            hg = create_group(dst, "hyperparameters")
            hg_src = src["hyperparameters"]
            for k in keys(hg_src)
                write(hg, String(k), read(hg_src[k]))
            end
            if spec.family == MadauDickinson
                haskey(hg, "gamma") || write(hg, "gamma", fid.gamma::Float64)
                haskey(hg, "kappa") || write(hg, "kappa", fid.kappa::Float64)
                haskey(hg, "z_peak") || write(hg, "z_peak", fid.z_peak::Float64)
            end

            sg2 = create_group(dst, "redshift_prior_spec")
            for k in keys(sg)
                write(sg2, String(k), read(sg[k]))
            end
        end
    end
end

function refresh_posterior_case_parity!(parity_path::AbstractString, cache_path::AbstractString)
    cache = load_cache(cache_path)
    h5open(parity_path, "r+") do f
        g = f["posterior_case"]
        theta = HyperParameters((; (
            Symbol(name) => Float64(read(g["theta/$(name)"])) for
            name in ("H0", "Omega_m", "chi0", "chin", "gamma", "kappa", "z_peak")
        )...,))
        priors = build_uniform_priors(
            Dict(
                name => (
                    Float64(read(g["prior_bounds/$(name)/low"])),
                    Float64(read(g["prior_bounds/$(name)/high"])),
                ) for
                name in ("H0", "Omega_m", "chi0", "chin", "gamma", "kappa", "z_peak")
            ),
        )
        ev = evaluate_importance_terms(theta, cache)
        function _overwrite!(grp, name, data)
            haskey(grp, name) && HDF5.delete_object(grp, name)
            write(grp, name, data)
        end
        _overwrite!(g, "dgw_theta_sq", collect(ev.dgw_theta_sq))
        _overwrite!(g, "weights", collect(ev.weights))
        _overwrite!(g, "spectral_density_full", collect(ev.spectral_density))
        _overwrite!(g, "spectral_density_in_band", collect(ev.spectral_density_in_band))
        _overwrite!(g, "expected_number_of_sources", ev.expected_number_of_sources)
        _overwrite!(g, "log_ratio", collect(ev.log_ratio))
        _overwrite!(g, "target_log_prob", collect(ev.target_log_prob))
        _overwrite!(g, "redshift_integral", ev.redshift_integral)
        _overwrite!(g, "log_prior", logprior(theta, priors))
        _overwrite!(g, "log_likelihood", loglikelihood(theta, cache))
        _overwrite!(g, "log_posterior", logposterior(theta, cache, priors))
        _overwrite!(g, "normalized_ess", normalized_ess(ev.weights))
        _overwrite!(g, "max_normalized_weight", max_normalized_weight(ev.weights))
        _overwrite!(g, "log_ratio_variance", log_ratio_variance(ev.log_ratio))
    end
end

root = joinpath(@__DIR__, "..")
fixtures = joinpath(root, "test", "fixtures")
par = joinpath(fixtures, "deterministic_parity.h5")

for name in ("importance_context_julia.h5", "posterior_cache_julia.h5", "posterior_cache_julia_v2_minimal.h5")
    src = joinpath(fixtures, name)
    isfile(src) || continue
    tmp = tempname() * ".h5"
    try
        migrate_cache_file(src, tmp)
        mv(tmp, src; force=true)
        println("migrated ", name)
    catch
        rm(tmp; force=true)
        rethrow()
    end
end

refresh_posterior_case_parity!(par, joinpath(fixtures, "posterior_cache_julia.h5"))
println("refreshed deterministic_parity.h5 posterior_case")
