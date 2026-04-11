using Distributions

const BNS_MASS_LOW = 1.1
const BNS_MASS_HIGH = 2.5
const BNS_LAMBDA_HIGH = 5000.0
const BNS_SPIN_A_MAX = 0.99

function build_uniform_priors(
    bounds::AbstractDict{<:AbstractString,<:Tuple{<:Real,<:Real}},
)
    return InferencePriors(
        Uniform(Float64(bounds["H0"][1]), Float64(bounds["H0"][2])),
        Uniform(Float64(bounds["Omega_m"][1]), Float64(bounds["Omega_m"][2])),
        Uniform(Float64(bounds["chi0"][1]), Float64(bounds["chi0"][2])),
        Uniform(Float64(bounds["chin"][1]), Float64(bounds["chin"][2])),
        Uniform(Float64(bounds["gamma"][1]), Float64(bounds["gamma"][2])),
        Uniform(Float64(bounds["kappa"][1]), Float64(bounds["kappa"][2])),
        Uniform(Float64(bounds["z_peak"][1]), Float64(bounds["z_peak"][2])),
    )
end

function logprior(h::HyperParameters, priors::InferencePriors)
    pop = h.population
    pop isa MadauDickinsonParameters || throw(
        ArgumentError("logprior with InferencePriors requires MadauDickinsonParameters"),
    )
    return (
        logpdf(priors.H0, h.cosmological.H0) +
        logpdf(priors.Omega_m, h.cosmological.Omega_m) +
        logpdf(priors.chi0, h.propagation.chi0) +
        logpdf(priors.chin, h.propagation.chin) +
        logpdf(priors.gamma, pop.gamma) +
        logpdf(priors.kappa, pop.kappa) +
        logpdf(priors.z_peak, pop.z_peak)
    )
end

function ordered_uniform_source_masses_logprob(
    mass_1_source::AbstractVector{<:Real},
    mass_2_source::AbstractVector{<:Real};
    low::Real=BNS_MASS_LOW,
    high::Real=BNS_MASS_HIGH,
)
    length(mass_1_source) == length(mass_2_source) || throw(
        ArgumentError("mass_1_source and mass_2_source must have matching lengths"),
    )
    lp = log(2.0) - 2.0 * log(high - low)
    return [
        (m1 >= m2 && m2 >= low && m1 <= high) ? lp : -Inf for
        (m1, m2) in zip(mass_1_source, mass_2_source)
    ]
end

function aligned_spin_chi_simple_logprob(
    chi::AbstractVector{<:Real};
    a_max::Real=BNS_SPIN_A_MAX,
)
    eps_value = eps(Float64)
    tiny = floatmin(Float64)
    return [
        log(
            max(
                if abs(value) <= a_max
                    -log(max(abs(value), eps_value) / a_max) / (2.0 * a_max)
                else
                    0.0
                end,
                tiny,
            ),
        ) for value in chi
    ]
end

function bounded_uniform_logprob(
    values::AbstractVector{<:Real};
    low::Real,
    high::Real,
)
    lp = -log(high - low)
    return [(low <= value <= high) ? lp : -Inf for value in values]
end

function bns_intrinsic_log_prob_samples(
    samples::FullBNSSamples,
    redshift_log_prob::AbstractVector{<:Real},
)
    return ordered_uniform_source_masses_logprob(
        samples.mass_1_source,
        samples.mass_2_source,
    ) .+
           redshift_log_prob .+
           aligned_spin_chi_simple_logprob(samples.chi_1) .+
           aligned_spin_chi_simple_logprob(samples.chi_2) .+
           bounded_uniform_logprob(samples.lambda_1; low=0.0, high=BNS_LAMBDA_HIGH) .+
           bounded_uniform_logprob(samples.lambda_2; low=0.0, high=BNS_LAMBDA_HIGH)
end
