using AdvancedHMC
using Bijectors
using FiniteDiff
using ForwardDiff
using LogDensityProblems
using LogDensityProblemsAD

const DEFAULT_PARAMETER_ORDER = (:H0, :Omega_m, :chi0, :chin, :gamma, :kappa, :z_peak)

struct ASGWBLogDensity{
    C<:ImportanceSamplingProblem,
    P<:AbstractDict{<:AbstractString,<:Distribution},
    D,
    B,
    N<:Tuple,
}
    problem::C
    priors::P
    prior_distribution::D
    transform::B
    parameter_order::N
end

function build_prior_distribution(
    priors::AbstractDict{<:AbstractString,<:Distribution},
    parameter_order::Tuple=DEFAULT_PARAMETER_ORDER,
)
    return product_distribution((;
        (name => priors[String(name)] for name in parameter_order)...
    ))
end

function ASGWBLogDensity(
    problem::ImportanceSamplingProblem,
    priors::AbstractDict{<:AbstractString,<:Distribution};
    parameter_order::Tuple=DEFAULT_PARAMETER_ORDER,
)
    prior_distribution = build_prior_distribution(priors, parameter_order)
    transform = bijector(prior_distribution)
    return ASGWBLogDensity(
        problem,
        priors,
        prior_distribution,
        transform,
        parameter_order,
    )
end

function constrained_parameters(ld::ASGWBLogDensity, z::AbstractVector{<:Real})
    theta, logabsdet = with_logabsdet_jacobian(inverse(ld.transform), z)
    return theta, logabsdet
end

function unconstrained_initial_point(ld::ASGWBLogDensity, theta::NamedTuple)
    return collect(Bijectors.link(ld.prior_distribution, theta))
end

LogDensityProblems.dimension(ld::ASGWBLogDensity) = length(ld.parameter_order)
LogDensityProblems.capabilities(::Type{<:ASGWBLogDensity}) =
    LogDensityProblems.LogDensityOrder{0}()

function LogDensityProblems.logdensity(
    ld::ASGWBLogDensity,
    z::AbstractVector{<:Real},
)
    theta, logabsdet = constrained_parameters(ld, z)
    return logposterior(theta, ld.problem, ld.priors) + logabsdet
end

function ad_logdensity(ld::ASGWBLogDensity)
    return LogDensityProblemsAD.ADgradient(:ForwardDiff, ld)
end

function finite_difference_logdensity_and_gradient(
    ld::ASGWBLogDensity,
    z::AbstractVector{<:Real},
)
    zf = collect(Float64, z)
    gradient = similar(zf)
    FiniteDiff.finite_difference_gradient!(
        gradient,
        x -> LogDensityProblems.logdensity(ld, x),
        zf,
    )
    return LogDensityProblems.logdensity(ld, zf), gradient
end

function sample_with_advancedhmc(
    problem::ImportanceSamplingProblem,
    priors::AbstractDict{<:AbstractString,<:Distribution},
    theta0::NamedTuple;
    n_adapts::Int=25,
    n_samples::Int=25,
    target_acceptance::Float64=0.8,
    parameter_order::Tuple=DEFAULT_PARAMETER_ORDER,
)
    ld = ASGWBLogDensity(problem, priors; parameter_order=parameter_order)
    z0 = unconstrained_initial_point(ld, theta0)
    ad_problem = ad_logdensity(ld)

    metric = DiagEuclideanMetric(length(z0))
    hamiltonian = Hamiltonian(metric, ad_problem)
    step_size = find_good_stepsize(hamiltonian, z0)
    integrator = Leapfrog(step_size)
    kernel = HMCKernel(
        Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()),
    )
    adaptor = StanHMCAdaptor(
        MassMatrixAdaptor(metric),
        StepSizeAdaptor(target_acceptance, integrator),
    )

    samples_unconstrained, stats = sample(
        hamiltonian,
        kernel,
        z0,
        n_samples,
        adaptor,
        n_adapts;
        progress=false,
    )

    samples_constrained = map(samples_unconstrained) do z
        theta, _ = constrained_parameters(ld, z)
        theta
    end

    return samples_constrained, stats, ld
end
