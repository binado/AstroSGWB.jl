using Distributions: ProductNamedTupleDistribution

function _batch_length(field)
    values = sample_values(field)
    return size(values, ndims(values))
end

"""
    validate_samples(prior::ProductNamedTupleDistribution, samples::NamedTuple) -> Int

Verify that `samples` contains every field in `prior.dists` and that all fields
share the same batch length (size along the last axis). Returns that length `n`.

Sample layout contract:
- univariate fields: `Vector` of length `n`
- multivariate fields: `Matrix` of shape `(length(d), n)`

Fields may be wrapped in [`SampleField`](@ref); metadata is ignored.
Extra keys in `samples` beyond those required by `prior` are allowed.
"""
function validate_samples(
        prior::ProductNamedTupleDistribution,
        samples::NamedTuple
)::Int
    ks = keys(prior.dists)
    isempty(ks) && return 0
    first_key = first(ks)
    haskey(samples, first_key) ||
        throw(ArgumentError("samples are missing population prior field $(repr(first_key))"))
    n = _batch_length(samples[first_key])
    for key in ks
        haskey(samples, key) ||
            throw(ArgumentError("samples are missing population prior field $(repr(key))"))
        n_key = _batch_length(samples[key])
        n_key == n ||
            throw(ArgumentError("population prior sample fields must have matching lengths"))
    end
    return n
end

"""
    stack_source_masses(mass_1_source, mass_2_source) -> Matrix{Float64}

Pack two same-length mass vectors into a `2 × n` matrix (row 1 = `mass_1_source`,
row 2 = `mass_2_source`), the layout expected under the `mass` field of a
proposal-sample `NamedTuple`.
"""
function stack_source_masses(
        mass_1_source::AbstractVector{<:Real},
        mass_2_source::AbstractVector{<:Real}
)::Matrix{Float64}
    n = length(mass_1_source)
    length(mass_2_source) == n ||
        throw(ArgumentError("mass_1_source and mass_2_source must have matching lengths"))
    return permutedims(
        hcat(collect(Float64, mass_1_source), collect(Float64, mass_2_source)),
    )
end
