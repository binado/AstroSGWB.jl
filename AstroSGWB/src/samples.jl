redshift(s::NamedTuple) = s.redshift

"""
    with_redshift_interpolant(samples::NamedTuple, query::GridQuery) -> NamedTuple

Attach the proposal redshift [`GridQuery`](@ref) to the `redshift` field of `samples`,
wrapping it in a [`SampleField`](@ref) so the redshift logpdf reuses the precomputed
per-sample grid locations instead of re-searching the grid every gradient evaluation.
Model authors call this when assembling the proposal caches and inside their
[`merger_rate_and_log_weights`](@ref) joint.
"""
function with_redshift_interpolant(samples::NamedTuple, query::GridQuery)
    return merge(samples, (; redshift = SampleField(samples.redshift, query)))
end
