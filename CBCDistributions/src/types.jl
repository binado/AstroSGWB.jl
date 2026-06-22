# Previously held RedshiftPriorSpec / RedshiftPriorFamily; removed in favour of
# the caller-owned PopulationModel interface and a module-level DEFAULT_Z_GRID constant.

"""
    SampleField(values, meta=nothing)

Wrapper for a batched sample field and optional per-field metadata. The
`logpdfvec` batching contract evaluates `values`; specialized distributions may
use `meta` for precomputed state tied to those fixed sample locations.
"""
struct SampleField{V, M}
    values::V
    meta::M
end

SampleField(values) = SampleField(values, nothing)

sample_values(field) = field
sample_values(field::SampleField) = field.values

sample_meta(field) = nothing
sample_meta(field::SampleField) = field.meta
