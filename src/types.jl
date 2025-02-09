##############################################################

"""
    TupleVec{T<:Union{Tuple,NamedTuple}}
    TupleVec(tuple)
    TupleVec(args...)
    TupleVec(; kwargs...)

A `TupleVec{T}` is a wrapper around a `Tuple` or `NamedTuple` object (of type `T`)
that can be treated as an element of an (abstract) vector space defined by the
[direct sum](https://en.wikipedia.org/wiki/Direct_sum) of the tuple components.

That is, a `TupleVec(a, b, …)` acts like a vector where `TupleVec(a, b, …) ± TupleVec(c, d, …)`
gives `TupleVec(a±c, b±c, …)` assuming the individual components `(a, b, …)` support `+` and `-`,
and similarly `TupleVec(a, b, …) * number` gives `TupleVec(a * number, b * number, …)`.

Furthermore, if the individual elements support an inner product (`dot` methods) and/or
a norm (`norm` methods), then the `TupleVec` does as well.  The inner product
`TupleVec(a, b, …) ⋅ TupleVec(c, d, …)` gives `a⋅c + b⋅c + …`, and `norm(TupleVec(a, b, …))`
correspondingly gives `sqrt(norm(a)^2 + norm(b)^2 + …)`.   The adjoint `v'`, or `adjoint(v)`,
is an element of the [dual space](https://en.wikipedia.org/wiki/Dual_space) (the generalization
of a "row vector"): it is a linear operator such that `v' * w == dot(v, w)`.

By implementing these methods, a `TupleVec` can potentially be used to pass combinations of
vectors to any function that only requires such abstract vector-space properties, e.g.
for numerical integration, differentiation, or differential equations.

Iterating over a `TupleVec` corresponds to iterating over each of the components in sequence,
similar to `Iterators.flatten((a, b, …))`.   `map(f, TupleVec(a, b, …))` applies `f` recursively
to the elements, returning a new tuple vector `TupleVec(map(f, a), map(f, b), …)`.

To get back the original `tuple` from `v = TupleVec(tuple)`, you can call `Tuple(v)`.  If
`tuple` is a `NamedTuple`, call `NamedTuple(v)` to get the named tuple back; alternatively,
for named tuples, you can access fields `tuple.field` directly as `v.field`.

# Examples:
```jldoctest
julia> v = TupleVec(1, [2,3,4], [5 6; 7 8]) # tuple of a scalar, a `Vector`, and a `Matrix`
TupleVec(1, [2, 3, 4], [5 6; 7 8])

julia> 2v
TupleVec(2, [4, 6, 8], [10 12; 14 16])

julia> v - v
TupleVec(0, [0, 0, 0], [0 0; 0 0])

julia> collect(v) # the result of iterating over v
8-element Vector{Int64}:
 1
 2
 3
 4
 5
 7
 6
 8

julia> using LinearAlgebra

julia> norm(v)^2, v ⋅ v, v' * v # three ways to get norm²
(204.00000000000003, 204, 204)

julia> v = TupleVec(scalar = 1, vector = [2,3,4], matrix = [5 6; 7 8]) # named tuple
TupleVec(scalar = 1, vector = [2, 3, 4], matrix = [5 6; 7 8])

julia> v.matrix
2×2 Matrix{Int64}:
 5  6
 7  8
```
"""
struct TupleVec{T<:Union{Tuple,NamedTuple}}
    t::T
end
TupleVec(t...) = TupleVec(t)
TupleVec(; kws...) = TupleVec(values(kws))

Base.Tuple(v::TupleVec) = Tuple(_t(v))
Base.NamedTuple(v::TupleVec{<:NamedTuple}) = _t(v)

Base.show(io::IO, v::TupleVec) = print(io, TupleVec, _t(v))

# since we overload getproperty
_t(v::TupleVec) = getfield(v, :t)

# for named tuples and destructuring
@inline Base.getproperty(v::TupleVec, name::Symbol) = getproperty(_t(v), name)
@inline Base.getproperty(v::TupleVec, name::Symbol, order::Symbol) = getproperty(_t(v), name, order)

for comp in (:(==), :isequal)
    @eval Base.$comp(v::TupleVec, w::TupleVec) = $comp(_t(v), _t(w))
end

##############################################################
# adjoint = the dual of a tuple vec (the linear operator that performs a dot product)

struct AdjointTupleVec{T<:TupleVec}
    v::T
end
Base.parent(a::AdjointTupleVec) = getfield(a, :v)
Base.adjoint(v::TupleVec) = AdjointTupleVec(v)
Base.adjoint(a::AdjointTupleVec) = parent(a)
Base.show(io::IO, a::AdjointTupleVec) = print(io, "adjoint(", parent(a), ')')

Base.Tuple(a::AdjointTupleVec) = map(adjoint, Tuple(parent(a)))
Base.NamedTuple(a::AdjointTupleVec) = map(adjoint, NamedTuple(parent(a)))

# for named tuples and destructuring
@inline Base.getproperty(a::AdjointTupleVec, name::Symbol) = getproperty(_t(parent(a)), name)'
@inline Base.getproperty(a::AdjointTupleVec, name::Symbol, order::Symbol) = getproperty(_t(parent(a)), name, order)'

for comp in (:(==), :isequal)
    @eval Base.$comp(v::AdjointTupleVec, w::AdjointTupleVec) = $comp(parent(v), parent(w))
end

##############################################################
