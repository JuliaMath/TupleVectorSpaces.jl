# TupleVecs

[![Build Status](https://github.com/JuliaMath/TupleVecs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaMath/TupleVecs.jl/actions/workflows/CI.yml?query=branch%3Amain)

This Julia package allows you to take a tuple of objects and treat it as a "vector", in the sense of an [abstract vector space](https://en.wikipedia.org/wiki/Vector_space) (*not* a 1d array), as long as the components are vectors — that is, as long as they support addition, subtraction, and multiplication by scalars.  Technically, this is known as a [direct sum](https://en.wikipedia.org/wiki/Direct_sum) of vector spaces, and is represented in this package by the `TupleVec` type.

The application of `TupleVec` is to allow you to easily take a *heterogeneous* combination of objects (e.g. scalars, column vectors, matrices, and more) and treat them together as a single "vector" for the purposes of algorithms like numerical integration, differentiation, interpolation, and extrapolation (that only rely on abstract vector-space properties), without having to manually "flatten" them into a single array of numbers.

If the elements of your tuple support an [inner product](https://en.wikipedia.org/wiki/Inner_product_space) and/or a [norm](https://en.wikipedia.org/wiki/Normed_vector_space), then your `TupleVec` will also.  (Algorithms for numerical integration or differentiation typically require a norm.)

## The `TupleVec` type

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

## Examples:
```jl
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

## Compatible packages
So far, the `TupleVec` type has been tested to be compatible with:

* [FiniteDifferences.jl](https://github.com/JuliaDiff/FiniteDifferences.jl) (numerical differentiation)
* [QuadGK.jl](https://github.com/JuliaMath/QuadGK.jl) and [HCubature.jl](https://github.com/JuliaMath/HCubature.jl) (numerical integration)
* [Richardson.jl](https://github.com/JuliaMath/Richardson.jl) (extrapolation)
* [Interpolations.jl](http://juliamath.github.io/Interpolations.jl) and [BasicInterpolators.jl](https://github.com/markmbaum/BasicInterpolators.jl) (interpolation)

It should work with any package that can act on opaque "vector" objects (objects supporting $\pm$, multiplication by scalars, norms or inner products, and related functions) without needing to "look inside" at the components of the objects.

(Compatibility with DifferentialEquations.jl, for ODE integration, is a work in progress.)
