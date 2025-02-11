##############################################################
# iteration, defined recursively over the components

# based on Iterators.Flatten:
Base.@propagate_inbounds function Base.iterate(v::TupleVec, state=())
    if state !== ()
        y = iterate(Base.tail(state)...)
        y !== nothing && return (y[1], (state[1], state[2], y[2]))
    end
    x = (state === () ? iterate(_t(v)) : iterate(_t(v), state[1]))
    x === nothing && return nothing
    y = iterate(x[1])
    while y === nothing
         x = iterate(_t(v), x[2])
         x === nothing && return nothing
         y = iterate(x[1])
    end
    return y[1], (x[2], x[1], y[2])
end

# the "scalar type" of the vector space
Base.eltype(v::TupleVec) = promote_type(map(eltype, _t(v))...)

Base.first(v::TupleVec) = first(first(_t(v)))
Base.last(v::TupleVec) = last(last(_t(v)))

# the dimension of the vector space
Base.length(v::TupleVec) = +(map(length, _t(v))...)
Base.length(v::TupleVec{Tuple{}}) = 0
Base.length(v::TupleVec{@NamedTuple{}}) = 0

Base.vec(v::TupleVec) = collect(v)

Base.map(f, v::TupleVec) = TupleVec(map(x -> map(f, x), _t(v)))
Base.map(f, v::TupleVec, w::TupleVec) = TupleVec(map((x,y) -> map(f, x, y), _t(v), _t(w)))
Base.map(f, v1::TupleVec, v2::TupleVec, vs::TupleVec...) = TupleVec(map((x...) -> map(f, x...), map(v -> _t(v), (v1, v2, vs...))...))

##############################################################
# broadcasting: treat as "atom", not container:
Base.broadcastable(v::TupleVec) = Ref(v)

##############################################################
# iteration on adjoint vectors

for f in (:length, :eltype)
    @eval Base.$f(a::AdjointTupleVec) = $f(parent(a))
end

# todo: support iteration on AdjointTupleVec?

##############################################################
# todo: inverse of vec — reshaping a vector into a TupleVec
# (given an instance, or a type if size information is in the type)

##############################################################
# todo: support indexing

Base.ndims(v::TupleVec) = 1 # needed by DifferentialEquations
Base.ndims(v::Type{<:TupleVec}) = 1 # needed by DifferentialEquations

##############################################################
