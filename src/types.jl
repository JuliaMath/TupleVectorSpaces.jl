##############################################################

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
