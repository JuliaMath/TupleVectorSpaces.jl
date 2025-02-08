using LinearAlgebra

##############################################################
# basic vector-space operations:

Base.zero(v::TupleVec) = TupleVec(map(zero, _t(v)))
Base.zero(::Type{TupleVec{T}}) where {T<:Tuple} =
    TupleVec(map(zero, fieldtypes(T)))

for op in (:+, :-)
    @eval Base.$op(v::TupleVec, w::TupleVec) = TupleVec(map($op, _t(v), _t(w)))
end
for f in (:+, :-, :real, :imag, :complex, :conj, :float, :big)
    @eval Base.$f(v::TupleVec) = TupleVec(map($f, _t(v)))
end
Base.:*(v::TupleVec, a::Number) = TupleVec(map(x -> x*a, _t(v)))
Base.:*(a::Number, v::TupleVec) = TupleVec(map(x -> a*x, _t(v)))

for comp in (:(==), :isequal)
    @eval Base.$comp(v::TupleVec, w::TupleVec) = $comp(_t(v), _t(w))
end

for f in (:isreal, :iszero)
    @eval Base.$f(v::TupleVec) = all(map($f, _t(v)))
end

Base.reim(v::TupleVec) = (real(v), imag(v))

##############################################################
# norms, defined recursively (consistent with inner product)

LinearAlgebra.norm(v::TupleVec) = hypot(map(norm, _t(v))...)

# p norms are *not* applied recursively, since
# the component vectors spaces are only assumed
# to have norms, not necessarily p norms.
function LinearAlgebra.norm(v::TupleVec, p::Real)
    norms = map(norm, _t(v))
    if p == 2
        return hypot(norms...)
    elseif p == 1
        return float(sum(norms))
    elseif p == Inf
        return float(maximum(norms))
    elseif p == 0
        return typeof(float(sum(norms)))(count(!iszero, norms))
    elseif p == -Inf
        return float(minimum(norms))
    else
        maxnorm = float(maximum(norms)) # rescale to avoid overflow/underflow
        iszero(maxnorm) && return maxnorm
        return maxnorm * sum(n -> (n/maxnorm)^p, norms)^inv(p)
    end
end

##############################################################
# inner products, defined recursively

LinearAlgebra.dot(v::TupleVec, w::TupleVec) = sum(map(dot, _t(v), _t(w)))
Base.:*(a::AdjointTupleVec, v::TupleVec) = dot(parent(a), v)

##############################################################
# vector-space operations on adjoint vectors,
# (delegated to parent vectors)

Base.zero(a::AdjointTupleVec) = parent(a)'
Base.zero(::Type{AdjointTupleVec{T}}) where {T} = zero(T)'

for op in (:+, :-)
    @eval Base.$op(v::AdjointTupleVec, w::AdjointTupleVec) = $op(parent(v), parent(w))'
end
for f in (:+, :-, :real, :imag, :complex, :conj, :float, :big)
    @eval Base.$f(v::AdjointTupleVec) = $f(parent(v))'
end
Base.:*(v::AdjointTupleVec, a::Number) = (a' * parent(v))'
Base.:*(a::Number, v::AdjointTupleVec) = (parent(v) * a')'

for comp in (:(==), :isequal)
    @eval Base.$comp(v::AdjointTupleVec, w::AdjointTupleVec) = $comp(parent(v), parent(w))
end

for f in (:isreal, :iszero)
    @eval Base.$f(v::AdjointTupleVec) = $f(parent(v))
end

Base.reim(v::AdjointTupleVec) = (real(v), imag(v))

LinearAlgebra.norm(v::AdjointTupleVec) = norm(parent(v))
LinearAlgebra.norm(v::AdjointTupleVec, p::Real) = norm(parent(v), p)

LinearAlgebra.dot(v::AdjointTupleVec, w::AdjointTupleVec) = conj(dot(parent(v), parent(w)))

##############################################################
