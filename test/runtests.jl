using TupleVectorSpaces, LinearAlgebra, Test

v = TupleVec(3,[4,5],6.0)
w = TupleVec(1.2f0,[1,2],1)
nv = TupleVec(a=4, b=[6,7], c=8)
nw = TupleVec(a=1.2f0, b=[1,2], c=1)
cv = TupleVec(3, [4, 5+6im], 7.0im)

@testset "TupleVec type" begin
    @test Tuple(v) == (3,[4,5],6.0) == Tuple(TupleVec(Tuple(v)))
    @test Tuple(nv) == (4,[6,7],8)
    @test NamedTuple(nv) == (a=4, b=[6,7], c=8) == NamedTuple(TupleVec(; NamedTuple(nv)...))
    @test repr(v) == "TupleVec(3, [4, 5], 6.0)"
    @test repr(nv) == "TupleVec(a = 4, b = [6, 7], c = 8)"
    (; a, c, b) = nv
    @test a == nv.a == 4 && b == nv.b == [6, 7] && c == nv.c == 8
    for (==′) in (==, isequal)
        @test v ==′ TupleVec(3,[4.0,5],6)
        @test nv ==′ TupleVec(a=4, b=[6.0, 7.0], c=8+0im)
    end
    @test !isequal(TupleVec(+0.0), TupleVec(-0.0)) && TupleVec(+0.0) == TupleVec(-0.0)
end

@testset "AdjointTupleVec type" begin
    @test Tuple(cv') == (3, [4 5-6im], -7.0im)
    @test NamedTuple(nv') == (a = 4, b = [6 7], c = 8)
    @test repr(cv') == "adjoint($cv)"
    @test nv'.b == [6 7]
    for (==′) in (==, isequal)
        @test v' ==′ TupleVec(3,[4.0,5],6)'
        @test nv' ==′ TupleVec(a=4, b=[6.0, 7.0], c=8+0im)'
    end
    @test !isequal(TupleVec(+0.0)', TupleVec(-0.0)') && TupleVec(+0.0)' == TupleVec(-0.0)'
end

@testset "TupleVec linear algebra" begin
    @test zero(v) == TupleVec(0, [0, 0], 0.0) == imag(v)
    @test zero(nv) == TupleVec(a = 0, b = [0, 0], c = 0) == imag(nv)
    @test_throws MethodError zero(typeof(v))
    @test zero(typeof(TupleVec(3, 4.0f0))) === TupleVec(0, 0.0f0)
    @test zero(typeof(TupleVec(a=3, b=4.0f0))) === TupleVec(a = 0, b = 0.0f0)
    @test 2v == v*2 == TupleVec(6, [8, 10], 12.0)
    @test v/2 == 2\v == TupleVec(3/2,[2,5/2],3.0)
    @test v + w == TupleVec(4.2f0, [5, 7], 7.0)
    @test v - w == TupleVec(1.8f0, [3, 3], 5.0)
    @test +v == v == real(v) == conj(v) == float(v) == complex(v)
    @test +nv == nv == real(nv) == conj(nv) == float(nv) == complex(nv)
    @test float(nv).a === 4.0
    @test complex(nv).a === 4+0im
    @test -v == TupleVec(-3, [-4, -5], -6.0)
    @test -nv == TupleVec(a = -4, b = [-6, -7], c = -8)
    @test 2nv == TupleVec(a = 8, b = [12, 14], c = 16)
    @test nv + nw == TupleVec(a = 5.2f0, b = [7, 9], c = 9)
    @test reim(cv) == (TupleVec(3, [4, 5], 0.0), TupleVec(0, [0, 6], 7.0))
    @test conj(cv) == TupleVec(3, [4, 5 - 6im], -7.0im)
    @test 2v == v*2 == TupleVec(6, [8, 10], 12.0)
    @test isreal(v) && isreal(nv) && !isreal(cv)
    @test !iszero(v) && !iszero(nv) && iszero(zero(v)) && iszero(zero(nv))
end

@testset "norms and inner products" begin
    @test norm(v) == norm(v,2) == hypot(3, hypot(4,5), 6)
    @test norm(nv) == norm(nv,2) == hypot(4, hypot(6,7), 8)
    @test norm(-v,1) == 3 + hypot(4,5) + 6
    @test norm(-v,Inf) == hypot(4,5)
    @test norm(-v,-Inf) == 3
    @test norm(v,0) == 3
    @test norm(TupleVec(1,[2,3],[0,0],0),0) == 2
    @test norm(v,1.5) ≈ (3^1.5 + hypot(4,5)^1.5 + 6^1.5)^(2//3)
    @test norm(-1e300 * v,1.5) ≈ 1e300 * (3^1.5 + hypot(4,5)^1.5 + 6^1.5)^(2//3)
    @test norm(1e-300 * v,1.5) ≈ 1e-300 * (3^1.5 + hypot(4,5)^1.5 + 6^1.5)^(2//3)
    @test dot(v,v) ≈ norm(v)^2
    @test dot(cv,cv) ≈ norm(cv)^2
    @test dot(nv,nv) ≈ norm(nv)^2
    @test dot(v,w) ≈ 3*1.2f0 + 4+10 + 6
    @test dot(nv,nw) ≈ 4*1.2f0 + 6+14 + 8
    @test v'cv == dot(v,cv) == conj(dot(cv,v)) == (cv'v)' == 9 + 16+25+30im + 42im
end

@testset "AdjointTupleVec linear algebra" begin
    @test zero(v') == TupleVec(0, [0, 0], 0.0)' == imag(v')
    @test zero(nv') == TupleVec(a = 0, b = [0, 0], c = 0)' == imag(nv')
    @test_throws MethodError zero(typeof(v'))
    @test zero(typeof(TupleVec(3, 4.0f0)')) === TupleVec(0, 0.0f0)'
    @test zero(typeof(TupleVec(a=3, b=4.0f0)')) === TupleVec(a = 0, b = 0.0f0)'
    z = 2+3im
    @test z*v' == v'*z == (z'*v)'
    @test v'/z == z\v' == (v/z')'
    @test v' + w' == (v + w)'
    @test v' - w' == (v - w)'
    @test +v' == v' == real(v') == conj(v') == float(v') == complex(v')
    @test +nv' == nv' == real(nv') == conj(nv') == float(nv') == complex(nv')
    @test (nv * im)'.a === -4im
    @test float(nv').a === 4.0
    @test complex(nv').a === 4+0im
    @test -(v') == (-v)'
    @test -(nv') == (-nv)'
    @test 2nv' == (2nv)'
    @test nv' + nw' == (nv + nw)'
    @test reim(cv') == (TupleVec(3, [4, 5], 0.0)', TupleVec(0, [0, 6], 7.0)')
    @test conj(cv') == conj(cv)'
    @test 2v' == v'*2 == (2v)'
    @test isreal(v') && isreal(nv') && !isreal(cv')
    @test !iszero(v') && !iszero(nv') && iszero(zero(v')) && iszero(zero(nv'))
end

@testset "iteration" begin
    @test ndims(v) == ndims(typeof(v)) == 1
    @test length(v) == 4 == length(v')
    @test length(TupleVec()) == length(TupleVec(())) == length(TupleVec((;))) == 0
    @test eltype(v) == Float64 == eltype(v')
    @test eltype(w) == Float32 == eltype(w')
    @test eltype(cv) == ComplexF64 == eltype(cv')
    @test first(v) === 3
    @test first(nv) === nv.a
    @test last(v) === 6.0
    @test vec(w) == collect(w) == [1.2f0, 1, 2, 1]
    @test map(x -> x^2, v) == TupleVec(3^2, [4^2, 5^2], 6.0^2)
    @test map(-, v, w) == v - w
    @test map(+, v, w, 2.5v) == v + w + 2.5v
end
