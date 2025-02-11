"""
The `TupleVectorSpaces` module exports a new type [`TupleVec`](@ref) that
wraps around a tuple (or named tuple), and allows it to act like
element of an abstract vector space defined by the
[direct sum](https://en.wikipedia.org/wiki/Direct_sum) of the tuple components.
"""
module TupleVectorSpaces
export TupleVec

include("types.jl")
include("linalg.jl")
include("iteration.jl")

end
