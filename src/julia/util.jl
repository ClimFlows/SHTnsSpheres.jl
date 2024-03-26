const Out{T} = Union{Void,T}

# isvoid(x) = isa(x, Void)
ifvoid(x, f, args) = x
ifvoid(x::Void, f, args) = f(args...)

# allow input arguments to be modified

# user :
#   pass inout(x) as input to express that it is OK to overwrite x
# this module:
#   x = readable(x) for inputs that will not be modified
#   x = writable(x) for inputs that will be modified

"""
Wraps data to mark it as writeable.
"""
struct InOut{A}
    data::A
end
const In{T} = Union{T, InOut{T}}

"""
    y = erase(x)
Some functions need to make copies of input arguments to avoid modifying their contents
and remain pure. Passing `erase(x)` as input argument is equivalent to passing `x`, except
that it explicitly allows to modify the contents of `x`, thus avoiding
copying and allocating.
"""
erase(x) = InOut(x)
erase(x::InOut) = x

""" Unwrap input argument. Used internally when we can promise that x will not be modified."""
readable(x) = x
readable(x::InOut) = x.data

""" Unwrap input argument. Used internally when we cannot promise that x will not be modified."""
writable(x) = copy_input(x)
writable(x::InOut) = x.data
copy_input(x) = copy(x)
copy_input(x::NamedTuple) = map(copy, x)
