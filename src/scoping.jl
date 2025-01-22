########################################################################
# Function interfacing with ScopedValue
########################################################################
# using Base.ScopedValues

nc::NodeCache =NodeCache()

for f in (:exprhead, :operation, :arity, :arguments, :istree)
    @eval TermInterface.$(f)(id::NodeID) = TermInterface.$(f)(nc.nodes[id.id])
end

Base.show(io::IO, id::NodeID) = print(io, "NodeID($(id.id)): ", expr(nc, id))
TermInterface.similarterm(T::NodeID, args...; kwargs...) = TermInterface.similarterm(nc, T, args...; kwargs...)


function Base.isequal(ni::NodeID, val::T)  where {T<:Integer}
    on = nc[ni]
    (on.head == :integer) && (T(on.v) == val)
end

typeof_value(x::NodeID) = typeof_value(nc[x])

get_value(x::NodeID) = get_value(nc[x])

function instantiate(left::NodeID, pat::PatTerm, mem)
  args = []
  for parg in arguments(pat)
    enqueue = parg isa PatSegment ? append! : push!
    enqueue(args, instantiate(left, parg, mem))
  end
  reference = istree(left) ? left : Expr(:call, :_)
  similarterm(reference, operation(pat), args; exprhead = exprhead(pat))
end

instantiate(left::NodeID, pat::Any, mem) = get!(nc, pat)

instantiate(left::NodeID, pat::AbstractPat, mem) = error("Unsupported pattern ", pat)

function instantiate(left::NodeID, pat::PatVar, mem)
  get!(nc, mem[pat.idx])
end

function instantiate(left::NodeID, pat::PatSegment, mem)
  get!(nc, mem[pat.idx])
end

intern!(ex) = get!(nc, ex)
intern!(ex::Nothing) = nothing

expr(ex::NodeID) = expr(nc, ex)

Base.:(>)(x::NodeID, y::Number) = nc[x].head ∈ (:float, :integer) ? nc[x].v > y : error("comparing non number to number") 
Base.:(<)(x::NodeID, y::Number) = nc[x].head ∈ (:float, :integer) ? nc[x].v < y : error("comparing non number to number") 
Base.:(!=)(x::NodeID, y::Number) = nc[x].head ∈ (:float, :integer) ? nc[x].v != y : true
Base.:(==)(x::NodeID, y::Number) = nc[x].head ∈ (:float, :integer) ? nc[x].v == y : false