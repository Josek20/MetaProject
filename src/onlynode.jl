
using TermInterface

"""
    This is an id of a node, which is effectively just a pointer to the structure.
    It is wrapped to NodeID to have its own type for stability.
"""
struct NodeID
    id::UInt32
end
"""
    The main structure of the node.
"""
struct OnlyNode
    head::Symbol
    iscall::Bool
    v::Float32
    left::NodeID
    right::NodeID
    hash_id::UInt64
end

const nullnode = OnlyNode(:null, false, 0f0, NodeID(1), NodeID(1), 0)
const nullid = NodeID(1)

function OnlyNode(head, iscall, v, left, right)
    id = hash((head, iscall, v, left, right))
    OnlyNode(head, iscall, v, left, right, id)
end

Base.hash(o::OnlyNode) = o.hash_id
Base.hash(o::OnlyNode, i::UInt64) = hash(o.hash_id, i)

@inline car(t::NodeID) = operation(t)
@inline cdr(t::NodeID) = arguments(t)

"""
struct NodeCache
    nodemap::Dict{OnlyNode, NodeID}
    nodes::Vector{OnlyNode}
end

    Contains cache of nodes and mappings from nodes to their ID. An inner constructor enforces
    that first element is `nullnode`, which is special and during construction we verify that 
    `nodemap[nullnode] != nullid `.
"""
struct NodeCache
    nodemap::Dict{OnlyNode, NodeID}
    nodes::Vector{OnlyNode}

    function NodeCache(nodemap, nodes)
        length(nodemap) != length(nodes) && error("length of nodemap and nodes has to be the same")
        if isempty(nodes)
            push!(nodes, nullnode)
            nodemap[nullnode] = nullid
            return(new(nodemap, nodes))
        end
        nodemap[nullnode] != nullid && error("The first item of nodes has to be nullnode")
        new(nodemap, nodes)
    end
end

Base.show(io::IO, nc::NodeCache) = print(io, "NodeCache(",length(nc.nodes), ")")


function NodeCache()
    NodeCache(Dict{OnlyNode, NodeID}(), OnlyNode[])
end

function Base.get!(nc::NodeCache, k::OnlyNode)
    get!(nc.nodemap, k) do 
        push!(nc.nodes, k)
        NodeID(length(nc.nodes))
    end
end

Base.getindex(nc::NodeCache, id::NodeID) = nc.nodes[id.id]

LeafNode(e::Integer) = OnlyNode(:integer, false, e, nullid, nullid, hash(e))
LeafNode(e::AbstractFloat) = OnlyNode(:float, false, e, nullid, nullid, hash(e))
LeafNode(e::Symbol) = OnlyNode(e, false, 0f0, nullid, nullid, hash(e))

Base.get!(nc::NodeCache, e::Union{Integer,<:AbstractFloat, Symbol}) = get!(nc, LeafNode(e))

function Base.get!(nc::NodeCache, e::Expr)
    if e.head == :call 
        length(e.args) > 3 && error("arguments too long")
        head = e.args[1]
        left = length(e.args) ≥ 2 ? get!(nc, e.args[2]) : nullid
        right = length(e.args) ≥ 3 ? get!(nc, e.args[3]) : nullid
        return(get!(nc, OnlyNode(head, true, 0f0, left, right)))
    elseif e.head in [:&&, :||]
        length(e.args) > 2 && error("arguments too long")
        head = e.head
        left = length(e.args) ≥ 1 ? get!(nc, e.args[1]) : nullid
        right = length(e.args) ≥ 2 ? get!(nc, e.args[2]) : nullid
        return(get!(nc, OnlyNode(head, true, 0f0, left, right)))    
    else
        length(e.args) > 2 && error("arguments too long")
        isempty(e.args) && return(get!(nc, e.head))
        head = e.head
        left = length(e.args) ≥ 1 ? get!(nc, e.args[1]) : nullid
        right = length(e.args) ≥ 2 ? get!(nc, e.args[2]) : nullid
        return(get!(nc, OnlyNode(head, false, 0f0, left, right)))
    end
end

Base.get!(nc::NodeCache, e::NodeID) = e

#####################################################################################
#       reconstruction
#####################################################################################
function expr(nc::NodeCache, id::NodeID)
    expr(nc, nc.nodes[id.id])
end

function expr(nc::NodeCache, e::OnlyNode)
    e.head == :integer && return(Int32(e.v))
    e.head == :float && return(e.v)
    if !e.iscall && (e.left == nullid) && (e.right ==nullid) && e.v == 0f0
        return(e.head)
    end

    head = e.iscall && !(e.head in [:&&, :||]) ? :call : e.head
    args = e.iscall && !(e.head in [:&&, :||]) ? Any[e.head] : Any[]
    e.left != nullid && push!(args, expr(nc, e.left))
    e.right != nullid && push!(args, expr(nc, e.right))
    return(Expr(head, args...))
end

#####################################################################################
#       Make it compatible with TermInterface
#####################################################################################
# function TermInterface.head(ex::OnlyNode)
#     ex.head == :integer && return(ex.v)
#     ex.head == :float && return(ex.v)
#     return(ex.head)
# end

# function TermInterface.arity(ex::OnlyNode)
#     return((ex.left != nullid) + (ex.right != nullid))
# end

# function TermInterface.children(ex::OnlyNode)
#     args = TermInterface.arguments(ex)
#     !iscall(ex) && return(args)
#     return(vcat([ex.head], args))
# end

# function TermInterface.arguments(ex::OnlyNode)
#     ex.right != nullid && return([ex.left, ex.right])
#     ex.left != nullid && return([ex.left])
#     return(NodeID[])
# end

# TermInterface.operation(ex::OnlyNode) = ex.head
# TermInterface.isexpr(ex::OnlyNode) = ex.iscall && (ex.left != nullnode)
# TermInterface.iscall(ex::OnlyNode) = ex.iscall

# function TermInterface.maketerm(nc::NodeCache, ::Type{NodeID}, head::Function, children, ::Nothing)
#     head = Symbol(head)
#     left = length(children) ≥ 1 ? get!(nc, children[1]) : nullid
#     right = length(children) ≥ 2 ? get!(nc, children[2]) : nullid
#     length(children) > 2 && error("Too many childrens")
#     id = hash((hash(:call), hash(head), hash(left), hash(right)))
#     @show head
#     @show left
#     @show right
#     return(get!(nc, OnlyNode(head, true, 0f0, left, right, id)))
# end


function TermInterface.exprhead(ex::OnlyNode)
    ex.head == :integer && return(ex.v)
    ex.head == :float && return(ex.v)
    ex.iscall && return(:call)
    return(ex.head)
end

function TermInterface.operation(ex::OnlyNode)
    ex.head == :integer && error("should not be called")
    ex.head == :float && error("should not be called")
    return(ex.head)
end

function TermInterface.arity(ex::OnlyNode)
    return((ex.left != nullid) + (ex.right != nullid))
end

function TermInterface.arguments(ex::OnlyNode)
    ex.right != nullid && return([ex.left, ex.right])
    ex.left != nullid && return([ex.left])
    return(NodeID[])
end

TermInterface.istree(ex::OnlyNode) = ex.iscall && (ex.left != nullnode)

function TermInterface.similarterm(nc::NodeCache, ::NodeID, head, children, symtype = nothing; metadata = nothing, exprhead = nothing)
    head = head isa Symbol ? head : nameof(head)
    left = length(children) ≥ 1 ? get!(nc, children[1]) : nullid
    right = length(children) ≥ 2 ? get!(nc, children[2]) : nullid
    length(children) > 2 && error("Too many childrens")
    iscall = exprhead == :call
    return(get!(nc, OnlyNode(head, iscall, 0f0, left, right)))
end
