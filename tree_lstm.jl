using BenchmarkTools
using Metatheory
using Statistics
using Flux
using Mill


struct ExprEncoding{B, BI, PM}
    buffer::B
    bag_ids::BI
    parent_map::PM
end

# Initialize model and example tree
all_symbols = [:+, :-, :/, :*, :<=, :>=, :min, :max, :<, :>, :select, :&&]
symbols_to_index = Dict(i=>ind for (ind, i) in enumerate(all_symbols))

function ex2mill(ex::Expr, symbols_to_index)
    if ex.head == :call
        fun_name, args =  ex.args[1], ex.args[2:end]
    elseif ex.head == Symbol("&&")
        fun_name = ex.head
        args = ex.args
    else
        error("unknown head $(ex.head)")
    end
    n = length(symbols_to_index) + 2
    ds = ProductNode((
        head = ArrayNode(Flux.onehotbatch([symbols_to_index[fun_name]], 1:n)),
        args = args2mill(args, symbols_to_index)
    ))
    return(ds)
end

function ex2mill(ex::Symbol, symbols_to_index)
    n = length(symbols_to_index) + 2
    ds = ProductNode((
        head = ArrayNode(Flux.onehotbatch([n], 1:n)),
        args = BagNode(missing, [0:-1])
        ))
end

function ex2mill(ex::Number, symbols_to_index)
    n = length(symbols_to_index) + 2
    ds = ProductNode((
        head = ArrayNode(Flux.onehotbatch([n-1], 1:n)),
        args = BagNode(missing, [0:-1])
        ))
end

function args2mill(args::Vector, symbols_to_index)
    isempty(args) && return(BagNode(missing, [0:-1]))
    BagNode(
        reduce(catobs, map(a -> ex2mill(a, symbols_to_index), args)),
        [1:length(args)]
        )
end


struct ExprModel{HM,A,JM,H}
    head_model::HM
    aggregation::A
    joint_model::JM    
    heuristic::H
end


function (m::ExprModel)(ds::ProductNode)
    m.heuristic(embed(m, ds))
end

function embed(m::ExprModel, ds::ProductNode)
    tmp = vcat(m.head_model(ds.data.head.data), embed(m, ds.data.args))
    m.joint_model(tmp)
end

function embed(m::ExprModel, ds::BagNode)
    m.aggregation(embed(m, ds.data), ds.bags)
end

embed(m::ExprModel, ds::Missing) = missing

function loss(heuristic, big_vector, hp=nothing, hn=nothing)
    o = heuristic(big_vector) 
    #diff = o * hn - o * hp
    diff = o * hp - o * hn
    return mean(log.(1 .+ exp.(diff)))
end
