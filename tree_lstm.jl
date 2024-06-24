#include("src/MyModule.jl")
using BenchmarkTools
#using .MyModule: load_data, preprosses_data_to_expressions
using Metatheory
using Statistics
using Flux
using Mill


struct ExprEncoding{B, BI, PM}
    buffer::B
    bag_ids::BI
    parent_map::PM
end

# Define the TreeLSTM model
mutable struct TreeLSTM
    W_i::Dense
    W_f::Dense
    W_o::Dense
    W_u::Dense
    U_i::Dense
    U_f::Dense
    U_o::Dense
    U_u::Dense
end

TreeLSTM(in_dim::Int, out_dim::Int) = TreeLSTM(
    Dense( in_dim, out_dim),
    Dense( in_dim, out_dim),
    Dense( in_dim, out_dim),
    Dense( in_dim, out_dim),
    Dense(out_dim, out_dim),
    Dense(out_dim, out_dim),
    Dense(out_dim, out_dim),
    Dense(out_dim, out_dim)
)


function (lstm::TreeLSTM)(x::Vector{Float32}, c::Vector{Float32}, h::Vector{Float32})
    i = sigmoid.(lstm.W_i(x) .+ lstm.U_i(h))
    f = sigmoid.(lstm.W_f(x) .+ lstm.U_f(h))
    o = sigmoid.(lstm.W_o(x) .+ lstm.U_o(h))
    u = tanh.(lstm.W_u(x) .+ lstm.U_u(h))
    c_next = f .* c .+ i .* u
    h_next = o .* tanh.(c_next)
    return c_next, h_next
end

function (lstm::TreeLSTM)(x::Matrix{Float32}, c::Matrix{Float32}, h::Matrix{Float32})
    i = sigmoid.(lstm.W_i(x) .+ lstm.U_i(h))
    f = sigmoid.(lstm.W_f(x) .+ lstm.U_f(h))
    o = sigmoid.(lstm.W_o(x) .+ lstm.U_o(h))
    u = tanh.(lstm.W_u(x) .+ lstm.U_u(h))
    c_next = f .* c .+ i .* u
    h_next = o .* tanh.(c_next)
    return c_next, h_next
end

@inline function tree_lstm(lstm::TreeLSTM, tree::Dict, c::Vector{Float32}, h::Vector{Float32})
    if isnothing(tree.children)
        data = tree.encoding
        return lstm(data, c, h)
    else
        c_combined = c
        h_combined = h
        for child in tree.children
            c_child, h_child = tree_lstm(lstm, child, c, h)
            c_combined += c_child
            h_combined += h_child
        end
        data = tree.encoding
        return lstm(data, c_combined, h_combined)
    end
end

struct LikelihoodModel
    tree_lstm::TreeLSTM
    fc::Dense
    output_layer::Dense
end

LikelihoodModel(input_size::Int, hidden_size::Int, mlp_hidden_size::Int) = LikelihoodModel(
    TreeLSTM(input_size, hidden_size),
    Dense(hidden_size, mlp_hidden_size, relu),
    Dense(mlp_hidden_size, 1, σ)
)

function (m::LikelihoodModel)(tree::Dict, c_init::Vector{Float32}=zeros(Float32, size(m.fc.weight)[2]), h_init::Vector{Float32}=zeros(Float32, size(m.fc.weight)[2]))::Float32
    h, _ = tree_lstm(m.tree_lstm, tree, c_init, h_init)
    h = m.fc(h)
    output = m.output_layer(h)
    return output[1]
end

function (m::LikelihoodModel)(buffer::Dict, parent_map::Dict, bag_ids::Dict)
    h, _ = tree_lstm1(m.tree_lstm, buffer, parent_map, bag_ids)
    h = m.fc(h)
    output = m.output_layer(h)
    return output[1]
end


function (m::LikelihoodModel)(expr_encoder::ExprEncoding)
    buffer, bag_ids, parent_map = expr_encoder.buffer, expr_encoder.bag_ids, expr_encoder.parent_map
    h, _ = tree_lstm1(m.tree_lstm, buffer, parent_map, bag_ids)
    h = m.fc(h)
    output = m.output_layer(h)
    return output[1]
end
# Initialize model and example tree
all_symbols = [:+, :-, :/, :*, :<=, :>=, :min, :max, :<, :>, :select, :&&]
in_dim = length(all_symbols) + 2
symbols_to_index = Dict(i=>ind for (ind, i) in enumerate(all_symbols))
out_dim = in_dim * 2
#ex = :(a - b + c * 100 / 200 - d - (a + b - c))
#ex = :((a - b) + (c / d))
ex = :(a - b + c) 

#symbol_encoding(sym::Symbol) = sym in all_symbols ? ones(length(all_symbols) + 2) * symbols_to_index[sym] : ones(length(all_symbols) + 2) * -2
#symbol_encoding(sym) = ones(length(all_symbols) + 2) * -1



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
    m.heuristic(embed(m, ds))[1]
end

function embed(m::ExprModel, ds::ProductNode)
    tmp = vcat(m.head_model(ds.data.head.data), embed(m, ds.data.args))
    m.joint_model(tmp)
end

function embed(m::ExprModel, ds::BagNode)
    m.aggregation(embed(m, ds.data), ds.bags)
end

embed(m::ExprModel, ds::Missing) = missing


#data = load_data("data/test.json")[1:100]
#data = preprosses_data_to_expressions(data)

hidden_size = 8
m = ExprModel(
    Flux.Chain(Dense(length(symbols_to_index) + 2, hidden_size,relu), Dense(hidden_size,hidden_size)),
    Mill.SegmentedSumMax(hidden_size),
    Flux.Chain(Dense(3*hidden_size, hidden_size,relu), Dense(hidden_size, hidden_size)),
    Flux.Chain(Dense(hidden_size, hidden_size,relu), Dense(hidden_size, 1))
    )

#dss = [ex2mill(ex, symbols_to_index) for ex in data[1:10]]
#@benchmark [ex2mill(ex, symbols_to_index) for ex in data[1:10]]
#ds = reduce(catobs, dss)
#reduce(hcat, map(m, dss)) ≈ m(ds)
#m(ex2mill(:(a-a), symbols_to_index))

function expression_encoder!(ex::Union{Expr, Symbol, Int64}, all_symbols::Vector{Symbol}, symbols_to_index::Dict{Symbol, Int64}, depth::Int, buffer::Dict{Int, Vector}, bag_ids::Dict{Int, Vector}, parent_id::Int, parent_map::Dict{Int,Int})
    if ex isa Expr
        encoded_symbol_root = zeros(Float32, length(all_symbols) + 2)
        if ex.head != :call
            encoded_symbol_root[symbols_to_index[ex.head]] = 1
        else
            encoded_symbol_root[symbols_to_index[ex.args[1]]] = 1
        end
        node_id = length(parent_map) + 1
        parent_map[node_id] = parent_id

        if depth == 1
            buffer[depth] = [encoded_symbol_root]
            bag_ids[depth] = [node_id]
        else
            if haskey(buffer, depth)
                push!(buffer[depth], encoded_symbol_root)
                push!(bag_ids[depth], node_id)
            else
                buffer[depth] = [encoded_symbol_root]
                bag_ids[depth] = [node_id]
            end
        end

        depth += 1
        encoded_symbol_first_child = expression_encoder!(ex.args[2], all_symbols, symbols_to_index, depth, buffer, bag_ids, node_id, parent_map)
        #=
        if haskey(buffer, depth)
            push!(buffer[depth], encoded_symbol_first_child)
            push!(bag_ids[depth], node_id+1)
        else
            buffer[depth] = [encoded_symbol_first_child]
            bag_ids[depth] = [node_id+1]
        end
        =#
        if length(ex.args) == 3
            encoded_symbol_second_child = expression_encoder!(ex.args[3], all_symbols, symbols_to_index, depth, buffer, bag_ids, node_id, parent_map)
            #push!(buffer[depth], encoded_symbol_second_child)
            #push!(bag_ids[depth], node_id+2)
        end
        return encoded_symbol_root
    elseif ex isa Symbol && ex in all_symbols
        encoded_symbol = zeros(Float32, length(all_symbols) + 2)
        encoded_symbol[symbols_to_index[ex]] = 1 
        node_id = length(parent_map) + 1
        parent_map[node_id] = parent_id
        if !haskey(bag_ids, depth)
            buffer[depth] = [encoded_symbol]
            bag_ids[depth] = [node_id]
        else
            push!(buffer[depth], encoded_symbol)
            push!(bag_ids[depth], node_id)
        end
        return encoded_symbol
    elseif ex isa Symbol 
        encoded_symbol = zeros(Float32, length(all_symbols) + 2)
        encoded_symbol[end] = 1
        node_id = length(parent_map) + 1
        parent_map[node_id] = parent_id
        if !haskey(bag_ids, depth)
            buffer[depth] = [encoded_symbol]
            bag_ids[depth] = [node_id]
        else
            push!(buffer[depth], encoded_symbol)
            push!(bag_ids[depth], node_id)
        end
        return encoded_symbol
    else
        encoded_symbol = zeros(Float32, length(all_symbols) + 2)
        encoded_symbol[end - 1] = 1
        node_id = length(parent_map) + 1
        parent_map[node_id] = parent_id
        if !haskey(bag_ids, depth)
            buffer[depth] = [encoded_symbol]
            bag_ids[depth] = [node_id]
        else
            push!(buffer[depth], encoded_symbol)
            push!(bag_ids[depth], node_id)
        end
        return encoded_symbol
    end
end

@inline function tree_lstm1(lstm::TreeLSTM, buffer::Dict, parent_map::Dict, bag_ids::Dict)
    h_tmp, c_tmp = lstm(buffer[1][1], zeros(Float32, out_dim), zeros(Float32, out_dim))
    return h_tmp, c_tmp
    #=
    all_nodes = length(parent_map)
    all_depth = length(buffer)
    out_dim = size(lstm.W_i.weight, 1)
    if all_nodes == 1
        h_tmp, c_tmp = lstm(buffer[1][1], zeros(Float32, out_dim), zeros(Float32, out_dim))
        return h_tmp, c_tmp
    end
    #c_init = zeros(Float32, out_dim, length(buffer[all_depth]))
    #h_init = zeros(Float32, out_dim, length(buffer[all_depth]))
    #println(buffer)
    c_tmp = zeros(Float32, out_dim, length(buffer[all_depth]))
    h_tmp = zeros(Float32, out_dim, length(buffer[all_depth]))
    while true
        x = hcat(buffer[all_depth]...)
        all_depth -= 1
        c_tmp, h_tmp = lstm(x, c_tmp, h_tmp) 
        if all_depth == 0
            return h_tmp, c_tmp
        end
        c_new = zeros(Float32, out_dim, length(buffer[all_depth]))
        h_new = zeros(Float32, out_dim, length(buffer[all_depth]))
        for (node_index, node_id) in enumerate(bag_ids[all_depth + 1])
            parent_index = findall(x-> x == parent_map[node_id], bag_ids[all_depth])[1]
            if parent_index !== nothing
                c_new[:, parent_index] += c_tmp[:, node_index]
                h_new[:, parent_index] += h_tmp[:, node_index]
            end
        end
        c_tmp = c_new
        h_tmp = h_new
    end
    =#
end


#mod2 = LikelihoodModel(in_dim, out_dim, out_dim * 2)
#for ex in data
function expression_encoder(ex::Union{Expr, Int}, all_symbols::Vector{Symbol}, symbols_to_index::Dict{Symbol, Int})::ExprEncoding
    buffer = Dict{Int, Vector}()
    bag_ids = Dict{Int, Vector}()
    parent_map = Dict{Int, Int}()
    #expression_encoder!(ex, all_symbols, symbols_to_index, 1, buffer, bag_ids, 1, parent_map)
    x = expression_encoder!(ex, all_symbols, symbols_to_index, 1, buffer, bag_ids, 1, parent_map)
    if isempty(buffer)
        buffer[1] = x 
        bag_ids[1] = [1]
    end
    return ExprEncoding(buffer, bag_ids, parent_map)  
end

function loss(model::ExprModel, depth_dict::Dict{Int, Vector{Any}})
  expanded_node_out = model(depth_dict[1][1])
  not_expanded_nodes_out = model.(depth_dict[1][2:end])
  return mean(log.(1 .+ exp.(expanded_node_out .- not_expanded_nodes_out)))
  #return mean([mean(log.(1 .+ exp.(model(depth_dict[i][1]) .- model.(depth_dict[i][2:end])))) for i in length(depth_dict):-1:1])
end

#function loss(model::LikelihoodModel, depth_dict::Dict{Int, Vector})
    #=
    result = []
    for i in length(depth_dict):-1:1 
        expanded_node_prob = model(depth_dict[i][1])
        not_expanded_node_prob = model.(depth_dict[i][2:end])
        lgbf_loss = mean(log.(1 .+ exp.(expanded_node_prob .- not_expanded_node_prob)))
        push!(result, lgbf_loss)
    end
    return mean(result) 
    =#
    #return mean([mean(log.(1 .+ exp.(model(depth_dict[i][1]) .- model.(depth_dict[i][2:end])))) for i in length(depth_dict):-1:1])
    #return model(depth_dict[1][1])
    #expr_encoder = depth_dict[1][1]
    #buffer, bag_ids, parent_map = expr_encoder.buffer, expr_encoder.bag_ids, expr_encoder.parent_map
    #output = model(buffer, parent_map, bag_ids)
    #return mean(log.(1 .+ exp.(output)))
#end
