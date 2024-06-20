using Metatheory
using Statistics
using Flux
#: Dense, sigmoid, tanh, relu, \sigmoid

struct MyNodes#{C, E<:Vector{UInt32}}
    children::Union{Vector{MyNodes}, Nothing}
    #children::C
    encoding::Vector{UInt32}
    #encoding::E
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


function (lstm::TreeLSTM)(x::Vector{UInt32}, c::Vector{Float32}, h::Vector{Float32})
    i = sigmoid.(lstm.W_i(x) .+ lstm.U_i(h))
    f = sigmoid.(lstm.W_f(x) .+ lstm.U_f(h))
    o = sigmoid.(lstm.W_o(x) .+ lstm.U_o(h))
    u = tanh.(lstm.W_u(x) .+ lstm.U_u(h))
    c_next = f .* c .+ i .* u
    h_next = o .* tanh.(c_next)
    return c_next, h_next
end

@inline function tree_lstm(lstm::TreeLSTM, tree::MyNodes, c::Vector{Float32}, h::Vector{Float32})
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

function (m::LikelihoodModel)(tree::MyNodes, c_init::Vector{Float32}=zeros(Float32, size(m.fc.weight)[2]), h_init::Vector{Float32}=zeros(Float32, size(m.fc.weight)[2]))::Float32
    h, _ = tree_lstm(m.tree_lstm, tree, c_init, h_init)
    h = m.fc(h)
    output = m.output_layer(h)
    return output[1]
end

# Initialize model and example tree
all_symbols = [:+, :-, :/, :*, :<=, :>=, :min, :max, :<, :>, :select]
in_dim = length(all_symbols) + 2
symbols_to_index = Dict(i=>ind for (ind, i) in enumerate(all_symbols))
out_dim = in_dim * 2
#ex = :(a - b + c * 100 / 200 - d - (a + b - c))
#ex = :((a - b) + (c / d))
ex = :(a - b + c) 

#symbol_encoding(sym::Symbol) = sym in all_symbols ? ones(length(all_symbols) + 2) * symbols_to_index[sym] : ones(length(all_symbols) + 2) * -2
#symbol_encoding(sym) = ones(length(all_symbols) + 2) * -1

function expression_encoder!(ex::Union{Expr, Symbol, Int64}, all_symbols::Vector{Symbol}, symbols_to_index::Dict{Symbol, Int64}, depth::Int, buffer::Dict{Int, Vector}, bag_ids::Vector)
    if ex isa Expr
        encoded_symbol_root = zeros(length(all_symbols) + 2)
        encoded_symbol_root[symbols_to_index[ex.args[1]]] = 1
        if depth == 1
            buffer[depth] = Vector[encoded_symbol_root]
        end
        depth += 1
        #push!(bag_ids, depth - 1 + )
        encoded_symbol_first_child = expression_encoder!(ex.args[2], all_symbols, symbols_to_index, depth, buffer, bag_ids)
        #encoded_symbol_first_child = symbol_encoding(ex.args[2])
        if haskey(buffer, depth)
            push!(buffer[depth], encoded_symbol_first_child)
        else
            buffer[depth] = Vector[encoded_symbol_first_child]
        end
        if length(ex.args) == 3
            #push!(bag_ids, depth - 1)
            encoded_symbol_second_child = expression_encoder!(ex.args[3], all_symbols, symbols_to_index, depth, buffer, bag_ids)
            push!(buffer[depth], encoded_symbol_second_child)
        end
        return encoded_symbol_root
    elseif ex isa Symbol && ex in all_symbols
        encoded_symbol = zeros(length(all_symbols) + 2)
        encoded_symbol[symbols_to_index[ex]] = 1
        return encoded_symbol
    elseif ex isa Symbol 
        encoded_symbol = zeros(length(all_symbols) + 2)
        encoded_symbol[end] = 1
        return encoded_symbol
    else
        encoded_symbol = zeros(length(all_symbols) + 2)
        encoded_symbol[end - 1] = 1
        return encoded_symbol
    end
end


buffer = Dict{Int, Vector}()
bag_ids = Vector()
expression_encoder!(ex, all_symbols, symbols_to_index, 1, buffer, bag_ids)
#@assert length(buffer[6]) == 2
#@assert length(buffer[5]) == 4
#=
loss(model::LikelihoodModel, expanded_nodes::AbstractVector, not_expanded_nodes::Nothing) = log(1 + exp(sum(model.(expanded_nodes))))
function loss(model::LikelihoodModel, expanded_nodes::AbstractVector, not_expanded_nodes::AbstractVector)
  return log(1 + exp(sum(model.(expanded_nodes)) - sum(model.(not_expanded_nodes))))
end

function loss(model::LikelihoodModel, first_any::Vector, second_any::Vector)
    my_loss = Float32[
        log(1 + exp(model(i) - model(j)))
        for (ind, i) in enumerate(first_any)
        for j in second_any[ind]
    ]
    return mean(my_loss)
end
=#
function lgbf_loss(model, h_pos, h_neg)
  #return isempty(h_neg) ? mean(log(1 + exp(model(h_pos)))) : mean(log.(1 .+ exp.(model(h_pos) .- model.(h_neg))))
  return mean(log(1 + exp(model(h_pos))))
end


function loss(model::LikelihoodModel, depth_dict::Dict{Int, Vector}, c_init::Vector{Float32}, h_init::Vector{Float32})
    #result = [lgbf_loss(model, i, second_any[ind])
              #for (ind, i) in enumerate(first_any)]
    #println(first_any)
    #println(length.(second_any))
   # return mean(result)
    #return mean(model(first_any[1]) .- model.(second_any[1]))
  #return mean(log.(1 .+ exp.(model.(first_any))))
  #return mean(log.(1 .+ exp.(model.(first_any, c_init, h_init))))
  #return mean([model(i, c_init, h_init) for i in first_any])
    return 1
end

#model = LikelihoodModel(in_dim, out_dim, out_dim * 2)


#result_prob = model(expression_encoder!(ex))
#println("Final result: $result_prob")

