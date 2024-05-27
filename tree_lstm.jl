using Metatheory
using Flux
#: Dense, sigmoid, tanh, relu, \sigmoid

struct MyNodes
    children::Union{Vector{MyNodes}, Nothing}
    encoding::Vector{UInt32}
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
    out_dim::Int
end

TreeLSTM(in_dim::Int, out_dim::Int) = TreeLSTM(
    Dense(in_dim, out_dim),
    Dense(in_dim, out_dim),
    Dense(in_dim, out_dim),
    Dense(in_dim, out_dim),
    Dense(out_dim, out_dim),
    Dense(out_dim, out_dim),
    Dense(out_dim, out_dim),
    Dense(out_dim, out_dim),
    out_dim
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

function tree_lstm(lstm::TreeLSTM, tree::MyNodes, c::Vector{Float32}, h::Vector{Float32})
    if isnothing(tree.children)
        data = tree.encoding
        return lstm(data, c, h)
    else
        for child in tree.children
          c, h = tree_lstm(lstm, child, c, h)
        end
        data = tree.encoding
        return lstm(data, c, h)
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

function (m::LikelihoodModel)(tree::MyNodes)
    out_dim = m.tree_lstm.out_dim
    c_init = zeros(Float32, out_dim)
    h_init = zeros(Float32, out_dim)
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
ex = :(a - b + c)

function expression_encoder!(ex)
    if ex isa Expr
        #println("new root node $ex")
        encoded_symbol = zeros(length(all_symbols) + 2)
        encoded_symbol[symbols_to_index[ex.args[1]]] = 1
        new_root = MyNodes(Vector(), encoded_symbol)
        if length(ex.args) == 3
          push!(new_root.children, expression_encoder!(ex.args[2]))
          push!(new_root.children, expression_encoder!(ex.args[3]))
        else
          push!(new_root.children, expression_encoder!(ex.args[2]))
        end
        return new_root
    elseif ex isa Symbol && ex in all_symbols
        #println("new alg expressio node")
        encoded_symbol = zeros(length(all_symbols) + 2)
        encoded_symbol[symbols_to_index[ex]] = 1
        return MyNodes(nothing, encoded_symbol)
    elseif ex isa Symbol 
        #println("new variable node")
        encoded_symbol = zeros(length(all_symbols) + 2)
        encoded_symbol[end] = 1
        return MyNodes(nothing, encoded_symbol)
    else
        #println("new integer node")
        encoded_symbol = zeros(length(all_symbols) + 2)
        encoded_symbol[end - 1] = 1
        return MyNodes(nothing, encoded_symbol)
    end
end

#loss(mode::LikelihoodModel, )
loss(model::LikelihoodModel, expanded_nodes::AbstractVector, not_expanded_nodes::Nothing) = log(1 + exp(sum(model.(expanded_nodes))))
function loss(model::LikelihoodModel, expanded_nodes::AbstractVector, not_expanded_nodes::AbstractVector)
  return log(1 + exp(sum(model.(expanded_nodes)) - sum(model.(not_expanded_nodes))))
end
#loss(model::LikelihoodModel, expanded_nodes::AbstractVector, not_expanded_nodes::AbstractVector) = log(1 + exp(sum(model.(expanded_nodes)) - sum(model.(not_expanded_nodes))))
#model = LikelihoodModel(in_dim, out_dim, out_dim * 2)


#result_prob = model(expression_encoder!(ex))
#println("Final result: $result_prob")

