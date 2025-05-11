using GraphNeuralNetworks
using Flux
using SparseArrays
using MyModule.Metatheory
using MyModule.Metatheory: ENodeLiteral
using MyModule.Metatheory.EGraphs: rebuild!
using MyModule: get_leaf_args
include("egraph_api.jl")


get_number_of_enodes(g::EGraph) = length(unique(keys(g.memo)))
function extract_adjacency_matrix1(g::EGraph)
    @assert length(keys(g.memo)) == length(unique(keys(g.memo)))
    num_nodes = get_number_of_enodes(g)
    adj_matrix = zeros(Int, num_nodes, num_nodes)
    for (k, v) in g.memo
        if k isa ENodeTerm
            for j in k.args
                adj_matrix[v, j] = 1
            end
        end
    end
    return adj_matrix
end

ex = :(v0 - 12 <= v0)
env = EGraphEnv(ex, [])
actions = action_space(env)
for a in actions
    act!(env, a)
end
adj_matrix = extract_adjacency_matrix1(env.egraph)
adj_sparse = sparse(adj_matrix)

function adj_to_edge_index(adj::SparseMatrixCSC)
    rows, cols, _ = findnz(adj)
    return hcat(rows, cols)
end

function encode_egraph(eg::EGraph)
    encoding = zeros(Float32, length(all_symbols), length(eg.memo))
    for v in keys(eg.memo)
        if v isa ENodeLiteral
            symbol_index, encoding_value = get_leaf_args(v.value)
        else
            symbol_index, encoding_value = get_leaf_args(v.operation)
        end
        encoding[symbols_to_ind[symbol_index]] = encoding_value
    end
    return encoding
end

update_epsilon(epsilon; eps_decay=0.95, eps_min=0.1) = max(eps_min, eps_decay * epsilon)

edge_index = adj_to_edge_index(adj_sparse)
gnn_graph = GNNGraph(edge_index[:, 1], edge_index[:, 2])

# Experiment with model architecture and parameters
input_size = length(all_symbols)
hidden_size = input_size * 2
model = GNNChain(GCNConv(input_size => hidden_size),
                BatchNorm(hidden_size),
                x -> relu.(x),     
                GCNConv(hidden_size => hidden_size, relu),
                GlobalPool(mean),
                Dense(hidden_size, length(theory)))

# Base.isequal(x::RewriteRule, y::RewriteRule) = isequal(x.left, y.left) && isequal(x.right, y.right)
# Base.isequal(x::PatTerm, y::PatTerm) = isequal(x.args, y.args) && isequal(x.exprhead, y.exprhead) && isequal(x.operation, y.operation)
# Base.isequal()
# Base.isequal(x::AbstractPat, y::AbstractPat) = isequal(x.args, y.args) && isequal(x.exprhead, y.exprhead) && isequal(x.operation, y.operation)
# Base.isequal(x::PatVar, y::PatVar) = x == y
function (m::GNNChain)(g::EGraph)
    adj_matrix = extract_adjacency_matrix1(g)
    adj_sparse = sparse(adj_matrix)
    edge_index = adj_to_edge_index(adj_sparse)
    gnn_graph = GNNGraph(edge_index[:, 1], edge_index[:, 2])
    x = encode_egraph(g)
    m(gnn_graph, x)
end

function train(data, model; max_iter=100, max_steps=50, epsilon=1.0, gamma=0.99, target_update=10)
    env = EGraphEnv(data[1], model, max_expansions=max_steps)
    MyModule.reset_all_function_caches()
    pol_optimizer = ADAM(lr)
    online_q_params = Flux.setup(pol_optimizer, env.model)
    target_update_freq = 0
    for itr in 1:max_iter
        for ex in data
            env.initial_expr = ex
            reset!(env)
            for _ in 1:max_steps
                possible_actions, rule_ind = action_space(env)
                if epsilon <= rand()
                    o = model(env.egraph)
                    a = possible_actions[argmax(o[rule_ind])]
                else
                    a = rand(possible_actions)
                end
                s = state(env)
                act!(env, a)
                r = reward(env)
                ns = state(env)
                add2buffer!(my_buffer, (s, r, ns))
                target_update_freq += 1
                if length(my_buffer.buffer) >= batch_size
                    update_ddqn!(my_buffer, env,model, target_gnn, online_q_params, batch_size=batch_size)
                    if mod(target_update_freq, target_update) == 0
                        # soft_update!(online_q, target_q, 0.5)
                        target_gnn = deepcopy(env.model)
			            target_update_freq = 0
                    end
                end
            end
            epsilon = update_epsilon(epsilon)
        end
    end
end
# Apply the GCN layer
output = model(gnn_graph, x)