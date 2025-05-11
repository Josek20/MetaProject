using MyModule
using MyModule.Mill
using MyModule.Flux
using MyModule.DataStructures
using Optimisers
using Serialization
using BenchmarkTools
using ProfileCanvas

using MyModule: load_data, preprosses_data_to_expressions
using MyModule: exp_size, initialize_tree_search, all_expand, intern!, Node, build_tree, expand_node!, NodeID, TreePolicyModel, Node, push_to_tree!, extract_smallest_node, OnlyNode

experiment_name = "monte_carlo_alpha_zero"
train_data_path = "./data/neural_rewrter/train.json"
train_data = load_data(train_data_path)[1:1_000]
train_data = filter(x->!occursin("select", x[1]), train_data)
train_data = preprosses_data_to_expressions(train_data)
sorted_data = sort(train_data, by=x->MyModule.exp_size(x))
data = sorted_data



input_dim = 512
hidden_dim = 256
max_steps = 50
hidden_size = 64

function ffnn(idim, hidden_size, layers)
    layers == 1 && return Dense(idim, hidden_size, Flux.gelu)
    layers == 2 && return Flux.Chain(Dense(idim, hidden_size, Flux.gelu), Dense(hidden_size, hidden_size, Flux.gelu))
end

head_model = ProductModel(
    (;head = ffnn(length(new_all_symbols), hidden_size, 1),
      args = ffnn(hidden_size, hidden_size, 1),  
        ),
    ffnn(2*hidden_size, hidden_size, 1)
    )

args_model = ProductModel(
    (;args = ffnn(hidden_size, hidden_size, 1),  
      position = Dense(2,hidden_size),  
        ),
    ffnn(2*hidden_size, hidden_size, 1)
    )

value_model = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    Flux.Chain(Dense(hidden_size, hidden_size, Flux.gelu), Dense(hidden_size, hidden_size, Flux.gelu), Dense(hidden_size, 1)),
    );


policy_model = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    Flux.Chain(Dense(hidden_size, hidden_size, Flux.gelu), Dense(hidden_size, hidden_size, Flux.gelu), Dense(hidden_size, length(theory)), softmax)
    );


# myex = :( (v0 + v1) + 119 <= min((v0 + v1) + 120, v2) && ((((v0 + v1) - v2) + 127) / (8 / 8) + v2) - 1 <= min(((((v0 + v1) - v2) + 134) / 16) * 16 + v2, (v0 + v1) + 119))
myex = :(v0 - 102 <= v0 - 102)

function my_rewrite!(ex::NodeID, pos, new_exp_part::NodeID)
    if isempty(pos)
        return new_exp_part
    end
    node = MyModule.nc[ex]

    if pos[1] == 1
        new_part = my_rewrite!(node.left, pos[2:end], new_exp_part)
        # node.left = new_part
        new_node = OnlyNode(node.head, node.iscall, node.v, new_part, node.right)
    else        
        new_part = my_rewrite!(node.right, pos[2:end], new_exp_part)
        # node.right = new_part
        new_node = OnlyNode(node.head, node.iscall, node.v, node.left, new_part)
    end
    return get!(MyModule.nc, new_node)
end


function get_all_subtrees!(ex::NodeID, all_subtrees::Vector, parent=[])
    node = MyModule.nc[ex]
    for (ind,i) in enumerate([node.left, node.right])
        if !(MyModule.nc[i].iscall) && MyModule.nc[i].head ∉ [:&&, :||]
            continue
        end
        if i == MyModule.nullid
            continue
        end
        pos = vcat(parent, ind)
        push!(all_subtrees, (i, pos))
        get_all_subtrees!(i, all_subtrees, pos)
    end
end


function forward(ex, policy_model, value_model;max_expansions=50)
    soltree = Dict{UInt64, Node}()
    current = Node(ex, (), hash(ex), 0)
    soltree[current.node_id] = current
    smallest_node = current
    smallest_node_size = exp_size(ex)
    rules_applied = []
    rewrite_sequence_subtree = []
    rewrite_sequence = [ex]
    for step in 1:max_expansions
        ex = current.ex
        subtrees = [(ex, [])]
        get_all_subtrees!(ex, subtrees)
        # @show subtrees
        tree_action = []
        for (st, pos) in subtrees
            new_ex = [(intern!(r(st)), pos, ind, st) for (ind,r) in enumerate(theory)]
            # filter empty rewrites
            filtered_ex = filter(x->!isnothing(x[1]), new_ex)
            # @show filtered_ex
            new_ex = map(x->(x..., my_rewrite!(ex, x[2], x[1])), filtered_ex)
            # @show length(new_ex)
            # filter repeated
            filtered_ex = filter(x->!haskey(soltree, hash(x[5])), new_ex)
            append!(tree_action, filtered_ex)
        end
        isempty(tree_action) && break
        # if isempty(tree_action)
        #     push!(rewrite_sequence, subtree)
        #     break
        # end
        root_embedding = MyModule.general_cached_inference(ex, Expr, policy_model)
        o = map(tree_action) do (_, _, rule_id, subtree, _)
            # In the original paper concatenated the root embedding to the subtree
            subtree_embedding = MyModule.general_cached_inference(subtree, Expr, policy_model)
            (only(value_model.heuristic(subtree_embedding)), only(policy_model.heuristic(subtree_embedding)[rule_id]))
        end
        min_index = argmax(o)
        _, pos, rule_id, subtree, current_exp = tree_action[min_index]
        current_size = exp_size(current_exp)
        next_current = Node(current_exp, (pos, rule_id), current.node_id, current.depth + 1)
        soltree[next_current.node_id] = next_current
        push!(current.children, next_current.node_id)
        current = next_current
        push!(rules_applied, rule_id)
        push!(rewrite_sequence_subtree, subtree)
        push!(rewrite_sequence, current_exp)
        if smallest_node_size > current_size
            smallest_node_size = current_size
            smallest_node = current
        end
    end
    return smallest_node, rewrite_sequence_subtree, rules_applied, rewrite_sequence
end

 
function proper_loss(policy_model, value_model, rewriting_sequence, rewards, rules_applied; gamma=0.9, alpha=10)
    seq_len = length(rewards)
    Q = MyModule.heuristic(value_model, rewriting_sequence)
    P = MyModule.heuristic(policy_model, rewriting_sequence)
    # Q = value_model(rewriting_sequence)
    # P = policy_model(rewriting_sequence)
    v_tmp = 0
    p_tmp = 0
    for t in 1:seq_len - 1
        r = rewards[t:seq_len]
        g = gamma .^ (collect(0:seq_len-t))
        tmp = abs(sum(g .* r - Q[t:seq_len]))
        v_tmp += tmp ^ 2
        p_tmp += tmp * log(P[rules_applied[t], t] + 1e-10) # P contains Float32 to avoid log(0) have added 1e-10
    end
    return -p_tmp + (alpha * v_tmp) / seq_len 
end


optimizer=ADAM()

value_opt_state = Flux.setup(optimizer, value_model)
policy_opt_state = Flux.setup(optimizer, policy_model)
epochs = 1
inner_epochs = 10
experiment_name = "test_no_boosting_ep$(epochs)_inep$(inner_epochs)_abs_loss"
# @assert 0 == 1
training_data = [(;subtree_embeddings_traces=nothing,rewards=[],rules_applied=[], rewrite_sequence=[], initial_expr=intern!(i), r=-1) for i in data]
for ep in 1:epochs
    t = @elapsed training_data = map(training_data) do d
        ex = d.initial_expr
        @show ex
        smallest_node, rewrite_sequence_subtree, rules_applied, rewrite_sequence = forward(ex, policy_model, value_model; max_expansions=50)
        best_size = exp_size.(rewrite_sequence[2:end])
        # if min(best_size...) < exp_size(ex) || isnothing(d.subtree_embeddings_traces)
        # @show rewrite_sequence[2:end][argmin(best_size)]
        rewards = Float32[]
        for i in 1:length(rewrite_sequence) - 1
            rew = exp_size(rewrite_sequence[i]) - exp_size(rewrite_sequence[i + 1])
            push!(rewards, rew)
        end
        subtree_embeddings_traces = MyModule.no_reduce_multiple_fast_ex2mill([MyModule.expr(MyModule.nc, i) for i in rewrite_sequence_subtree])
        (;subtree_embeddings_traces=subtree_embeddings_traces,rewards=rewards,rules_applied=rules_applied, rewrite_sequence=rewrite_sequence[2:end], initial_expr=d.initial_expr, r=min(best_size...))
        # else
        #     d
        # end
    end
    @show t
    for _ in 1:inner_epochs
        total_loss = 0
        tt = @elapsed for (ind, (subtree_embeddings_traces, rewards, rules_applied, rewrite_sequence, _, _)) in enumerate(training_data)
            # subtree_embeddings_traces = policy_model(subtree_embeddings_traces) # both networks will compute only the output of the embedding
            sa, grad = Flux.Zygote.withgradient(value_model, policy_model) do vm, pm
                proper_loss(pm, vm, subtree_embeddings_traces, rewards, rules_applied)
            end
            if isinf(sa)
                serialize("models/neural_rewriter_value_model_loss_inf.bin", value_model)
                serialize("models/neural_rewriter_policy_model_loss_inf.bin", policy_model)
                break
            end
            if isnan(sa)
                serialize("models/neural_rewriter_value_model_loss_nan.bin", value_model)
                serialize("models/neural_rewriter_policy_model_loss_nan.bin", policy_model)
                break
            end
            total_loss += sa
            Optimisers.update!(value_opt_state, value_model, grad[1])
            Optimisers.update!(policy_opt_state, policy_model, grad[2])
        end
        @show total_loss / length(training_data)
        @show tt
    end
    validation_stats = [exp_size(i.initial_expr) - i.r for i in training_data]
    @show sum(validation_stats) / length(training_data)
end
serialize("models/trained_neural_rewriter_value_model_$(experiment_name).bin", value_model)
serialize("models/trained_neural_rewriter_policy_model_$(experiment_name).bin", policy_model)