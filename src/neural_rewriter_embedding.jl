# Embedding any subtree vcat(root node embedding, subtree embedding) 

input_dim = 512
hidden_dim = 512 * 2
max_steps = 50
value_model = Chain(
    Dense(input_dim, hidden_dim, leakyrelu), Dense(hidden_dim, hidden_dim, leakyrelu), Dense(hidden_dim, 1)
)

# policy_model = Chain(
#     Dense(input_dim, hidden_dim, leakyrelu), Dense(hidden_dim, hidden_dim, leakyrelu), Dense(hidden_dim, length(theory)), Dense(length(theory), 1)
# )

policy_model = Chain(
    Dense(input_dim, hidden_dim, leakyrelu), Dense(hidden_dim, hidden_dim, leakyrelu), Dense(hidden_dim, length(theory))
)

myex = :( (v0 + v1) + 119 <= min((v0 + v1) + 120, v2) && ((((v0 + v1) - v2) + 127) / (8 / 8) + v2) - 1 <= min(((((v0 + v1) - v2) + 134) / 16) * 16 + v2, (v0 + v1) + 119))

function loss(rules_probs, rules_choosen, region_values, region_values_target, value_loss_coef=0.5)
    policy_loss = -sum(rules_probs .* rules_choosen)
    value_loss = smooth_l1_loss(region_values, region_values_target)
    total_loss = policy_loss + value_loss * value_loss_coef
    return policy_loss, value_loss, total_loss
end


function get_all_subtrees!(ex, pos, all_subtrees::Set)
    if !isa(ex, Expr)
        return
    end
    push!(all_subtrees, (ex, copy(pos)))
    # tmp = length(ex.args) > 2 ? 2 : 1
    for (ind,i) in enumerate(ex.args)
        push!(pos, ind)
        get_all_subtrees!(i, pos, all_subtrees)
        pop!(pos)
    end
end


function forward(policy_model, value_model, embedding_heuristic, ex, max_steps=max_steps)
    tree_traces = []
    rule_probs_traces = []
    region_values_traces = []
    cache = LRU(maxsize=1000)
    for _ in 1:max_steps
        tmp = MyModule.cached_inference!(ex, cache, embedding_heuristic, new_all_symbols, sym_enc)
        all_subtrees = Set()
        get_all_subtrees!(ex, Int[], all_subtrees)
        subtree_embedding = []
        input_embeddings = zeros(length(tmp) * 2, length(all_subtrees))
        # @show tmp
        for (i, subtree) in enumerate(all_subtrees)
            tree, pos = subtree
            a = vcat(tmp, cache[tree])
            push!(subtree_embedding, (subtree, a))
            input_embeddings[:, i] = a
        end
        region_values = value_model(input_embeddings)
        rules_probs = policy_model(input_embeddings)
        subtree_to_change = argmax(region_values)
        rule_to_apply = argmax(rules_probs[:, subtree_to_change[2]])
        push!(tree_traces, ex)
        push!(rule_probs_traces, rules_probs)
        push!(region_values_traces, region_values)
        @show collect(all_subtrees)[subtree_to_change[2]]
        @show rule_to_apply
        subtree, subtree_path = collect(all_subtrees)[subtree_to_change[2]]
        MyModule.my_rewriter!(subtree_path, ex, theory[rule_to_apply])
    end
    return tree_traces, rule_probs_traces, region_values_traces
end


function get_reward(tree_traces, max_steps=max_steps, gamma=0.9)
    decay_coef = 1.0
    size_cache = LRU(maxsize=10000)
    initial_expression = tree_traces[1]
    initial_size = MyModule.exp_size(initial_expression, size_cache)
    current_reward = initial_size - MyModule.exp_size(tree_traces[2], size_cache) - 1 
    for i in 2:max_steps-1
        current_reward = max(decay_coef * min(MyModule.exp_size(tree_traces[i], size_cache) - MyModule.exp_size(tree_traces[i+1], size_cache) - 1,initial_size), current_reward)
        decay_coef *= gamma
    end

    current_reward /= initial_size
    region_value_target = current_reward
    return  region_value_target
end
