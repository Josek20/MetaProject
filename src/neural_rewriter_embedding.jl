# Embedding any subtree vcat(root node embedding, subtree embedding) 

input_dim = 512
hidden_dim = 256
max_steps = 50
value_model = Chain(
    Dense(input_dim, hidden_dim, leakyrelu), Dense(hidden_dim, hidden_dim, leakyrelu), Dense(hidden_dim, 1)
)

# policy_model = Chain(
#     Dense(input_dim, hidden_dim, leakyrelu), Dense(hidden_dim, hidden_dim, leakyrelu), Dense(hidden_dim, length(theory)), Dense(length(theory), 1)
# )

rules_groups = length(theory)
rules_groups = 10
policy_model = Chain(
    Dense(input_dim, hidden_dim, leakyrelu), Dense(hidden_dim, hidden_dim, leakyrelu), Dense(hidden_dim, rules_groups)
)

# myex = :( (v0 + v1) + 119 <= min((v0 + v1) + 120, v2) && ((((v0 + v1) - v2) + 127) / (8 / 8) + v2) - 1 <= min(((((v0 + v1) - v2) + 134) / 16) * 16 + v2, (v0 + v1) + 119))
myex = :(v0 - 102 <= v0 - 102)
# function loss(rules_probs, rules_choosen, region_values, region_values_target, value_loss_coef=0.5)
#     policy_loss = -sum(rules_probs .* rules_choosen)
#     value_loss = smooth_l1_loss(region_values, region_values_target)
#     total_loss = policy_loss + value_loss * value_loss_coef
#     return policy_loss, value_loss, total_loss
# end


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
    tree_traces = [ex]
    rule_probs_traces = []
    region_values_traces = []
    rules_applied = []
    subtree_embeddings_traces = []
    cache = LRU(maxsize=1000)
    for _ in 1:max_steps
        # @show cache
        tmp = MyModule.cached_inference!(ex, cache, embedding_heuristic, new_all_symbols, sym_enc)
        if isempty(tmp)
            error("cached_inference! returned empty results for the expression: $ex")
        end
        all_subtrees = Set()
        get_all_subtrees!(ex, Int[], all_subtrees)
        # @show length(all_subtrees), tmp, cache
        subtree_embedding = []
        input_embeddings = zeros(length(tmp) * 2, length(all_subtrees))
        # @show tmp
        for (i, subtree) in enumerate(all_subtrees)
            tree, pos = subtree
            a = vcat(tmp, cache[tree])
            push!(subtree_embedding, (subtree, a))
            input_embeddings[:, i] = a
        end
        region_values1 = value_model(input_embeddings)
        region_values = softplus(region_values1)
        rules_probs = policy_model(input_embeddings)
        subtree_to_change = argmax(region_values)
        rule_to_apply = argmax(rules_probs[:, subtree_to_change[2]])
        push!(tree_traces, ex)
        push!(subtree_embeddings_traces, input_embeddings[:, subtree_to_change[2]])
        # push!(rule_probs_traces, rules_probs[:, subtree_to_change[2]])
        # push!(region_values_traces, region_values[1,:])
        push!(rule_probs_traces, maximum(rules_probs[:, subtree_to_change[2]]))
        push!(region_values_traces, maximum(region_values[1,:]))
        push!(rules_applied, rule_to_apply)
        # @show collect(all_subtrees)[subtree_to_change[2]]
        # @show rule_to_apply
        subtree, subtree_path = collect(all_subtrees)[subtree_to_change[2]]
        # old_ex = copy(ex)
        for i in theory[(rule_to_apply - 1) * rules_groups + 1: rule_to_apply * rules_groups]
            o = MyModule.my_rewriter!(subtree_path, ex, i)
            if !(o isa Nothing)
                ex = o
                break
            end
        end
    end
    return tree_traces, rule_probs_traces, region_values_traces, rules_applied, subtree_embeddings_traces
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


# total_loss(loss_rule_picking, loss_region_picking; alpha=10) = loss_rule_picking + alpha*loss_region_picking 


# function reward(s1::Expr, s2::Expr, size_cache)
#     return MyModule.exp_size(s1, size_cache) - MyModule.exp_size(s2, size_cache)
# end


# function loss_region_picking(tree_traces, region_values_traces; gamma=0.9, max_steps=max_steps)
#     size_cache = LRU(maxsize=1000)
#     total_loss = []
#     for j in 1:max_steps
#         tmp = 0
#         for i in j:max_steps
#             tmp += gamma^(i - j) * reward(tree_traces[i], tree_traces[i + 1], size_cache) - region_values_traces[i] 
#         end
#         push!(total_loss, tmp^2)
#     end
#     return mean(total_loss)
# end


# function loss_rule_picking(tree_traces, rule_probs_traces; gamma=0.9, max_steps=max_steps)
#     size_cache = LRU(maxsize=1000)
#     total_loss = []
#     for j in 1:max_steps
#         tmp = 0
#         for i in j:max_steps
#             tmp += gamma^(i - j) * reward(tree_traces[i], tree_traces[i + 1], size_cache) - rule_probs_traces[i] 
#         end
#         push!(total_loss, tmp * log(rule_probs_traces[j]))
#     end
#     return -sum(total_loss)
# end


function loss(policy_model, value_model, input_values, rules_applied, rewards; gamma=0.9, alpha=10, max_steps=max_steps)
    total_loss_region = 0
    total_loss_rules = 0
    for j in 1:max_steps
        tmp1 = 0
        tmp2 = 0
        for i in j:max_steps
            reward = rewards[i]
            k = policy_model(input_values[i])
            k = k[rules_applied[i]]
            tmp1 += gamma^(i - j) * reward - k
            tmp2 += gamma^(i - j) * reward - only(value_model(input_values[i]))
        end
        total_loss_rules += tmp1 * log(policy_model(input_values[j])[rules_applied[j]])
        total_loss_region += tmp2^2
    end
    total_loss = -total_loss_rules + alpha * total_loss_region / max_steps
    return total_loss
end


function loss(tree_traces, rule_probs_traces, region_values_traces, size_cache, ; gamma=0.9, alpha=10, max_steps=max_steps)
    total_loss_region = []
    total_loss_rules = []
    for j in 1:max_steps
        tmp1 = 0
        tmp2 = 0
        for i in j:max_steps
            reward = MyModule.exp_size(tree_traces[i], size_cache) - MyModule.exp_size(tree_traces[i + 1], size_cache)
            tmp1 += gamma^(i - j) * reward - rule_probs_traces[i]
            tmp2 += gamma^(i - j) * reward - region_values_traces[i]
        end
        push!(total_loss_rules, tmp1 * log(rule_probs_traces[j]))
        push!(total_loss_region, tmp2^2)
    end
    total_loss = -sum(total_loss_rules) + alpha * mean(total_loss_region)
    return total_loss
end

ps = Flux.params(policy_model, value_model)
optimizer = Adam()
size_cache = LRU(maxsize=100_000)
for ep in 1:10
    for ex in sorted_train_data[1:100]
        tree_traces, rule_probs_traces, region_values_traces, rules_applied, subtree_embeddings_traces = forward(policy_model, value_model, heuristic, ex)
        rewards = []
        for i in 1:length(tree_traces) - 1
            rew = MyModule.exp_size(tree_traces[i], size_cache) - MyModule.exp_size(tree_traces[i+1], size_cache)
            push!(rewards, rew)
        end
        sa, grad = Flux.Zygote.withgradient(ps) do
            loss(policy_model, value_model, subtree_embeddings_traces, rules_applied, rewards)
        end
        @show tree_traces
        @show sa
        Flux.update!(optimizer, ps, grad)
    end
end