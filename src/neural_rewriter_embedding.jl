# Embedding any subtree vcat(root node embedding, subtree embedding) 

input_dim = 512
hidden_dim = 512 * 2
max_steps = 50
value_model = Chain(
    Dense(input_dim, hidden_dim, leakyrelu), Dense(hidden_dim, hidden_dim, leakyrelu), Dense(hidden_dim, 1)
)

policy_model = Chain(
    Dense(input_dim, hidden_dim, leakyrelu), Dense(hidden_dim, hidden_dim, leakyrelu), Dense(hidden_dim, length(theory))
)


function loss(rules_probs, rules_choosen, region_values, region_values_target, value_loss_coef=0.5)
    policy_loss = -sum(rules_probs .* rules_choosen)
    value_loss = smooth_l1_loss(region_values, region_values_target)
    total_loss = policy_loss + value_loss * value_loss_coef
    return policy_loss, value_loss, total_loss
end
# function cached_inference!(ex::Expr, cache, model, all_symbols, symbols_to_ind)
#     get!(cache, ex) do 
#         if ex.head == :call
#             fun_name, args =  ex.args[1], ex.args[2:end]
#         elseif ex.head in all_symbols
#             fun_name = ex.head
#             args = ex.args
#         else
#             error("unknown head $(ex.head)")
#         end
#         encoding = zeros(Float32, length(all_symbols))
#         encoding[symbols_to_index[fun_name]] = 1
#         args = cached_inference!(args, cache, model, all_symbols, symbols_to_ind)
#         if isa(model, ExprModel)
#             tmp = model.head_model(encoding)
#         else
#             tmp = model.expr_model.head_model(encoding, model.model_params.head_model)
#         end
#         tmp = vcat(tmp, args)
#     end
# end



# function cached_inference!(args::Vector, cache, model, all_symbols, symbols_to_ind)
#     l = length(args)
#     my_tmp = [cached_inference!(a, cache, model, all_symbols, symbols_to_ind) for a in args]
#     my_args = hcat(my_tmp...)
#     if l == 2
#         tmp = vcat(my_args, const_both)
#     else
#         tmp = vcat(my_args, const_one)
#     end

#     if isa(model, ExprModel)
#         tmp = model.joint_model(tmp)
#         a = model.aggregation(tmp,  Mill.AlignedBags([1:l])) 
#     else
#         tmp = model.expr_model.joint_model(tmp, model.model_params.joint_model)
#         a = model.expr_model.aggregation(tmp,  Mill.AlignedBags([1:l])) 
#     end
#     return a[:,1]
# end

function get_all_subtrees!(ex, all_subtrees::Set)
    if !isa(ex, Expr)
        return
    end
    push!(all_subtrees, ex)
    tmp = length(ex.args) > 2 ? 2 : 1
    for i in tmp:length(ex.args)
        get_all_subtrees!(ex.args[i], all_subtrees)
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
        get_all_subtrees!(ex, all_subtrees)
        subtree_embedding = []
        input_embeddings = zeros(length(tmp) * 2, length(all_subtrees))
        for (i, subtree) in enumerate(all_subtrees)
            a = vcat(tmp, cache[subtree])
            push!(subtree_embedding, (subtree, a))
            input_embeddings[:, i] = a
        end
        region_values = value_model(input_embeddings)
        rules_probs = policy_model(input_embeddings)
        rule_to_apply = argmax(rules_probs)
        subtree_to_change = argmax(region_values)
        push!(tree_traces, ex)
        push!(rule_probs_traces, rules_probs)
        push!(region_values_traces, region_values)
        my_rewriter!(subtree_to_change, ex, theory[rule_to_apply])
    end
    return tree_traces, rule_probs_traces, region_values_traces
end


function get_reward(tree_traces, max_steps=max_steps, gamma=0.9)
    decay_coef = 1.0
    initial_expression = tree_traces[1]
    current_reward = exp_size(initial_expression) - exp_size(tree_traces[2]) - 1 
    for i in 2:max_steps-1
        current_reward = max(decay_coef * min(exp_size(tree_traces[i]) - exp_size(tree_traces[i+1]) - 1, exp_size(initial_expression)), current_reward)
        decay_coef *= gamma
    end
    current_reward /= exp_size(initial_expression)
    region_value_target = current_reward
    return  region_value_target
end
