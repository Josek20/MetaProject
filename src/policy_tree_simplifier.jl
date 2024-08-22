function count_numbers_of_circles(depth_dict, ex)
    circles_dict = Dict{Expr, Int}() 
    prev_ex = ex
    for (d, rules) in depth_dict
        applied_rule = rules[1]
        new_ex = apply!(prev_ex, applied_rule)
        if haskey(circles_dict, new_ex)
            circles_dict[new_ex] += 1
        else
            circles_dict[new_ex] = 0
        end
    end
    return sum(values(circles_dict))
end


function compute_reward(old_ex, new_ex)
    old_size = exp_size(old_ex)
    new_size = exp_size(new_ex)
    return old_size - new_size 
end


function policy_loss(correct_rule_prob, rest_rule_prob)
    return sum(log.(1 .+ exp.(correct_rule_prob .- rest_rule_prob))) 
end


function build_tree(ex, policy, theory, max_itr, symbols_to_index, all_symbols)
    counter = 0
    proof = []
    while max_itr != counter
        applicable_rules = filter(r -> r[2] != ex, execute(ex, theory))
        if length(applicable_rules) == 0
            break
        end
        probs = []
        for (rule_id, new_expr) in applicable_rules
            ee = ex2mill(new_expr, symbols_to_index, all_symbols)   
            push!(probs, only(policy(ee)))
        end
        chosen_expr_index = argmin(probs)
        ex = applicable_rules[chosen_expr_index][2]
        push!(proof, applicable_rules[chosen_expr_index][1])
        @show ex
        if ex == 1
            break
        end
        counter += 1 
    end 
    return ex, proof
end

# function test_heuristic(heuristic, data, max_steps, max_depth)
#     result = []
#     result_proof = []
#     simp_expressions = []
#     for (index, i) in enumerate(data)
#         # simplified_expression, _, _, _, _, _, proof_vector = heuristic_forward_pass(heuristic, i, max_steps, max_depth)
#         simplified_expression, depth_dict, big_vector, saturated, hp, hn, root, proof_vector = heuristic_forward_pass(heuristic, i, max_steps, max_depth)
#         original_length = exp_size(i)
#         simplified_length = exp_size(simplified_expression)
#         push!(result, original_length - simplified_length)
#         push!(result_proof, proof_vector)
#         push!(simp_expressions, simplified_expression)
#     end
#     return result, result_proof, simp_expressions
# end

function test_policy(policy, data, theory, max_itr, symbols_to_index, all_symbols)
    result = []
    result_proof = []
    simp_expressions = []
    for (index, i) in enumerate(data)
        @show i
        simplified_expression, proof = build_tree(i, policy, theory, max_itr, symbols_to_index, all_symbols) 
        @show simplified_expression
        original_length = exp_size(i)
        simplified_length = exp_size(simplified_expression)
        push!(result, original_length - simplified_length)
        push!(result_proof, proof)
        push!(simp_expressions, simplified_expression)
    end
    return result, result_proof, simp_expressions
end


function policy_loss_func1(policy, ee, yj)
    p = policy(ee)
    diff = p[1, yj] .- p
    filtered_diff = filter(x-> x != 0, diff)
    loss = sum(log.(1 .+ exp.(filtered_diff)))
end


function policy_loss_func2(policy, ee, yj, hp, hn)
    p = policy(ee)
    pp = p * hp
    pn = p[1, :] .* hn
    diff = 0
    for i in 1:size(pp)[2]
          
        diff += sum(log.(1 .+ exp.(filtered_diff)))
    end
    # filtered_diff = filter(x-> x != 0, diff)
    # loss = sum(log.(1 .+ exp.(filtered_diff)))
end


function policy_loss_func(heuristic, big_vector, hp=nothing, hn=nothing)
    o = heuristic(big_vector) 
    p = (o * hp) .* hn

    diff = p - o[1, :] .* hn
    filtered_diff = filter(x-> x != 0, diff)
    return sum(log.(1 .+ exp.(diff)))
end


function tree_sample_to_policy_sample(sample, ex, symbols_to_index, all_symbols)
    ee = []
    hn = []
    hp = []
    for (j,jr) in sample
        applicable_rules = filter(r -> r[2] != ex, execute(ex, theory))
        # applicable_rules = execute(td, theory)
        final_index = 0
        # @show length(applicable_rules)
        for (ind, (r, new_expr)) in enumerate(applicable_rules)
            if r[1] == j && r[2] == jr
                final_index = ind
                push!(hp, 1)
            else
                push!(hp, 0)
            end
            push!(ee, ex2mill(new_expr, symbols_to_index, all_symbols))
        end
    end
    ds = reduce(catobs, ee) 
    return ds, hn, hp
end
