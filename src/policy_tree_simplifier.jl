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
            ee = ex2mill(new_expr, symbols_to_index, all_symbols, collect(1:100)) 
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


function policy_loss_func(policy, big_vector, hp=nothing, hn=nothing, surrogate::Function = logistic)
    o = policy(big_vector) 

    diff = hn .* (o * hp) .- hn .* o[1,:]
    filtered_diff = filter(!=(0), diff)
    loss = sum(surrogate.(filtered_diff))
    return loss
end


mutable struct PolicyTrainingSample{D, HP, HN}
    training_data::D
    hp::HP
    hn::HN
end


function update_policy_training_samples(training_samples, symbols_to_index, all_symbols, theory)
    policy_training_samples = []
    for i in training_samples
        if isnothing(i.training_data)
            continue
        end
        a, hn, hp = tree_sample_to_policy_sample(i.proof, i.initial_expr, symbols_to_index, all_symbols, theory)
        push!(policy_training_samples, PolicyTrainingSample(a, hp, hn))
    end
    # @save "data/training_data/policy_training_samplesk$(length(policy_training_samples))_v5.jld2" policy_training_samples
end


function tree_sample_to_policy_sample(sample, ex, symbols_to_index, all_symbols, theory)
    ee = []
    hn = []
    hp = []
    for (j,jr) in sample
        applicable_rules = filter(r -> r[2] != ex, execute(ex, theory))
        # applicable_rules = execute(td, theory)
        final_index = 0
        # @show length(applicable_rules)
        if isempty(hp)
            tmp_hp = []
            tmp_hn = []
        else
            tmp_hp = zeros(length(hp[end]))
            tmp_hn = zeros(length(hn[end]))
        end
        for (ind, (r, new_expr)) in enumerate(applicable_rules)
            if r[1] == j && r[2] == jr
                final_index = ind
                push!(tmp_hp, 1)
                push!(tmp_hn, 0)
            else
                push!(tmp_hp, 0)
                push!(tmp_hn, 1)
            end
            push!(ee, ex2mill(new_expr, symbols_to_index, all_symbols, collect(1:100)))
        end
        @assert final_index != 0
        ex = applicable_rules[final_index][2]
        push!(hp, tmp_hp)
        push!(hn, tmp_hn)
    end
    max_length = maximum(length, hn)

    padding_value = 0
    padded_vectors = [vcat(vec, fill(padding_value, max_length - length(vec))) for vec in hn]
    hn = hcat(padded_vectors...)
    padded_vectors = [vcat(vec, fill(padding_value, max_length - length(vec))) for vec in hp]
    hp = hcat(padded_vectors...)
    # for i in 2:size(hn, 2)
    #     hn[:, i] .= hn[:, i] + hn[:, i - 1]
    # end

    # for i in size(hn, 2) - 1:-1:1
    #     hn[:, i] .= hn[:, i] + hn[:, i + 1]
    # end
    ds = reduce(catobs, ee) 
    return ds, Matrix{Int}(hn), Matrix{Int}(hp)
end


function str_to_expr1(string_tokens)
    if isempty(string_tokens)
        return 
    end
    tk = popfirst!(string_tokens)
    if tk == "(" 
        args = []
        tk_symbol = popfirst!(string_tokens)
        # for i in string_tokens[tk_visited + 1:end]
        size_args = tk_symbol == "!" ? 1 : 2
        while length(args) != size_args
            tmp = str_to_expr1(string_tokens)
            if !isnothing(tmp)
                push!(args, tmp)
            end
        end
        return Expr(:call, Symbol(tk_symbol), args...)
    end
    if tk == ")"
        return
    end
    if isnothing(tryparse(Int, tk))
        return Symbol(tk)
    else
        return parse(Int, tk)
    end
end


function caviar_data_parser(file)
    data = open(file, "r") do f
        return JSON.parse(f)
    end
    x = []
    y = []
    r = []
    for i in data
        # @show i
        tmp_x = i["expression"]["start"]
        expression_tokens = split(tmp_x, " ") 
        tmp_x = str_to_expr1(expression_tokens)
        tmp_y = i["expression"]["end"]
        tmp_r = i["rules"]
        push!(x , tmp_x)
        push!(y , tmp_y)
        push!(r , tmp_r)
    end 
    return x, y, r
end
