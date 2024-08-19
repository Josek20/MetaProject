include("src/small_data_loader.jl")
include("tree_lstm.jl")
using JSON
using JLD2
using Plots
using BSON
using CSV
using Metatheory
using Metatheory.Rewriters
using DataStructures
using DataFrames



train_data_path = "data/train.json"
test_data_path = "data/test.json"

train_data = isfile(train_data_path) ? load_data(train_data_path)[1:1000] : load_data(test_data_path)[1:1000]
test_data = load_data(test_data_path)[1:1000]
train_data = preprosses_data_to_expressions(train_data)
test_data = preprosses_data_to_expressions(test_data)
#data = [:(a - b + c / 109 * 109)]
#abstract type LikelihoodModel end
theory = @theory a b c begin
    a::Number + b::Number => a + b
    a::Number - b::Number => a - b
    a::Number * b::Number => a * b
    a::Number / b::Number => a / b
    #(a / b::Number) * c::Number => :($a * div(c, b))
    #(a * b::Number) / c::Number => :($a * div(b, c))


    a * (b * c) --> (a * b) * c
    (a * b) * c --> a * (b * c)
    a + (b + c) --> (a + b) + c
    (a + b) + c --> a + (b + c)
    a + (b + c) --> (a + c) + b
    (a + b) + c --> (a + c) + b
    (a - b) + b --> a
    (-a + b) + a --> b


    a + b --> b + a
    a * (b + c) --> (a * b) + (a * c)
    (a + b) * c --> (a * c) + (b * c)
    (a / b) * c --> (a * c) / b
    (a / b) * b --> a
    (a * b) / b --> a



    #-a --> -1 * a
    #a - b --> a + -b
    1 * a --> a
    a + a --> 2*a

    0 * a --> 0
    a + 0 --> a
    a - a --> 0

    a <= a --> 1
    a + b <= c + b --> a <= c
    a + b <= b + c --> a <= c
    a * b <= c * b --> a <= c
    a / b <= c / b --> a <= c
    a - b <= c - b --> a <= c
    a::Number <= b::Number => a<=b ? 1 : 0
    a <= b - c --> a + c <= b
    #a <= b - c --> a - b <= -c
    a <= b + c --> a - c <= b
    a <= b + c --> a - b <= c

    a + c <= b --> a <= b - c
    a - c <= b --> a <= b + c

    min(a, min(a, b)) --> min(a, b)
    min(min(a, b), b) --> min(a, b)
    max(a, max(a, b)) --> max(a, b)
    max(max(a, b), b) --> max(a, b)

    max(a, min(a, b)) --> a
    min(a, max(a, b)) --> a
    max(min(a, b), b) --> b
    min(max(a, b), b) --> b

    min(a + b::Number, a + c::Number) => b < c ? :($a + $b) : :($a + $c)
    max(a + b::Number, a + c::Number) => b < c ? :($a + $c) : :($a + $b)
    min(a - b::Number, a - c::Number) => b < c ? :($a - $c) : :($a - $b)
    max(a - b::Number, a - c::Number) => b < c ? :($a - $b) : :($a - $c)
    min(a * b::Number, a * c::Number) => b < c ? :($a * $b) : :($a * $c)
    max(a + b::Number, a + c::Number) => b < c ? :($a + $c) : :($a + $b)
    max(a, b::Number) <= c::Number => b > c ? 0 : :($max($a, $b) <= $c)
    #min(a, b::Number) <= c::Number => b < c ? 0 : 1
    #min(a + b, a + c) --> min(b, c)
end



# exp_size(node::Node) = exp_size(node.ex)
exp_size(ex::Expr) = sum(exp_size.(ex.args))
exp_size(ex::Symbol) = 1f0
exp_size(ex::Int) = 1f0
exp_size(ex::Float64) = 1f0
exp_size(ex::Float32) = 1f0

# execute(ex, theory) = map(th->Postwalk(Metatheory.Chain([th]))(ex), theory)
function my_rewriter!(position, ex, rule)
    if isempty(position)
        # @show ex
        return rule(ex) 
    end
    ind = position[1]
    # @show ex
    ret = my_rewriter!(position[2:end], ex.args[ind], rule)
    if !isnothing(ret)
        # @show ret
        # println("Suc")
        ex.args[ind] = ret
    end
    return nothing
end


function traverse_expr!(ex, matcher, tree_ind, trav_indexs, tmp)
    if typeof(ex) != Expr
        # trav_indexs = []
        return
    end
    a = matcher(ex)
    # @show trav_indexs
    if !isnothing(a)
        b = copy(trav_indexs)
        push!(tmp, b)
    end

    # Traverse sub-expressions
    for (ind, i) in enumerate(ex.args)
        # Update the traversal index path with the current index
        push!(trav_indexs, ind)

        # Recursively traverse the sub-expression
        traverse_expr!(i, matcher, tree_ind, trav_indexs, tmp)

        # After recursion, pop the last index to backtrack to the correct level
        pop!(trav_indexs)
    end
end

function execute(ex, theory)
    res = []
    old_ex = copy(ex)
    for (ind, r) in enumerate(theory)
        tmp = []
        traverse_expr!(ex, r, 1, [], tmp) 
        if isempty(tmp)
            push!(res, ((ind, 0), ex))
        else
            for (ri, i) in enumerate(tmp)
                old_ex = copy(ex)
                if isempty(i)
                    push!(res, ((ind, ri), r(old_ex)))
                else 
                    my_rewriter!(i, old_ex, r)
                    push!(res, ((ind, ri), old_ex))
                end
            end
        end
    end
    return res
end

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


function expand_state!(ex::Expr, theory::Vector, depth_dict::Dict,  policy, symbols_to_index)
    # applicable_rules = filter(r -> r[2] != ex, collect(enumerate(execute(ex, theory))))
    applicable_rules = filter(r -> r[2] != ex, execute(ex, theory))
    # println(typeof(applicable_rules))
    # rule_ids, new_expr = collect(applicable_rules)
    probs = []
    for (rule_id, new_expr) in applicable_rules
        ee = ex2mill(new_expr, symbols_to_index)   
        push!(probs, only(policy(ee)))
    end
    min_prob = argmin(probs)
    return applicable_rules[min_prob][2]
end


function compute_reward(old_ex, new_ex)
    old_size = exp_size(old_ex)
    new_size = exp_size(new_ex)
    return old_size - new_size 
end


function policy_loss(correct_rule_prob, rest_rule_prob)
    return sum(log.(1 .+ exp.(correct_rule_prob .- rest_rule_prob))) 
end


function build_tree(ex, policy, theory, max_itr, symbols_to_index)
    counter = 0
    proof = []
    while max_itr != counter
        applicable_rules = filter(r -> r[2] != ex, execute(ex, theory))
        if length(applicable_rules) == 0
            break
        end
        probs = []
        for (rule_id, new_expr) in applicable_rules
            ee = ex2mill(new_expr, symbols_to_index)   
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


function my_rewriter!(position, ex, rule)
    if isempty(position)
        # @show ex
        return rule(ex) 
    end
    ind = position[1]
    # @show ex
    ret = my_rewriter!(position[2:end], ex.args[ind], rule)
    if !isnothing(ret)
        # @show ret
        # println("Suc")
        ex.args[ind] = ret
    end
    return nothing
end


function traverse_expr!(ex, matcher, tree_ind, trav_indexs, tmp)
    if typeof(ex) != Expr
        # trav_indexs = []
        return
    end
    a = matcher(ex)
    # @show trav_indexs
    if !isnothing(a)
        b = copy(trav_indexs)
        push!(tmp, b)
    end

    # Traverse sub-expressions
    for (ind, i) in enumerate(ex.args)
        # Update the traversal index path with the current index
        push!(trav_indexs, ind)

        # Recursively traverse the sub-expression
        traverse_expr!(i, matcher, tree_ind, trav_indexs, tmp)

        # After recursion, pop the last index to backtrack to the correct level
        pop!(trav_indexs)
    end
end


function test_training_samples(training_samples, train_data, theory)
    counter_failed = 0
    for (sample, ex) in zip(training_samples, train_data)
        counter_failed_rule = 0
        # @show sample.initial_expr
        # @show sample.proof
        for (p,k) in sample.proof
            tmp = []
            try

                traverse_expr!(ex, theory[p], 1, [], tmp)
                # @show tmp
                if isempty(tmp[k])
                    ex = theory[p](ex) 
                else
                    my_rewriter!(tmp[k], ex, theory[p])
                end
            catch
                counter_failed_rule += 1
            end
        end
        if counter_failed_rule != 0
            counter_failed += 1
        end
    end
    @show counter_failed
    return counter_failed
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

function test_policy(policy, data, theory, max_itr, symbols_to_index)
    result = []
    result_proof = []
    simp_expressions = []
    for (index, i) in enumerate(data)
        @show i
        simplified_expression, proof = build_tree(i, policy, theory, max_itr, symbols_to_index) 
        @show simplified_expression
        original_length = exp_size(i)
        simplified_length = exp_size(simplified_expression)
        push!(result, original_length - simplified_length)
        push!(result_proof, proof)
        push!(simp_expressions, simplified_expression)
    end
    return result, result_proof, simp_expressions
end

hidden_size = 128 
heuristic = ExprModel(
    Flux.Chain(Dense(length(symbols_to_index) + 2, hidden_size,relu), Dense(hidden_size,hidden_size)),
    Mill.SegmentedSumMax(hidden_size),
    Flux.Chain(Dense(3*hidden_size, hidden_size,relu), Dense(hidden_size, hidden_size)),
    Flux.Chain(Dense(hidden_size, hidden_size,relu), Dense(hidden_size, 1))
    )

pc = Flux.params([heuristic.head_model, heuristic.aggregation, heuristic.joint_model, heuristic.heuristic])
# train_policy(train_data[1], heuristic, symbols_to_index, pc)
#
mutable struct TrainingSample{D, S, E, P, HP, HN, IE}
    training_data::D
    saturated::S
    expression::E
    proof::P
    hp::HP
    hn::HN
    initial_expr::IE
end

training_samples = Vector{TrainingSample}()
@load "training_samplesk1000_v3.jld2" training_samples

@assert test_training_samples(training_samples, train_data, theory) == 0

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

df = DataFrame([[], [], [], [], []], ["Epoch", "Id", "Simplified Expr", "Proof", "Length Reduced"])
optimizer = Flux.ADAM()
epoch = 100
# yk = ones(length(theory))
plot_loss = []
plot_reduction = []
for ep in 1:epoch
    @show ep
    for i in training_samples[1:10]
        td = copy(i.initial_expr)
        # hn = copy(i.hn)
        # for k in size(hn)[2]:-1:1
        #     hn[:, begin:k-1] .= hn[:, begin:k-1] .- hn[:, k]
        # end
        # gd = gradient(pc) do
        #     loss = policy_loss_func(heuristic, i.training_data, j, i.hp, hn)
        #     @show loss
        #     return loss
        # end
        # Flux.update!(optimizer, pc, gd)
        # @show td
        # @show i.proof
        for (j,jr) in i.proof
            applicable_rules = filter(r -> r[2] != td, execute(td, theory))
            # applicable_rules = execute(td, theory)
            ee = []
            final_index = 0
            # @show length(applicable_rules)
            for (ind, (r, new_expr)) in enumerate(applicable_rules)
                if r[1] == j && r[2] == jr
                    final_index = ind
                end
                push!(ee, ex2mill(new_expr, symbols_to_index))
            end
            ds = reduce(catobs, ee) 
            # @show final_index

            gd = gradient(pc) do
                loss = policy_loss_func1(heuristic, ds, final_index)
                # @show loss
                return loss
            end

            loss = policy_loss_func1(heuristic, ds, final_index)
            push!(plot_loss, loss)
            td = applicable_rules[final_index][2]
            # tmp = []
            # traverse_expr!(td, theory[j], 1, [], tmp)
            # @show tmp
            # if isempty(tmp[jr])
            #     td = theory[j](td) 
            # # if !isempty(tmp)
            # else
            #     # if isempty(tmp[jr])
            #     #     td = theory[j](td)
            #     # else
            #     my_rewriter!(tmp[jr], td, theory[j])
            #     # end
            # end
            # @show td
            Flux.update!(optimizer, pc, gd)
        end
        # @show td
    end
    # length_reduction, proof, simp_expressions = test_policy(heuristic, test_data[1:10], theory, 1000, symbols_to_index)
    # # push!(plot_reduction, avarage_length_reduction)
    # new_df_rows = [(ep, ind, s[1], s[2], s[3]) for (ind,s) in enumerate(zip(simp_expressions,  proof, length_reduction))]
    # for row in new_df_rows
    #     push!(df, row)
    # end
end

length_reduction, proof, simp_expressions = test_policy(heuristic, train_data[1:10], theory, 60, symbols_to_index)
# push!(plot_reduction, avarage_length_reduction)
new_df_rows = [(1, ind, s[1], s[2], s[3]) for (ind,s) in enumerate(zip(simp_expressions,  proof, length_reduction))]
for row in new_df_rows
    push!(df, row)
end
BSON.@save "policy_search_heuristic.bson" heuristic
# CSV.write("policy_data1000.csv", df)
# loss = policy_loss_func1(heuristic, ds, final_index)
# plot(1:length(plot_loss), plot_loss, xlabel="Epoch", ylabel="Loss", title="Training Loss", legend=false)
# savefig("policy_training_loss.png")
# plot(1:length(plot_reduction), plot_reduction, xlabel="Epoch", ylabel="Loss", title="Avarage reduction length", legend=false)
# savefig("policy_reduction_training.png")
# avarage_length_reduction = test_policy(heuristic, train_data[1:10], theory, 100, symbols_to_index)
# avarage_length_reduction = test_heuristic(heuristic, test_data[1:100], max_steps, max_depth)
