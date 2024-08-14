include("src/small_data_loader.jl")
include("tree_lstm.jl")
using JSON
using JLD2
using Metatheory
using Metatheory.Rewriters
using DataStructures



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


function expand_state!(ex::Expr, theory::Vector, depth_dict::Dict, embeded_rules::Vector, embeded_ex, policy, depth)
    # applicable_rules = filter(r -> r[2] != ex, collect(enumerate(execute(ex, theory))))
    applicable_rules = filter(r -> r[2] != ex, execute(ex, theory))
    # println(typeof(applicable_rules))
    # rule_ids, new_expr = collect(applicable_rules)
    rule_ids = [x[1] for x in applicable_rules]
    new_expr = [x[2] for x in applicable_rules]
    # println(rule_ids)
    # println(new_expr)
    # rule_prob = policy.(embeded_ex, embeded_rules[rule_ids])
    rule_prob = policy(embeded_ex)
    # println(size(rule_prob))
    depth_dict[depth] = []
    for (ind, rule_id) in enumerate(rule_ids)
        push!(depth_dict[depth], [rule_id, rule_prob[ind]])
    end
    return rule_prob[:, 1]
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
    while max_itr == counter
        embeded_ex = ex2mill(ex, symbols_to_index)
        embeded_rules = []
        prob = policy(embeded_ex)
        chosen_rule_index = argmax(prob)
        tmp = []
        traverse_expr!(ex, theory[chosen_rule_index], 1, [], tmp) 
        if !isempty(tmp)
            if isempty(tmp[1])
                ex = theory[chosen_rule_index](ex)
            else
                my_rewriter!(tmp[1], ex, theory[chosen_rule_index])
            end
        end
        
        # succesors = filter(r -> r[2] != ex, execute(ex, theory))
        counter += 1 
    end 
    return ex
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
    for sample in training_samples
        ex = copy(sample.initial_expr)
        counter_failed_rule = 0
        for (p,k) in sample.proof
            tmp = []
            try
                traverse_expr!(ex, theory[p], 1, [], tmp)
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
end


function test_policy(policy, data, theory, max_itr, symbols_to_index)
    result = []
    for (index, i) in enumerate(data)
        @show i
        simplified_expression = build_tree(i, policy, theory, max_itr, symbols_to_index) 
        @show simplified_expression
        original_length = exp_size(i)
        simplified_length = exp_size(simplified_expression)
        push!(result, original_length - simplified_length)
    end
    return mean(result)
end

hidden_size = 128 
heuristic = ExprModel(
    Flux.Chain(Dense(length(symbols_to_index) + 2, hidden_size,relu), Dense(hidden_size,hidden_size)),
    Mill.SegmentedSumMax(hidden_size),
    Flux.Chain(Dense(3*hidden_size, hidden_size,relu), Dense(hidden_size, hidden_size)),
    Flux.Chain(Dense(hidden_size, hidden_size,relu), Dense(hidden_size, length(theory)))
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

# test_training_samples(training_samples, train_data, theory)

optimizer = Flux.ADAM()
# yk = ones(length(theory))
for _ in 1:1
    for i in training_samples
        td = copy(i.initial_expr)
        @show td
        @show i.proof
        for (j,jr) in i.proof
            ee = ex2mill(td, symbols_to_index)
            # yi = zeros(length(theory))
            # yi[j] = 1
            gd = gradient(pc) do
                p = heuristic(ee)
                diff = p[j] .- p
                filtered_diff = filter(x-> x != 0, diff)
                loss = sum(log.(1 .+ exp.(filtered_diff)))
                @show loss
                return loss
            end
            tmp = []
            traverse_expr!(td, theory[j], 1, [], tmp)
            @show tmp
            if isempty(tmp[jr])
                td = theory[j](td) 
            # if !isempty(tmp)
            else
                # if isempty(tmp[jr])
                #     td = theory[j](td)
                # else
                my_rewriter!(tmp[jr], td, theory[j])
                # end
            end
            @show td
            Flux.update!(optimizer, pc, gd)
        end
        # @show td
    end
end
avarage_length_reduction = test_policy(heuristic, train_data[1:10], theory, 1000, symbols_to_index)
# avarage_length_reduction = test_heuristic(heuristic, test_data[1:100], max_steps, max_depth)
