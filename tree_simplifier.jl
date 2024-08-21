include("src/small_data_loader.jl")
include("tree_lstm.jl")
using JSON
using BSON
using JLD2
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
theory = @theory a b c d x y begin
    # add.rc
    a + b --> b + a
    # + (~a, ~b) --> + (~b,~a)
    a + (b + c) --> (a + b) + c
    a + 0 --> a
    a * (b + c) --> a*b + a*c
    a*b + a*c --> a * (b + c)
    (a / b) + c --> (a + (c * b)) / b
    (a + (c * b)) / b --> (a / b) + c
    x / 2 + x % 2 --> (x + 1) / 2
    x * a + y * b --> ((a / b) * x + y) * b
    
    # and.rc
    a && b --> b && a
    a && (b && c) --> (a && b) && c
    1 && a --> a
    a && a --> a
    a && !a --> 0
    (a == c::Number) && (b == x::Number) => c != x ? 0 : :($a)
    !a::Number && b::Number => a != b ? 0 : :($a)
    (a < y) && (a < b) --> a < min(y, b)
    a < min(y, b) --> (a < y) && (a < b)
    (a <= y) && (a <= b) --> a <= min(y, b)
    a <= min(y, b) --> (a <= y) && (a <= b)
    (a > y) && (a > b) --> x > max(y, b)
    a > max(y, b) --> (a > y) && (a > b)
    
    (a >= y) && (a >= b) --> a >= max(y, b)
    a >= max(y, b) --> (a >= y) && (a >= b)
    
    (a::Number > b) && (c::Number < b) => a < c ? 0 : :($a > $b) && :($c < $b)
    (a::Number >= b) && (c::Number <= b) => a < c ? 0 : :($a >= $b) && :($c <= $b)
    (a::Number >= b) && (c::Number < b) => a <= c ? 0 : :($a >= $b) && :($c < $b)
    
    a && (b || c) --> (a && b) || (a && c)
    a || (b && c) --> (a || b) && (a || c)
    b || (b && c) --> b

    # div.rs
    0 / a --> 0
    a / a --> 1
    (-1 * a) / b --> a / (-1 * b)
    a / (-1 * b) --> (-1 * a) / b
    -1 * (a / b) --> (-1 * a) / b
    (-1 * a) / b --> -1 * (a / b)

    (a * b) / c --> a / (c / b)
    (a * b) / c --> a * (b / c)
    (a + b) / c --> (a / c) + (b / c)
    ((a * b) + c) / d --> ((a * b) / d) + (c / d)

    # eq.rs
    x == y --> y == x
    x == y --> (x - y) == 0
    x + y == a --> x == a - y
    x == x --> 1
    x*y == 0 --> (x == 0) || (y == 0)
    max(x,y) == y --> x <= y
    min(x,y) == y --> y <= x
    y <= x --> min(x,y) == y

    # ineq.rs
    x != y --> ! (x == y)

    # lt.rs
    x > y --> y < x
    x < y --> (-1 * y) < (-1 * x)
    a < a --> 0
    a + b < c --> a < c - b

    a - b < a --> 0 < b
    0 < a::Number => 0 < a ? 1 : 0
    a::Number < 0 => a < 0 ? 1 : 0
    min(a , b) < a --> b < a
    min(a, b) < min(a , c) --> b < min(a, c)
    max(a, b) < max(a , c) --> max(a ,b) < c

    # max.rs
    max(a, b) --> (-1 * min(-1 * a, -1 * b))
    
    # min.rs
    min(a, b) --> min(b, a)
    min(min(a, b), c) --> min(a, min(b, c))
    min(a,a) --> a
    min(max(a, b), a) --> a
    min(max(a, b), max(a, c)) --> max(min(b, c), a)
    min(max(min(a,b), c), b) --> min(max(a,c), b)
    min(a + b, c) --> min(b, c - a) + a
    min(a, b) + c --> min(a + c, b + c)
    min(a, a + b::Number) => b > 0 ? :($a) : :($a + $b)

end
mutable struct Node{E, P}
    ex::E
    rule_index::Tuple
    children::Vector{UInt64}
    parent::P
    depth::Int64
    node_id::UInt64
    #expression_encoding::Union{MyNodes, Nothing}
    expression_encoding::Union{ExprEncoding, ProductNode}
end

Node(ex::Int, rule_index, parent, depth, ee) = Node(ex, rule_index, UInt64[],  parent, depth, hash(ex), ee)

function Node(ex::Expr, rule_index::Tuple, parent::UInt64, depth::Int64, ee::Union{ExprEncoding, ProductNode})
    Node(ex, rule_index, UInt64[],  parent, depth, hash(ex), ee)
end

exp_size(node::Node) = exp_size(node.ex)
exp_size(ex::Expr) = sum(exp_size.(ex.args))
exp_size(ex::Symbol) = 1f0
exp_size(ex::Int) = 1f0
exp_size(ex::Float64) = 1f0
exp_size(ex::Float32) = 1f0

# get all availabel rules to apply
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


function push_to_tree!(soltree::Dict, new_node::Node)
    node_id = new_node.node_id
    if haskey(soltree, node_id)
        old_node = soltree[node_id]
        if new_node.depth < old_node.depth
            soltree[node_id] = new_node
            soltree[node_id].children = old_node.children
        else
            soltree[node_id] = old_node
        end
        #soltree[node_id] = new_node.depth < old_node.depth ? new_node : old_node
        return (false)
    else
        soltree[node_id] = new_node
        return (true)
    end
end


function expand_node!(parent::Node, soltree::Dict{UInt64, Node}, heuristic::ExprModel, open_list::PriorityQueue, encodings_buffer::Dict{UInt64, ProductNode}, all_symbols::Vector{Symbol}, symbols_to_index::Dict{Symbol, Int64}) 
    ex = parent.ex
    #println("Expanding nodes from expression $ex")
    succesors = filter(r -> r[2] != ex, execute(ex, theory))

    for (rule_index, new_exp) in succesors
        if !haskey(encodings_buffer, hash(new_exp))
            #encodings_buffer[hash(new_exp)] = expression_encoder(new_exp, all_symbols, symbols_to_index)
            encodings_buffer[hash(new_exp)] = ex2mill(new_exp, symbols_to_index)
        end
        new_node = Node(new_exp, rule_index, parent.node_id, parent.depth + 1, encodings_buffer[hash(new_exp)])
        if push_to_tree!(soltree, new_node)
            push!(parent.children, new_node.node_id)
            #println(new_exp)
            enqueue!(open_list, new_node, only(heuristic(new_node.expression_encoding)))
        end
        if new_exp == 1
            return true
        end
    end
    return false
end


function build_tree!(soltree::Dict{UInt64, Node}, heuristic::ExprModel, open_list::PriorityQueue, close_list::Set{UInt64}, encodings_buffer::Dict{UInt64, ProductNode}, all_symbols::Vector{Symbol}, symbols_to_index::Dict{Symbol, Int64}, max_steps, max_depth, expansion_history)
    step = 0
    reached_goal = false
    while length(open_list) > 0
        if max_steps == step
            break
        end
        node, prob = dequeue_pair!(open_list)
        expansion_history[node.node_id] = [step, prob]
        step += 1
        # println(node.rule_index)
        if node.depth >= max_depth
            continue
        end
        # if node.node_id in close_list
        #     println("Already been expanded $(node.ex)")
        #     continue
        # end

        reached_goal = expand_node!(node, soltree, heuristic, open_list, encodings_buffer, all_symbols, symbols_to_index)

        if reached_goal
            return true 
        end
        push!(close_list, node.node_id)
    end
    return false
end


function extract_loss_matrices(node::Node, soltree::Dict{UInt64, Node})
    proof_nodes_ids = []
    push!(proof_nodes_ids, node.node_id)
    while node.parent != node.node_id
        parent_id = node.parent
        node = soltree[parent_id]
        push!(proof_nodes_ids, node.node_id)
    end
    return proof_nodes_ids
end

function extract_rules_applied(node::Node, soltree::Dict{UInt64, Node}) 
    proof_vector = Vector()
    depth_dict = Dict{Int, Vector{Any}}()
    big_vector = Vector()
    tmp_hp = Vector{Vector{Int16}}()
    tmp_hn = Vector{Vector{Int16}}()
    node_proof_vector = Vector() 
    #while !isnothing(node.parent)
    #nodes_to_use = filter(n->n.depth<node.depth, values(soltree)
    while node.parent != node.node_id
        push!(proof_vector, node.rule_index)
        # push!(node_proof_vector, node.ex)
        push!(node_proof_vector, node.node_id)
        depth_dict[node.depth] = [node.expression_encoding]
        push!(big_vector, node.expression_encoding)
        push!(tmp_hp, Int16[])
        push!(tmp_hn, Int16[])
        padding = length(tmp_hn) > 1 ? zeros(Int16, length(tmp_hn[end - 1])) : Int16[]
        append!(tmp_hp[end], padding)
        append!(tmp_hn[end], padding)
        push!(tmp_hn[end], 0)
        push!(tmp_hp[end], 1)
        for children_id in soltree[node.parent].children
            if children_id == node.node_id
                continue
            end
            push!(depth_dict[node.depth], soltree[children_id].expression_encoding)
            push!(big_vector, soltree[children_id].expression_encoding)
            # push!(node_proof_vector, soltree[children_id].ex)
            push!(node_proof_vector, node.node_id)
            push!(tmp_hp[end], 0)
            push!(tmp_hn[end], 1)
        end
        node = soltree[node.parent]
    end
    padding_value = 0
    hn = nothing
    if !isempty(tmp_hn)
        max_length = maximum(length, tmp_hn)

        padded_vectors = [vcat(vec, fill(padding_value, max_length - length(vec))) for vec in tmp_hn]
        hn = hcat(padded_vectors...)
        # for i in 2:size(hn, 2)
        #     hn[:, i] .= hn[:, i] + hn[:, i - 1]
        # end

        for i in size(hn, 2) - 1:-1:1
            hn[:, i] .= hn[:, i] + hn[:, i + 1]
        end
    end

    hp = nothing
    if !isempty(tmp_hp)
        max_length = maximum(length, tmp_hp)

        padded_vectors = [vcat(vec, fill(padding_value, max_length - length(vec))) for vec in tmp_hp]
        hp = hcat(padded_vectors...)
    end

    ds = nothing
    if !isempty(big_vector)
        ds = reduce(catobs, big_vector) 
    end
    #hp = findall(x-> x == 1, hp)

    return reverse(proof_vector), depth_dict, ds, hp, hn, node_proof_vector
end


function extract_smallest_terminal_node(soltree::Dict{UInt64, Node}, close_list::Set{UInt64})
    all_nodes = [i for i in values(soltree)] 
    exs = [i.ex for i in values(soltree)] 
    smallest_expression_node = argmin(exp_size.(exs))
    return all_nodes[smallest_expression_node]
end


function heuristic_forward_pass(heuristic, ex::Expr, max_steps, max_depth)
    soltree = Dict{UInt64, Node}()
    open_list = PriorityQueue{Node, Float32}()
    close_list = Set{UInt64}()
    expansion_history = Dict{UInt64, Vector}()
    #encodings_buffer = Dict{UInt64, ExprEncoding}()
    encodings_buffer = Dict{UInt64, ProductNode}()
    println("Initial expression: $ex")
    #encoded_ex = expression_encoder(ex, all_symbols, symbols_to_index)
    encoded_ex = ex2mill(ex, symbols_to_index)
    root = Node(ex, (0,0), hash(ex), 0, encoded_ex)
    soltree[root.node_id] = root
    #push!(open_list, root.node_id)
    enqueue!(open_list, root, only(heuristic(root.expression_encoding)))

    reached_goal = build_tree!(soltree, heuristic, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, max_steps, max_depth, expansion_history)
    println("Have successfuly finished bulding simplification tree!")

    smallest_node = extract_smallest_terminal_node(soltree, close_list)
    for (ind, (i, cof)) in enumerate(open_list)
        expansion_history[i.node_id] = [length(expansion_history) + ind - 1, cof]
    end
    @show expansion_history[smallest_node.node_id]
    simplified_expression = smallest_node.ex
    println("Simplified expression: $simplified_expression")

    proof_vector, depth_dict, big_vector, hp, hn, node_proof_vector = extract_rules_applied(smallest_node, soltree)
    println("Proof vector: $proof_vector")

    return simplified_expression, depth_dict, big_vector, length(open_list) == 0 || reached_goal, hp, hn, root, proof_vector
end

mutable struct TrainingSample{D, S, E, P, HP, HN, IE}
    training_data::D
    saturated::S
    expression::E
    proof::P
    hp::HP
    hn::HN
    initial_expr::IE
end

function isbetter(a::TrainingSample, b::TrainingSample)
    if exp_size(a.expression) > exp_size(b.expression)
        return true
      elseif exp_size(a.expression) == exp_size(b.expression) && length(a.proof) > length(b.proof)
        return true
    else
        return false
    end
end

function train_heuristic!(heuristic, data, training_samples, max_steps, max_depth)  
    for (index, i) in enumerate(data)
        println("Index: $index")
        if length(training_samples) > index && training_samples[index].saturated
            continue
        end
        #try
        simplified_expression, depth_dict, big_vector, saturated, hp, hn, root, proof_vector = heuristic_forward_pass(heuristic, i, max_steps, max_depth)
        println("Saturated: $saturated")
        new_sample = TrainingSample(big_vector, saturated, simplified_expression, proof_vector, hp, hn, i)
        if length(training_samples) > index 
            training_samples[index] = isbetter(training_samples[index], new_sample) ? new_sample : training_samples[index]
        end
        if length(training_samples) < index
            push!(training_samples, new_sample)
        end
        if isempty(depth_dict)
            println("Error")
            continue
        end
        # catch e
        #     println("Error with $(i)")
        #     continue
        # end
    end
    data_len = length(data)
    @save "training_samplesk$(data_len)_v3.jld2" training_samples
    #return training_samples
end

function test_heuristic(heuristic, data, max_steps, max_depth)
    result = []
    result_proof = []
    simp_expressions = []
    for (index, i) in enumerate(data)
        # simplified_expression, _, _, _, _, _, proof_vector = heuristic_forward_pass(heuristic, i, max_steps, max_depth)
        simplified_expression, depth_dict, big_vector, saturated, hp, hn, root, proof_vector = heuristic_forward_pass(heuristic, i, max_steps, max_depth)
        original_length = exp_size(i)
        simplified_length = exp_size(simplified_expression)
        push!(result, original_length - simplified_length)
        push!(result_proof, proof_vector)
        push!(simp_expressions, simplified_expression)
    end
    return result, result_proof, simp_expressions
end

function apply_proof_to_expr(ex, proof_vector, theory)
    println("Root: $ex")
    for rule_index in proof_vector
        rule_applied = theory[rule_index]
        ex = collect(execute(ex, [rule_applied]))[1]
        println("$ex")
    end
end

function heuristic_sanity_check(heuristic, training_samples, training_data)
    count_matched = 0
    count_all = 0
    for (ex, sample) in zip(training_data, training_samples)
        proof = sample.proof
        println(sample.expression)
        # if isempty(proof) || length(proof) <= 3
        #     continue
        # end
        # res = heuristic_forward_pass(heuristic, ex, length(proof) + 1, length(proof) + 1)
        res = heuristic_forward_pass(heuristic, ex, 1000, 10)
        learned_proof = res[end]
        println("Training proof $proof: learned proof $learned_proof")
        count_all += 1
        if proof == learned_proof
            count_matched += 1
        end
    end
    println("====>Test results all checked samples: $count_all")
    println("====>Test results all matched samples: $count_matched")
    not_matched = count_all - count_matched
    println("====>Test results all not matched samples: $not_matched")
end


function single_sample_check!(heuristic, training_sample, training_data, pc, optimizer)
    n = 10
    for _ in 1:n
        grad = gradient(pc) do
            # o = heuristic(training_sample.training_data)
            a = heuristic_loss(heuristic, training_sample.training_data, training_sample.hp, training_sample.hn)
            # a = loss1(o, training_sample.hp, training_sample.hn)
            # a = loss(heuristic, training_sample.training_data, training_sample.hp, training_sample.hn)
            @show a
            return a
        end
        Flux.update!(optimizer, pc, grad)
    end
    heuristic_sanity_check(heuristic, [training_sample], [training_data])
end


function test_training_samples(training_samples, train_data, theory)
    counter_failed = 0
    for (sample, ex) in zip(training_samples, train_data)
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

hidden_size = 128 
heuristic = ExprModel(
    Flux.Chain(Dense(length(symbols_to_index) + 2, hidden_size,relu), Dense(hidden_size,hidden_size)),
    Mill.SegmentedSumMax(hidden_size),
    Flux.Chain(Dense(3*hidden_size, hidden_size,relu), Dense(hidden_size, hidden_size)),
    Flux.Chain(Dense(hidden_size, hidden_size,relu), Dense(hidden_size, 1))
    )

epochs = 100
optimizer = ADAM()
training_samples = Vector{TrainingSample}()
pc = Flux.params([heuristic.head_model, heuristic.aggregation, heuristic.joint_model, heuristic.heuristic])
max_steps = 1000
max_depth = 10
n = 1

df = DataFrame([[], [], [], [], []], ["Epoch", "Id", "Simplified Expr", "Proof", "Length Reduced"])
# @load "training_samplesk1000_v3.jld2" training_samples
if isfile("tree_search_heuristic.bson")
    BSON.@load "tree_search_heuristic.bson" heuristic
elseif isfile("tr2aining_samplesk1000_v3.jld2")
    @load "training_samplesk1000_v3.jld2" training_samples
else
    # train_heuristic!(heuristic, train_data, training_samples, max_steps, max_depth)
    @load "training_samplesk1000_v3.jld2" training_samples

    test_training_samples(training_samples, train_data, theory)

    for ep in 1:epochs 
        # train_heuristic!(heuristic, train_data, training_samples, max_steps, max_depth)
        for sample in training_samples
            if isnothing(sample.training_data) 
                continue
            end
            grad = gradient(pc) do
                # o = heuristic(sample.training_data)
                a = heuristic_loss(heuristic, sample.training_data, sample.hp, sample.hn)
                # a = loss(heuristic, sample.training_data, sample.hp, sample.hn)
                @show a
                # if isnan(a)
                #     println(sample.expression)
                # end
                return a
            end
            Flux.update!(optimizer, pc, grad)
        end
        # @show avarage_length_reduction
        length_reduction, proofs, simp_expressions = test_heuristic(heuristic, test_data[1:10], max_steps, max_depth)
        # @show length_reduction
        # @show proofs
        # @show length_reduction
        new_df_rows = [(ep, ind, s[1], s[2], s[3]) for (ind,s) in enumerate(zip(simp_expressions,  proofs, length_reduction))]
        for row in new_df_rows
            push!(df, row)
        end
    end
    BSON.@save "tree_search_heuristic.bson" heuristic
    CSV.write("data100.csv", df)
end

length_reduction, proofs, simp_expressions = test_heuristic(heuristic, test_data, max_steps, max_depth)
# BSON.@save "tree_search_heuristic.bson" heuristic
function plot_reduction_stats(df, epochs, n_eq)
    # a = combine(groupby(df, "Epoch"), "Length Reduced"  => mean => "Average Reduction")

    plot() 
    for eq_id in 1:n_eq
        eq_data = df[df[!,"Id"] .== eq_id, :]
        plot!(eq_data[!, "Epoch"], eq_data[!, "Length Reduced"],label="Equation $eq_id" )
    end
    xlabel!("Epoch")
    ylabel!("Length Reduction")
    title!("Neural Network Simplification Performance")
    # legend(:topright)
    display(plot())
end
# v2 <= v2 && ((v0 + v1) + 120) - 1 <= (v0 + v1) + 119
# println("ALR: $avarage_length_reduction")
# heuristic_forward_pass(heuristic, training_data[n], 1000, 10)
# apply_proof_to_expr(train_data[1], [17, 13, 30, 8, 11, 1, 29], theory)
# heuristic_sanity_check(heuristic, training_samples[n:n], train_data[n:n])
# avarage_length_reduction = test_heuristic(heuristic, test_data[1:100], max_steps, max_depth)

# single_sample_check!(heuristic, training_samples[n], train_data[n], pc, optimizer)
