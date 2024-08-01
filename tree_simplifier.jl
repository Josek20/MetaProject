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

mutable struct Node{E, P}
    ex::E
    rule_index::Int64
    children::Vector{UInt64}
    parent::P
    depth::Int64
    node_id::UInt64
    #expression_encoding::Union{MyNodes, Nothing}
    expression_encoding::Union{ExprEncoding, ProductNode}
end

Node(ex::Int, rule_index, parent, depth, ee) = Node(ex, rule_index, UInt64[],  parent, depth, hash(ex), ee)

function Node(ex::Expr, rule_index::Int64, parent::UInt64, depth::Int64, ee::Union{ExprEncoding, ProductNode})
    Node(ex, rule_index, UInt64[],  parent, depth, hash(ex), ee)
end

exp_size(node::Node) = exp_size(node.ex)
exp_size(ex::Expr) = sum(exp_size.(ex.args))
exp_size(ex::Symbol) = 1f0
exp_size(ex::Int) = 1f0
exp_size(ex::Float64) = 1f0
exp_size(ex::Float32) = 1f0

# get all availabel rules to apply
execute(ex, theory) = map(th->Postwalk(Metatheory.Chain([th]))(ex), theory)


# build a tree


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
    succesors = filter(r -> r[2] != ex, collect(enumerate(execute(ex, theory))))

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
        println(node.rule_index)
        if node.depth >= max_depth
            continue
        end
        # if node.node_id in close_list
        #     println("Already been expanded $(node.ex)")
        #     continue
        # end

        reached_goal = expand_node!(node, soltree, heuristic, open_list, encodings_buffer, all_symbols, symbols_to_index)

        if reached_goal
            break
        end
        push!(close_list, node.node_id)
    end
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
    root = Node(ex, 0, hash(ex), 0, encoded_ex)
    soltree[root.node_id] = root
    #push!(open_list, root.node_id)
    enqueue!(open_list, root, only(heuristic(root.expression_encoding)))

    build_tree!(soltree, heuristic, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, max_steps, max_depth, expansion_history)
    println("Have successfuly finished bulding simplification tree!")

    smallest_node = extract_smallest_terminal_node(soltree, close_list)
    simplified_expression = smallest_node.ex
    println("Simplified expression: $simplified_expression")

    proof_vector, depth_dict, big_vector, hp, hn, node_proof_vector = extract_rules_applied(smallest_node, soltree)
    println("Proof vector: $proof_vector")

    return simplified_expression, depth_dict, big_vector, length(open_list) == 0, hp, hn, root, proof_vector
end

mutable struct TrainingSample{D, S, E, P, HP, HN}
    training_data::D
    saturated::S
    expression::E
    proof::P
    hp::HP
    hn::HN
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
        new_sample = TrainingSample(big_vector, saturated, simplified_expression, proof_vector, hp, hn)
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
    @save "training_samples1k_v2.jld2" training_samples
    #return training_samples
end

function test_heuristic(heuristic, data, max_steps, max_depth)
    result = []
    for (index, i) in enumerate(data)
        simplified_expression, _ = heuristic_forward_pass(heuristic, i, max_steps, max_depth)
        original_length = exp_size(i)
        simplified_length = exp_size(simplified_expression)
        push!(result, original_length - simplified_length)
    end
    return mean(result)
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
        res = heuristic_forward_pass(heuristic, ex, length(proof) + 1, length(proof) + 1)
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
            # a = heuristic_loss(o, training_sample.hp, training_sample.hn)
            # a = loss1(o, training_sample.hp, training_sample.hn)
            a = loss(heuristic, training_sample.training_data, training_sample.hp, training_sample.hn)
            println(a)
            return a
        end
        Flux.update!(optimizer, pc, grad)
    end
    heuristic_sanity_check(heuristic, [training_sample], [training_data])
end


hidden_size = 128 
heuristic = ExprModel(
    Flux.Chain(Dense(length(symbols_to_index) + 2, hidden_size,relu), Dense(hidden_size,hidden_size)),
    Mill.SegmentedSumMax(hidden_size),
    Flux.Chain(Dense(3*hidden_size, hidden_size,relu), Dense(hidden_size, hidden_size)),
    Flux.Chain(Dense(hidden_size, hidden_size,relu), Dense(hidden_size, 1))
    )

epochs = 1
optimizer = ADAM()
training_samples = Vector{TrainingSample}()
pc = Flux.params([heuristic.head_model, heuristic.aggregation, heuristic.joint_model, heuristic.heuristic])
max_steps = 1000
max_depth = 10
n = 1
# Check : 2368
# Iitial expression: (((min(v0, 509) + 6) / 8) * 8 + (v1 * 516 + v2)) + 1 <= (((509 + 13) / 16) * 16 + (v1 * 516 + v2)) + 2
# Simplified expression: ((v2 + (min(v0, 509) + v1 * 516)) + 7) - 2 <= (522 + v2) + v1 * 516
if isfile("training_samples1k_v2.jld2")
    @load "training_samples1k_v2.jld2" training_samples
else
    train_heuristic!(heuristic, train_data, training_samples, max_steps, max_depth)
end
for _ in 1:epochs 
    # train_heuristic!(heuristic, train_data, training_samples, max_steps, max_depth)
    for sample in training_samples[1:10]
        if isnothing(sample.training_data) 
            continue
        end
        grad = gradient(pc) do
            o = heuristic(sample.training_data)
            a = heuristic_loss(o, sample.hp, sample.hn)
            # a = loss(heuristic, sample.training_data, sample.hp, sample.hn)
            println(a)
            # if isnan(a)
            #     println(sample.expression)
            # end
            return a
        end
        Flux.update!(optimizer, pc, grad)
    end
end
# println("ALR: $avarage_length_reduction")
# heuristic_forward_pass(heuristic, training_data[1], 1000, 10)
# apply_proof_to_expr(train_data[1], [17, 13, 30, 8, 11, 1, 29], theory)
heuristic_sanity_check(heuristic, training_samples[1:10], train_data[1:10])
# avarage_length_reduction = test_heuristic(heuristic, test_data[1:100], max_steps, max_depth)

# single_sample_check!(heuristic, training_samples[n], train_data[n], pc, optimizer)
