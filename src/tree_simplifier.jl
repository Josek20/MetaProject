mutable struct Node{E, P}
    ex::E
    rule_index::Tuple
    children::Vector{UInt64}
    parent::P
    depth::Int64
    node_id::UInt64
    #expression_encoding::Union{MyNodes, Nothing}
    # expression_encoding::Union{ExprEncoding, ProductNode}
    expression_encoding::ProductNode
end


Node(ex::Int, rule_index, parent, depth, ee) = Node(ex, rule_index, UInt64[],  parent, depth, hash(ex), ee)


function Node(ex::Expr, rule_index::Tuple, parent::UInt64, depth::Int64, ee::ProductNode)
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
        return rule(ex) 
    end
    ind = position[1]
    ret = my_rewriter!(position[2:end], ex.args[ind], rule)
    if !isnothing(ret)
        ex.args[ind] = ret
    end
    return nothing
end


function traverse_expr!(ex, matchers, tree_ind, trav_indexs, tmp)
    if typeof(ex) != Expr
        # trav_indexs = []
        return
    end
    a = filter(em->!isnothing(em[2]), collect(enumerate(rt(ex) for rt in matchers)))
    if !isempty(a)
        b = copy(trav_indexs)
        append!(tmp, [(b, i[1]) for i in a])
    end

    # Traverse sub-expressions
    for (ind, i) in enumerate(ex.args)
        # Update the traversal index path with the current index
        push!(trav_indexs, ind)

        # Recursively traverse the sub-expression
        traverse_expr!(i, matchers, tree_ind, trav_indexs, tmp)

        # After recursion, pop the last index to backtrack to the correct level
        pop!(trav_indexs)
    end
end


function old_traverse_expr!(ex, matchers, tree_ind, trav_indexs, tmp)
    if typeof(ex) != Expr
        # trav_indexs = []
        return
    end

    a = matchers(ex)
    if !isnothing(a)
        b = copy(trav_indexs)
        push!(tmp, b)
    end

    # Traverse sub-expressions
    for (ind, i) in enumerate(ex.args)
        # Update the traversal index path with the current index
        push!(trav_indexs, ind)

        # Recursively traverse the sub-expression
        old_traverse_expr!(i, matchers, tree_ind, trav_indexs, tmp)

        # After recursion, pop the last index to backtrack to the correct level
        pop!(trav_indexs)
    end
end


function old_execute(ex, theory)
    res = []
    old_ex = copy(ex)
    for (ind, r) in enumerate(theory)
        tmp = []
        old_traverse_expr!(ex, r, 1, Int32[], tmp) 
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


function execute(ex, theory)
    res = []
    tmp = []
    traverse_expr!(ex, theory, 1, Int32[], tmp) 
    for (pl, r) in tmp
        old_ex = copy(ex)
        if isempty(pl)
            push!(res, ((pl, r), theory[r](old_ex)))
        else
            my_rewriter!(pl, old_ex, theory[r])
            push!(res, ((pl, r), old_ex))
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
            push!(soltree[new_node.parent].children, new_node.node_id)
            # @show keys(soltree)
            # @show soltree[old_node.parent].children
            # @assert haskey(soltree, old_node.parent)
            # tmp1 = soltree[old_node.parent].children
            # deleteat!(tmp1, UInt64(old_node.node_id))
            filter!(x->x!=old_node.node_id, soltree[old_node.parent].children) 
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


function expand_node!(parent::Node, soltree::Dict{UInt64, Node}, heuristic::ExprModel, open_list::PriorityQueue, encodings_buffer::Dict{UInt64, ProductNode}, all_symbols::Vector{Symbol}, symbols_to_index::Dict{Symbol, Int64}, theory::Vector, variable_names::Dict) 
    ex = parent.ex
    #println("Expanding nodes from expression $ex")
    # succesors = filter(r -> r[2] != ex, old_execute(ex, theory))
    # 14.989794 seconds (22.69 M allocations: 3.844 GiB, 3.48% gc time)
    # 7.636094 seconds (19.08 M allocations: 2.482 GiB, 4.21% gc time)

    succesors = execute(ex, theory)
    # 14.612299 seconds (22.69 M allocations: 3.844 GiB, 3.27% gc time)
    # 7.739473 seconds (18.29 M allocations: 2.389 GiB, 4.01% gc time)

    for (rule_index, new_exp) in succesors
        expr_hash = hash(new_exp)
        if !haskey(encodings_buffer, expr_hash)
            #encodings_buffer[hash(new_exp)] = expression_encoder(new_exp, all_symbols, symbols_to_index)
            # @show new_exp
            #
            # @show rule_index 
            encodings_buffer[expr_hash] = ex2mill(new_exp, symbols_to_index, all_symbols, variable_names) 
        end
        new_node = Node(new_exp, rule_index, parent.node_id, parent.depth + 1, encodings_buffer[expr_hash])
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


function build_tree!(soltree::Dict{UInt64, Node}, heuristic::ExprModel, open_list::PriorityQueue, close_list::Set{UInt64}, encodings_buffer::Dict{UInt64, ProductNode}, all_symbols::Vector{Symbol}, symbols_to_index::Dict{Symbol, Int64}, max_steps, max_depth, expansion_history, theory, variable_names)
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

        reached_goal = expand_node!(node, soltree, heuristic, open_list, encodings_buffer, all_symbols, symbols_to_index, theory, variable_names)

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


function extract_smallest_terminal_node1(soltree::Dict{UInt64, Node}, close_list::Set{UInt64})
    all_nodes = [i for i in values(soltree)] 
    exs = [i.ex for i in values(soltree)] 
    smallest_expression_node = argmin(exp_size.(exs))
    return all_nodes[smallest_expression_node]
end

function extract_smallest_terminal_node(soltree::Dict{UInt64, Node}, close_list::Set{UInt64})
    min_node = nothing
    for (k, n) in soltree
        if isnothing(min_node)
            min_node = n
        elseif exp_size(n.ex) < exp_size(min_node.ex)
            min_node = n
        elseif exp_size(n.ex) == exp_size(min_node.ex)
            if n.depth < min_node.depth && n.depth != 0
                min_node = n
            end
        end
    end
    # all_nodes = [i for i in values(soltree)] 
    # exs = [i.ex for i in values(soltree)] 
    # smallest_expression_node = argmin(exp_size.(exs))
    return min_node
end


function heuristic_forward_pass(heuristic, ex::Expr, max_steps, max_depth, all_symbols, theory, variable_names)
    soltree = Dict{UInt64, Node}()
    open_list = PriorityQueue{Node, Float32}()
    close_list = Set{UInt64}()
    expansion_history = Dict{UInt64, Vector}()
    #encodings_buffer = Dict{UInt64, ExprEncoding}()
    encodings_buffer = Dict{UInt64, ProductNode}()
    println("Initial expression: $ex")
    #encoded_ex = expression_encoder(ex, all_symbols, symbols_to_index)
    encoded_ex = ex2mill(ex, symbols_to_index, all_symbols, variable_names)
    root = Node(ex, (0,0), hash(ex), 0, encoded_ex)
    soltree[root.node_id] = root
    #push!(open_list, root.node_id)
    enqueue!(open_list, root, only(heuristic(root.expression_encoding)))

    reached_goal = build_tree!(soltree, heuristic, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, max_steps, max_depth, expansion_history, theory, variable_names)
    println("Have successfuly finished bulding simplification tree!")

    smallest_node = extract_smallest_terminal_node(soltree, close_list)
    # for (ind, (i, cof)) in enumerate(open_list)
    #     expansion_history[i.node_id] = [length(expansion_history) + ind - 1, cof]
    # end
    # @show expansion_history[smallest_node.node_id]
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


function train_heuristic!(heuristic, data, training_samples, max_steps, max_depth, all_symbols, theory, variable_names)  
    # @threads for i in data
    for (index, i) in enumerate(data)
        println("Index: $index")
        # if length(training_samples) > index && training_samples[index].saturated
        #     continue
        # end
        #try
        simplified_expression, depth_dict, big_vector, saturated, hp, hn, root, proof_vector = heuristic_forward_pass(heuristic, i, max_steps, max_depth, all_symbols, theory, variable_names)
        println("Saturated: $saturated")
        new_sample = TrainingSample(big_vector, saturated, simplified_expression, proof_vector, hp, hn, i)
        if length(training_samples) >= index 
            training_samples[index] = isbetter(training_samples[index], new_sample) ? new_sample : training_samples[index]
        end
        if length(training_samples) < index
        # lock() do
            push!(training_samples, new_sample)
        end
        # end
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
    # @save "data/training_data/training_samplesk$(data_len)_v5.jld2" training_samples
    #return training_samples
end


function apply_proof_to_expr(ex, proof_vector, theory)
    println("Root: $ex")
    for rule_index in proof_vector
        rule_applied = theory[rule_index]
        ex = collect(execute(ex, [rule_applied]))[1]
        println("$ex")
    end
end


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
