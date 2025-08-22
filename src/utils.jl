function get_training_data_from_proof(proof::Vector, initial_expression::Expr)
    soltree = Dict{UInt64, Node}()
    # exp_cache = LRU(maxsize=10_000)
    # expr_cache = LRU(maxsize=10_000)
    root = Node(initial_expression, (), hash(ex), 0)
    soltree[root.node_id] = root
    ex = initial_expression
    smallest_node = root
    for (ind, i) in enumerate(proof)
        new_ex, rules_applied = all_expand(parent.ex, theory)
        new_nodes = map(x->Node(x[1], x[2], parent.node_id, parent.depth + 1), zip(new_ex, rules_applied))
        filtered_new_nodes = filter(x-> push_to_tree!(soltree, x), new_nodes)
        nodes_ids = map(x->x.node_id, filtered_new_nodes)
        append!(smallest_node.children, nodes_ids)
        smallest_node = only(filter(x->x.rule_index == i, filtered_new_nodes))
    end
    ds, hp, hn, proof_vector, tr = extract_training_data(smallest_node, soltree, root)
    return (;ds, hp, hn, tr)
end


function track_proof(initial_expression::Expr, proof::Vector, finall_expression::Union{Expr, Int})
    ex = initial_expression
    for (ind,i) in enumerate(proof)
        pos, rule_index = i
        println("Index $(ind), ex $(ex), theory $(theory[rule_index])")
        o = my_rewriter!(pos, ex, theory[rule_index])
        ex = isnothing(o) ? ex : o
    end
    @assert ex == finall_expression
end


function simple_to_flux(m1::ExprModelSimpleChains, m2::ExprModel)
    simple_weights = SimpleChains.weights(m1.expr_model.args_model, m1.model_params.args_model)
    for i in 1:length(m2.args_model.layers)
        m2.args_model.layers[i].weight .= simple_weights[i]
    end
    simple_weights = SimpleChains.weights(m1.expr_model.head_model, m1.model_params.head_model)
    for i in 1:length(m2.head_model.layers)
        m2.head_model.layers[i].weight .= simple_weights[i]
    end
    simple_weights = SimpleChains.weights(m1.expr_model.heuristic, m1.model_params.heuristic)
    for i in 1:length(m2.heuristic.layers)
        m2.heuristic.layers[i].weight .= simple_weights[i]
    end
    return m2
end


function flux_to_simple(m1::ExprModelSimpleChains, m2::ExprModel)
    simple_weights = SimpleChains.weights(m1.expr_model.args_model, m1.model_params.args_model)
    simple_biases = SimpleChains.biases(m1.expr_model.args_model, m1.model_params.args_model)
    for i in 1:length(m2.args_model.layers)
        simple_weights[i] .= m2.args_model.layers[i].weight
        simple_biases[i] .= m2.args_model.layers[i].bias 
    end
    simple_weights = SimpleChains.weights(m1.expr_model.head_model, m1.model_params.head_model)
    simple_biases = SimpleChains.biases(m1.expr_model.head_model, m1.model_params.head_model)
    for i in 1:length(m2.head_model.layers)
        simple_weights[i] .= m2.head_model.layers[i].weight
        simple_biases[i] .= m2.head_model.layers[i].bias 
    end
    simple_weights = SimpleChains.weights(m1.expr_model.heuristic, m1.model_params.heuristic)
    simple_biases = SimpleChains.biases(m1.expr_model.heuristic, m1.model_params.heuristic)
    for i in 1:length(m2.heuristic.layers)
        simple_weights[i] .= m2.heuristic.layers[i].weight
        simple_biases[i] .= m2.heuristic.layers[i].bias
    end 
    return m1
end


function get_conflicting_inequalities_filtered(training_samples)
    check_dict = Dict()
    new_samples = []
    for (sample_ind, (ds, I₊, I₋, tr)) in enumerate(training_samples)
        # add_sample = 0
        bad_ineq = []
        for (ind, ex) in enumerate(tr)
            is_pos = ind ∈ I₊
            is_neg = ind ∈ I₋
            if haskey(check_dict, ex)
                # add_sample 
                _, b, c, _ = check_dict[ex][end]
                if is_pos == b && is_neg == c
                    push!(check_dict[ex], (ind, is_pos, is_neg, sample_ind))
                else
                    push!(bad_ineq, ind)
                end
            else
                check_dict[ex] = [(ind, is_pos, is_neg, sample_ind)]
            end
        end
        if !isempty(bad_ineq)
            # @show bad_ineq
            hp_indices = findall(x->x ∈ bad_ineq, I₊)
            # @show unique(I₊)
            # @show unique(I₋)
            hn_indices = findall(x->x ∈ bad_ineq, I₋)
            # @assert abs(length(hp_indices) - length(hn_indices)) == length(hp_indices) + length(hn_indices)
            filtered_indices = vcat(hp_indices, hn_indices)
            I₊ = [x[2] for x in enumerate(I₊) if x[1] ∉ filtered_indices]
            I₋ = [x[2] for x in enumerate(I₋) if x[1] ∉ filtered_indices]
            @assert length(I₋) == length(I₊)
            push!(new_samples, (ds, I₊, I₋, tr))
        else
            push!(new_samples, (ds, I₊, I₋, tr))
        end
    end
    return new_samples
end


function reset_all_function_caches()
    empty!(memoize_cache(exp_size))
    empty!(memoize_cache(all_expand))
    empty!(memoize_cache(general_expr_cached_inference))
    empty!(memoize_cache(general_leaf_cached_inference))
end

function cache_status()
    (
    node_cache = round(Base.summarysize(nc) / 1_000_000, digits = 2),
    exp_size = round(Base.summarysize(memoize_cache(exp_size)) / 1_000_000, digits = 2),
    all_expand = round(Base.summarysize(memoize_cache(all_expand)) / 1_000_000, digits = 2),
    all_cached_inference = round((Base.summarysize(memoize_cache(general_expr_cached_inference)) + Base.summarysize(memoize_cache(general_leaf_cached_inference))) / 1_000_000, digits = 2),
    )
end


function reset_inference_caches()
    empty!(memoize_cache(general_expr_cached_inference))
    empty!(memoize_cache(general_leaf_cached_inference))
end