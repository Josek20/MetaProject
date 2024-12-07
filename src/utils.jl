function get_training_data_from_proof(proof::Vector, initial_expression::Expr)
    soltree = Dict{UInt64, Node}()
    exp_cache = LRU(maxsize=10_000)
    expr_cache = LRU(maxsize=10_000)
    root = Node(initial_expression, (0,0), nothing, 0, expr_cache)
    soltree[root.node_id] = root
    ex = initial_expression
    smallest_node = root
    for (ind, i) in enumerate(proof)
        # reached_goal = build_tree!(soltree, heuristic, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, max_steps, max_depth, expansion_history, theory, variable_names, cache, exp_cache, size_cache, expr_cache, alpha)
        # @show i
        succesors = execute(smallest_node.ex, theory, exp_cache)
        new_nodes = map(x-> Node(x[2], x[1], smallest_node.node_id, smallest_node.depth + 1, expr_cache), succesors)
        filtered_new_nodes = filter(x-> push_to_tree!(soltree, x), new_nodes)
        nodes_ids = map(x->x.node_id, filtered_new_nodes)
        append!(smallest_node.children, nodes_ids)
        # @show filtered_new_nodes[1].rule_index
        # @show [j.rule_index for j in filtered_new_nodes]
        smallest_node = only(filter(x->x.rule_index == i, filtered_new_nodes))
        # ex = smallest_node.ex.ex
    end
    big_vector, hp, hn, proof_vector = extract_training_data(smallest_node, soltree)
    return big_vector, hp, hn 
end


function train_heuristic_on_data_overfit(heuristic, training_samples, pipeline_config, optimizer=Adam())
    pc = Flux.params([heuristic.head_model, heuristic.aggregation, heuristic.joint_model, heuristic.heuristic])    
    loss_stats = [[]]
    ireducible_stats = []
    for (ind,sample) in enumerate(training_samples)
        bd, hp, hn = MyModule.get_training_data_from_proof(sample.proof, sample.initial_expr)
        @show ind
        for _ in 1:10
            sa, grad = Flux.Zygote.withgradient(pc) do
                MyModule.loss(heuristic, bd, hp, hn)
            end
            @show sa
            if any(g->any(isinf, g) || any(isnan, g), grad)
                println("Gradient is Inf/NaN")
                serialize("models/trainied_heuristic_inf.bin", heuristic)
                serialize("data/training_data/training_sample_inf_grad.bin", sample)
                @assert 0 == 1
            end
            Flux.update!(optimizer, pc, grad)
        end
        # if mod(ind, 100) == 0
        #     pipeline_config.heuristic = heuristic
        #     empty!(pipeline_config.inference_cache)
        #     empty!(pipeline_config.heuristic_cache)
        #     number_of_irreducible, number_of_improved = MyModule.train_heuristic!(pipeline_config)
        #     push!(ireducible_stats, number_of_irreducible)
        #     for (ind,sample) in enumerate(training_samples)
        #         bd, hp, hn = MyModule.get_training_data_from_proof(sample.proof, sample.initial_expr)
        #         sa = MyModule.loss(heuristic, bd, hp, hn)
        #         push!(loss_stats[end], sa)
        #     end
        #     push!(loss_stats, [])
        # end
    end
    return heuristic, ireducible_stats, loss_stats
end


function train_heuristic_on_data_epochs(heuristic, training_samples, pipeline_config, optimizer=Adam(), epochs=10)
    pc = Flux.params([heuristic.head_model, heuristic.aggregation, heuristic.joint_model, heuristic.heuristic])    
    loss_stats = [[]]
    ireducible_stats = []
    for ep in 1:epochs
        @show ep
        # for (ind,sample) in enumerate(training_samples)
        for ind in 1:length(training_samples)
            sample = Statistics.sampe(training_samples)
            bd, hp, hn = MyModule.get_training_data_from_proof(sample.proof, sample.initial_expr)
            @show ind
            sa, grad = Flux.Zygote.withgradient(pc) do
                MyModule.loss(heuristic, bd, hp, hn)
            end
            @show sa
            if any(g->any(isinf, g) || any(isnan, g), grad)
                println("Gradient is Inf/NaN")
                serialize("models/trainied_heuristic_inf.bin", heuristic)
                serialize("data/training_data/training_sample_inf_grad.bin", sample)
                @assert 0 == 1
            end
            Flux.update!(optimizer, pc, grad)
        end
        # pipeline_config.heuristic = heuristic
        # empty!(pipeline_config.inference_cache)
        # empty!(pipeline_config.heuristic_cache)
        # number_of_irreducible, number_of_improved = MyModule.train_heuristic!(pipeline_config)
        # push!(ireducible_stats, number_of_irreducible)
        # for (ind,sample) in enumerate(training_samples)
        #     bd, hp, hn = MyModule.get_training_data_from_proof(sample.proof, sample.initial_expr)
        #     sa = MyModule.loss(heuristic, bd, hp, hn)
        #     push!(loss_stats[end], sa)
        # end
        # push!(loss_stats, [])
    end
    return heuristic, ireducible_stats, loss_stats
end


function create_batches_varying_length(data, n_batches, size_cache=LRU(maxsize=10000))
    # Step 1: Sort the data by length of each element
    sorted_data = sort(data, by = x->MyModule.exp_size(x,size_cache))

    # Step 2: Calculate the number of elements in each batch
    total_data_length = length(sorted_data)
    
    # Determine the approximate size of each batch, but with varying lengths
    batch_sizes = [floor(Int, total_data_length / n_batches) for _ in 1:n_batches]
    
    # Distribute the remainder data (if total_data_length is not perfectly divisible)
    remainder = total_data_length % n_batches
    for i in 1:remainder
        batch_sizes[i] += 1
    end

    # Step 3: Split the sorted data into batches according to the calculated sizes
    batches = []
    start_idx = 1
    for batch_size in batch_sizes
        end_idx = start_idx + batch_size - 1
        push!(batches, sorted_data[start_idx+100:start_idx + 200 - 1])
        start_idx = end_idx + 1
    end

    return batches
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
