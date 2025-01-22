function get_training_data_from_proof_with_soltree(proof::Vector, initial_expression::Expr)
    soltree = Dict{UInt64, Node}()
    exp_cache = LRU(maxsize=10_000)
    # expr_cache = LRU(maxsize=10_000)
    open_list = PriorityQueue{Node, Float32}()
    close_list = Set{UInt64}()

    root = Node(initial_expression, (0,0), nothing, 0)
    soltree[root.node_id] = root
    ex = initial_expression
    smallest_node = root
    for (ind, i) in enumerate(proof)
        # reached_goal = build_tree!(soltree, heuristic, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, max_steps, max_depth, expansion_history, theory, variable_names, cache, exp_cache, size_cache, expr_cache, alpha)
        # @show i
        succesors = execute(smallest_node.ex, theory, exp_cache)
        new_nodes = map(x-> Node(x[2], x[1], smallest_node.node_id, smallest_node.depth + 1), succesors)
        filtered_new_nodes = filter(x-> push_to_tree!(soltree, x), new_nodes)
        nodes_ids = map(x->x.node_id, filtered_new_nodes)
        append!(smallest_node.children, nodes_ids)
        # @show filtered_new_nodes[1].rule_index
        # @show [j.rule_index for j in filtered_new_nodes]
        smallest_node = only(filter(x->x.rule_index == i, filtered_new_nodes))
        # ex = smallest_node.ex.ex
    end
    # function build_tree!(soltree::Dict{UInt64, Node}, heuristic, open_list::PriorityQueue, close_list::Set{UInt64}, encodings_buffer::Dict{UInt64, ProductNode}, all_symbols::Vector{Symbol}, symbols_to_index::Dict{Symbol, Int64}, max_steps, max_depth, expansion_history, theory, variable_names, cache, exp_cache, size_cache, expr_cache, alpha)
    build_tree!(soltree, heuristic, open_list, close_list, new_all_symbols, sym_enc, 100, 50, theory, exp_cache)
    big_vector, hp, hn, proof_vector, _ = extract_training_data(smallest_node, soltree)
    return big_vector, hp, hn
end


function get_training_data_from_proof(proof::Vector, initial_expression::Expr)
    soltree = Dict{UInt64, Node}()
    exp_cache = LRU(maxsize=10_000)
    expr_cache = LRU(maxsize=10_000)
    root = Node(initial_expression, (0,0), nothing, 0)
    soltree[root.node_id] = root
    ex = initial_expression
    smallest_node = root
    for (ind, i) in enumerate(proof)
        # reached_goal = build_tree!(soltree, heuristic, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, max_steps, max_depth, expansion_history, theory, variable_names, cache, exp_cache, size_cache, expr_cache, alpha)
        # @show i
        succesors = execute(smallest_node.ex, theory, exp_cache)
        new_nodes = map(x-> Node(x[2], x[1], smallest_node.node_id, smallest_node.depth + 1), succesors)
        filtered_new_nodes = filter(x-> push_to_tree!(soltree, x), new_nodes)
        nodes_ids = map(x->x.node_id, filtered_new_nodes)
        append!(smallest_node.children, nodes_ids)
        # @show filtered_new_nodes[1].rule_index
        # @show [j.rule_index for j in filtered_new_nodes]
        smallest_node = only(filter(x->x.rule_index == i, filtered_new_nodes))
        # ex = smallest_node.ex.ex
    end
    big_vector, hp, hn, proof_vector, _ = extract_training_data(smallest_node, soltree)
    return big_vector, hp, hn
end


function train_heuristic_on_data_overfit(heuristic, training_samples, pipeline_config, optimizer=Adam())
    opt_state = Flux.setup(optimizer, heuristic)
    loss_stats = [[]]
    ireducible_stats = []
    matched_stats = []
    for (ind,sample) in enumerate(training_samples)
        bd, hp, hn = MyModule.get_training_data_from_proof(sample.proof, sample.initial_expr)
        @show ind
        for _ in 1:10
            sa, grad = Flux.Zygote.withgradient(heuristic) do hfun
                MyModule.loss(hfun, bd, hp, hn)
            end
            @show sa
            if any(g->any(isinf, g) || any(isnan, g), grad)
                println("Gradient is Inf/NaN")
                serialize("models/trainied_heuristic_inf.bin", heuristic)
                serialize("data/training_data/training_sample_inf_grad.bin", sample)
                @assert 0 == 1
            end
            Flux.update!(optimizer, heuristic, grad)
        end
        if mod(ind, 100) == 0
            matched_count = heuristic_sanity_check(heuristic, training_samples, [])
            push!(matched_stats, matched_count)
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
        end
    end
    return heuristic, ireducible_stats, loss_stats, matched_stats
end


function train_heuristic_on_data_epochs(heuristic, training_samples, pipeline_config, optimizer=ADAM(), epochs=10)
    opt_state = Flux.setup(optimizer, heuristic)
    loss_stats = [[]]
    ireducible_stats = []
    matched_stats = []
    length_diff_stats = []
    samples = [MyModule.get_training_data_from_proof(sample.proof, sample.initial_expr) for sample in training_samples]
    for ep in 1:epochs
        @show ep
        total_ep_loss = 0
        for (ind,sample) in enumerate(samples)
        # for ind in 1:length(training_samples)
            # sample = Statistics.sample(training_samples)
            # bd, hp, hn = MyModule.get_training_data_from_proof(sample.proof, sample.initial_expr)
            # bd = sample.training_data
            # hp = sample.hp
            # hn = sample.hn
            @show ind
            sa, grad = Flux.Zygote.withgradient(pc) do
                MyModule.loss(heuristic, bd, hp, hn)
            end
            @show sa
            total_ep_loss += sa
            if any(g->any(isinf, g) || any(isnan, g), grad)
                println("Gradient is Inf/NaN")
                serialize("models/trainied_heuristic_inf.bin", heuristic)
                serialize("data/training_data/training_sample_inf_grad.bin", sample)
                @assert 0 == 1
            end
            Flux.update!(optimizer, pc, grad)
        end
        matched_count, length_diff = heuristic_sanity_check(heuristic, training_samples, [])
        push!(matched_stats, matched_count)
        push!(length_diff_stats, length_diff)
        push!(loss_stats, total_ep_loss)
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
    return heuristic, ireducible_stats, loss_stats, matched_stats, length_diff_stats
end



function train_heuristic_on_data_epochs_batched(heuristic, training_samples, pipeline_config; optimizer=Adam(), epochs=10, batch_size=10)
    # opt_state = Flux.setup(optimizer, heuristic)
    loss_stats = [[]]
    ireducible_stats = []
    matched_stats = []
    length_diff_stats = []
    samples = [MyModule.get_training_data_from_proof(sample.proof, sample.initial_expr) for sample in training_samples]
    batched_samples = []
    for i in 1:batch_size:length(samples)-1
        pds = reduce(catobs, [j[1] for j in samples[i:batch_size - 1 + i]])
        hps = []
        hns = []
        for (ind, (bd, hp, hn)) in enumerate(samples[i:batch_size - 1 + i])
            # hps = reduce(vcat, [j[2] for j in samples[batch_size*(i - 1) + 1:batch_size*i]])
            # hns = reduce(vcat, [j[3] for j in samples[batch_size*(i - 1) + 1:batch_size*i]])
            # hp .+= (ind - 1) * (length(hps))
            # hn .+= (ind - 1) * (length(hns))
            hp .+= size(bd.data.head)[2]
            hn .+= size(bd.data.head)[2]
            append!(hps, hp) 
            append!(hns, hn) 
        end
        # for (pn, hp, hn) in samples[batch_size*(i - 1) + 1:batch_size*i]
        #     push!()
        # end
        push!(batched_samples, (pds, hps, hns))
    end
    for ep in 1:epochs
        @show ep
        total_ep_loss = 0
        for (ind,(bd, hp, hn)) in enumerate(batched_samples)
        # for ind in 1:length(training_samples)
            # sample = Statistics.sample(training_samples)
            # bd, hp, hn = MyModule.get_training_data_from_proof(sample.proof, sample.initial_expr)
            # bd = sample.training_data
            # hp = sample.hp
            # hn = sample.hn
            @show ind
            # sa, grad = Flux.Zygote.withgradient(heuristic) do hfun
            #     MyModule.loss(hfun, bd, hp, hn)
            # end
            sa, grad = Flux.Zygote.withgradient(pc) do
                MyModule.loss(heuristic, bd, hp, hn)
            end
            @show sa
            total_ep_loss += sa
            if any(g->any(isinf, g) || any(isnan, g), grad)
                println("Gradient is Inf/NaN")
                serialize("models/trainied_heuristic_inf.bin", heuristic)
                serialize("data/training_data/training_sample_inf_grad.bin", sample)
                @assert 0 == 1
            end
            Flux.update!(optimizer, pc, grad)
            # Optimisers.update!(opt_state, heuristic, grad[1])
        end
        matched_count, length_diff = heuristic_sanity_check(heuristic, training_samples, [])
        push!(matched_stats, matched_count)
        push!(length_diff_stats, length_diff)
        push!(loss_stats, total_ep_loss)
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
    return heuristic, ireducible_stats, loss_stats, matched_stats, length_diff_stats
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


function reset_all_function_caches()
    empty!(memoize_cache(exp_size))
    empty!(memoize_cache(all_expand))
    empty!(memoize_cache(interned_cached_inference!))
    empty!(memoize_cache(hashed_expr_cached_inference!))
    empty!(memoize_cache(expr_cached_inference!))
end


function reset_inference_caches()
    empty!(memoize_cache(interned_cached_inference!))
    empty!(memoize_cache(hashed_expr_cached_inference!))
    empty!(memoize_cache(expr_cached_inference!))
end