function test_heuristic(heuristic, data, max_steps, max_depth, variable_names, theory)
    result = []
    result_proof = []
    simp_expressions = []
    for (index, i) in enumerate(data)
        # simplified_expression, _, _, _, _, _, proof_vector = heuristic_forward_pass(heuristic, i, max_steps, max_depth)
        simplified_expression, depth_dict, big_vector, saturated, hp, hn, root, proof_vector = heuristic_forward_pass(heuristic, i, max_steps, max_depth, all_symbols, theory, variable_names)
        original_length = exp_size(i)
        simplified_length = exp_size(simplified_expression)
        push!(result, original_length - simplified_length)
        push!(result_proof, proof_vector)
        push!(simp_expressions, simplified_expression)
    end
    return result, result_proof, simp_expressions
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
    @assert counter_failed == 0
end


function test_expr_embedding(policy, samples, theory, symbols_to_index, all_symbols, variable_names)
    full_proof_tmp = []
    counter = 0
    # @show size(variable_names)
    # @show size(var_one_enc)
    for (ne,sample) in enumerate(samples)
        ex = sample.initial_expr
        @show ne
        # for pr in sample.proof
        applicable_rueles = filter(r-> r[2] != ex, execute(ex, theory))
        tmp = []
        finall_ind = 0
        for (ind,i) in enumerate(applicable_rueles)
            # em = nothing
            # try

            em = ex2mill(i[2], symbols_to_index, all_symbols, variable_names)
            # catch e
            #     @show i[1]
            #     @show i[2]
            # end
            o = policy(em)
            if o in tmp 
                @show i
                # @show ind
                index = findfirst(x->x==o, tmp)
                # @show ind
                @show applicable_rueles[index]
                counter += 1 
            else
                push!(tmp, o)
            end
            # @show pr
            # @show i[1]
            # if i[1] == pr
            #     finall_ind = 1
            #     ex = i[2]
            # end
        end 
            # @assert finall_ind == 1
        @assert counter == 0
        # end
    end 
    @show counter
end


function test_expr_embedding_simple_examples(heuristic, symbols_to_index, all_symbols, variable_names)
    ex1 = :(100 - 11 <= 1011)
    ex2 = :(100 - 1011 <= 11)
    p1 = ex2mill(ex1, symbols_to_index, all_symbols, variable_names)
    p2 = ex2mill(ex2, symbols_to_index, all_symbols, variable_names)
    @assert heuristic(p1) != heuristic(p2) 
    ex1 = :(100 - v0 * 12 <= 1011)
    ex2 = :(100 - 12 * v0 <= 1011)
    p1 = ex2mill(ex1, symbols_to_index, all_symbols, variable_names)
    p2 = ex2mill(ex2, symbols_to_index, all_symbols, variable_names)
    @assert heuristic(p1) != heuristic(p2)
end


function test_old_new_ex2mill(expressions::Vector{Expr}, heuristic, symbols_to_index, all_symbols, variable_names)
    for ex in expressions
        @show ex
        eo = ex2mill(ex, symbols_to_index, all_symbols, variable_names)
        en = MyModule.single_fast_ex2mill(ex, MyModule.sym_enc)
        @assert heuristic(eo) == heuristic(en)
    end
end


function test_old_new_ex2mill(expressions::Vector{Expr}, heuristic, symbols_to_index, all_symbols, variable_names, sym_enc)
    cache = Dict()
    for ex in expressions
        @show ex
        a = @time begin
            eo = ex2mill(ex, symbols_to_index, all_symbols, variable_names, Dict(), heuristic)
            ho = heuristic(eo)
        end
        b = @time begin
            en = MyModule.cached_inference(ex, cache, heuristic, all_symbols, sym_enc)
            hn = heuristic.heuristic(heuristic.joint_model(vcat(en, zeros(Float32, 2, 1))))
        end
        # println("Base method took $a")
        # println("New method took $b")
        @assert abs(only(ho) - only(hn)) <= 0.001 
    end
end


function test_stats_proofs(initial_expr, proofs)
    for i in proofs
        position, rule = i
        @show initial_expr
        MyModule.my_rewriter!(position, initial_expr, rule)
        # MyModule.single_fast_ex2mill(initial_expr)
    end
end
