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

