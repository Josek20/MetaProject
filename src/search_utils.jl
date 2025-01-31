function exp_size(ex::Union{Expr, Symbol, Number})
    if isa(ex, Symbol) || isa(ex, Number)
        return 1f0
    end
    res = sum(exp_size(a) for a in ex.args)
    return ex.head in [:&&, :||] ? res + 1 : res
end


function exp_size(x::NodeID)
    node = nc[x]
    if node == nullnode
        return 0f0
    elseif !(node.iscall) && node.head ∉ [:&&, :||]
        return 1f0
    end
    return exp_size(node.left) + exp_size(node.right) + 1
end


function all_expand(ex::Expr, theory)
    return [ex], [()]
end


function all_expand(ex::NodeID, theory)
    return [ex], [()]
end


function show_proof(initial_expr::NodeID, proof, theory)
    ex = initial_expr
    for (pos, rule_index) in proof
        r = theory[rule_index]
        for j in pos
            ex = r(ex)
        end
    end
end


function show_proof(init_ex, proof)
    ex = copy(init_ex)
    for i in proof
        position, rule = i
        println("=====================================================================================================")
        println("Start: $(ex)")
        println("Rule: $(theory[rule])")
        new_ex = my_rewriter!(position, ex, theory[rule])
        if new_ex isa Nothing
            println("Result: $(ex)")
        else
            println("Result: $(new_ex)")
            ex = new_ex
        end
    end
end