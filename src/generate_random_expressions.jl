# To generate a random binary tree with n internal nodes, we use the following one-pass procedure.
# Starting with an empty root node, we determine at each step the position of the next internal nodes
# among the empty nodes, and repeat until all internal nodes are allocated.
# Start with an empty node, set e = 1;
# while n > 0 do
# Sample a position k from K(e, n);
# Sample the k next empty nodes as leaves;
# Sample an operator, create two empty children;
# Set e = e − k + 1 and n = n − 1;
# end
# We denote by e the number of empty nodes, by n > 0 the number of operators yet to be generated,
# and by K(e, n) the probability distribution of the position (0-indexed) of the next internal node to
# allocate.
# To calculate K(e, n), let us define D(e, n), the number of different binary subtrees that can be
# generated from e empty elements, with n internal nodes to generate. We have
# D(0, n) = 0
# D(e, 0) = 1
# D(e, n) = D(e − 1, n) + D(e + 1, n − 1)
# The first equation states that no tree can be generated with zero empty node and n > 0 operators. The
# second equation says that if no operator is to be allocated, empty nodes must all be leaves and there
# is only one possible tree. The last equation states that if we have e > 0 empty nodes, the first one is
# either a leaf (and there are D(e − 1, n) such trees) or an internal node (D(e + 1, n − 1) trees). This
# allows us to compute D(e, n) for all e and n.
# To calculate distribution K(e, n), observe that among the D(e, n) trees with e empty nodes and n
# operators, D(e + 1, n − 1) have a binary node in their first position. Therefore
# P(K(e, n) = 0) = D(e + 1, n − 1)
# D(e, n)
# Of the remaining D(e − 1, n) trees, D(e, n − 1) have a binary node in their first position (same
# argument for e − 1), that is
# P(K(e, n) = 1) = D(e, n − 1)
# D(e, n)
# By induction over k, we have the general formula
# P
# To compute L(e, n), we derive D(e, n), the number of subtrees with n internal nodes that can be
# generated from e empty nodes. We have, for all n > 0 and e:
# D(0, n) = 0
# D(e, 0) = 1
# D(e, n) = D(e − 1, n) + D(e, n − 1) + D(e + 1, n − 1)
# The first equation states that no tree can be generated with zero empty node and n > 0 operators.
# The second says that if no operator is to be allocated, empty nodes must all be leaves and there is
# only one possible tree. The third equation states that with e > 0 empty nodes, the first one will either
# be a leaf (D(e − 1, n) possible trees), a unary operator (D(e, n − 1) trees), or a binary operator
# (D(e + 1, n − 1) trees).
# To derive L(e, n), we observe that among the D(e, n) subtrees with e empty nodes and n internal
# nodes to be generated, D(e, n − 1) have a unary operator in position zero, and D(e + 1, n − 1) have
# a binary operator in position zero. As a result, we have
# P

function generate_expr(n=10)
    while n > 0
        e = e - k + 1
        n -= 1
    end
end

function compute_D(max_e, max_n)
    D = zeros(Int, max_e + 1, max_n + 1)
    D[2:end, 1] .= 1  # Base case

    for n in 1:max_n
        for e in 1:max_e
            if n > 1 && e > 1
                D[e, n] = D[e - 1, n] + D[e + 1, n - 1]
            else
                D[e, n] = (e == 1) ? 1 : 0
            end
        end
    end
    
    return D
end

function compute_K(D, max_e, max_n)
    K = zeros(Float64, max_e + 1, max_n + 1)
    
    for n in 1:max_n
        for e in 1:max_e
            if e >= 1 && n > 1
                K[e, n] = D[e + 1, n - 1] / D[e, n]
            end
            if e >= 1 && n > 1
                K[e, n] = D[e, n - 1] / D[e, n]
            end
        end
    end
    
    return K
end

using Random

function generate_random_binary_tree(n, D, K)
    empty_nodes = 1  # Start with one empty root node
    internal_nodes_remaining = n
    tree = []

    while internal_nodes_remaining > 0
        # Sample the position k
        k_probs = [K[empty_nodes, internal_nodes_remaining] for _ in 1:empty_nodes]
        @show k_probs
        k = rand(1:length(k_probs), p=k_probs) - 1
        
        # Sample k next empty nodes as leaves
        leaves = rand(k)
        
        # Sample an operator and create two empty children
        # Here you would append the operator and children to your tree structure
        push!(tree, (operator = rand(["+", "-", "*", "/"]), leaves))
        
        # Update e and n
        empty_nodes = empty_nodes - k + 1
        internal_nodes_remaining -= 1
    end
    
    return tree
end


function generate_expr(initial_ex::Expr, theory::Vector, max_proof_length=10, min_rule_length=5, number_of_samples=10)
    new_rules = [RewriteRule(i.right, i.left) for i in theory if !isa(i, DynamicRule)]
    exp_cache = LRU{Expr, Vector}(maxsize=1000)
    o = rand(min_rule_length:max_proof_length,number_of_samples)
    for j in o
        copy_initial = copy(initial_ex)
        for i in 1:j
            possible_exp = execute(copy_initial, new_rules, exp_cache)
            new_ex = rand(possible_exp)
            copy_initial = new_ex
        end
        @show copy_initial
    end
end