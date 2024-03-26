using Metatheory
using Statistics

# Define a function to create a one-hot encoding for an expression
function encode_expression(expr)
    # encodings schema: [+,-,*,/,<constant>,variable1,variable2,...]
    encoding = zeros(Float64, 10)  # Create an array for one-hot encoding with 10 elements
    
    if isa(expr, ENodeTerm)
        # Check the operation of the expression
        op = expr.operation
        if op in (:+, :-, :*, :/)
            # Encode algebraic operations [+,-,*,/] in the first 4 indices
            op_index = findfirst(x-> x == op, (:+, :-, :*, :/))
            encoding[op_index] = 1
        end
    elseif isa(expr, ENodeLiteral)
        # Check if expr is a variable or a constant
        expr_value = expr.value
        if expr_value in (:a, :b, :c)
            # Encode variables [a, b, c] in the last 3 indices
            variable_index = findfirst(x-> x == expr_value, (:a, :b, :c))
            encoding[end-3+variable_index] = 1
        else
            # Encode constants in the 5th index
            encoding[5] = expr_value
        end
    end
    
    return encoding
end

# Define a function to create an EGraph from an expression and return the one-hot encoding
function encode_egraph_classes(g::EGraph) 
    encodings = []
    for (i, class_nodes) in g.classes
        #tmp = []
        for node in class_nodes 
            encoding = encode_expression(node)
            push!(encodings, encoding)
        end
        #push!(encodings, mean(tmp))
    end
    
    return hcat(encodings...)
end

#function encode_egraph_nodes(g::EGraph)
#    encodings = []
#    for (node) in g.memo
#
#    end
#    return hcat(encodings...)
#end

function preprosses_rule(rule)
    result = []
    for i in rule.args
        tmp1 = string(i)
        tmp2 = last(tmp1, length(tmp1) - 1)
        push!(result, Symbol(tmp2))
    end
    return result
end

function encode_theory_rules(theory_rules)
    encoding = zeros(length(theory_rules), 10)
    for (index, rule) in enumerate(theory_rules)
        lhs_expr = Expr(rule.left.exprhead,Symbol(rule.left.operation),preprosses_rule(rule.left)...) 
        println(lhs_expr)
        left_encoding = encode_egraph_classes(EGraph(lhs_expr))
        #right_encoding = encode_expression(rule.right)
        #rule_encoding = vcat(left_encoding, right_encoding)
        encoding[index,:] .= mean(left_encoding, dims=2)
    end
    return encoding
end

function get_adjacency_matrix1(g::EGraph)
    adj_matrix = zeros(length(g.memo), length(g.memo))
    for (i, j) in g.memo
        adj_matrix[] .= 1 
    end
end


function get_adjacency_matrix(egraph::EGraph)
    num_nodes = length(egraph.memo)
    adj_matrix = zeros(Int, num_nodes, num_nodes)
    counter = 1
    for (id, nodes) in egraph.classes
        for node in nodes
            for edge in egraph.memo[node]
                adj_matrix[counter, edge] .= 1
                counter += 1
            end
        end
    end
    
    return adj_matrix
end

# Example expression
ex = :(a - a)

# Get the one-hot encoding for the EGraph
graph_encoding = encode_egraph_classes(EGraph(ex))


my_rules = @theory a b c begin
    a - a --> 0
    (a + b) + c --> a + (b + c)
end

# Get the one-hot encoding for the rules
#rules_encoding = encode_theory_rules(my_rules)


adj_matrix = get_adjacency_matrix(EGraph(ex))



