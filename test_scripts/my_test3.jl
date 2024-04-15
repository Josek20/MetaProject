using Metatheory
using Statistics
using GraphNeuralNetworks
using Flux
using JSON


all_symbols = [[Symbol("v$i") for i in 0:12]; [:a, :b, :c]]

function encode_expression(expr::ENodeTerm)
    encoding = zeros(Int, 5+length(all_symbols))  # Create an array for one-hot encoding with 10 elements

    op = expr.operation
    if op in (:+, :-, :*, :/)
        # Encode algebraic operations [+,-,*,/] in the first 4 indices
        op_index = findfirst(x-> x == op, (:+, :-, :*, :/))
        encoding[op_index] = 1
    end
    return encoding
end

function encode_expression(expr::ENodeLiteral)
    encoding = zeros(Int, 5 + length(all_symbols))  # Create an array for one-hot encoding with 10 elements
    # Check if expr is a variable or a constant
    expr_value = expr.value
    if expr_value in all_symbols
        # Encode variables [a, b, c] in the last 3 indices
        variable_index = findfirst(x-> x == expr_value, all_symbols)
        encoding[end-length(all_symbols)+variable_index] = 1
    else
        # Encode constants in the 5th index
        encoding[5] = expr_value
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


function preprosses_rule(rule)
    result = []
    for i in rule.args
        if typeof(i) == PatTerm
            for j in i.args
                tmp1 = string(j)
                tmp2 = last(tmp1, length(tmp1) - 1)
                push!(result, Symbol(tmp2))
            end
        else
            tmp1 = string(i)
            tmp2 = last(tmp1, length(tmp1) - 1)
            push!(result, Symbol(tmp2))
        end
    end
    return result
end

function encode_theory_rules(theory_rules)
    encoding = zeros(length(theory_rules), 10)
    for (index, rule) in enumerate(theory_rules)
        lhs_expr = Expr(rule.left.exprhead,Symbol(rule.left.operation),preprosses_rule(rule.left)...) 
        #println(lhs_expr)
        #left_encoding = encode_egraph_classes(EGraph(rule.left))
        left_encoding = encode_egraph_classes(EGraph(lhs_expr))
        #right_encoding = encode_expression(rule.right)
        #rule_encoding = vcat(left_encoding, right_encoding)
        encoding[index,:] .= mean(left_encoding, dims=2)
    end
    return encoding
end

function get_nodes_number(egraph::EGraph)
    counter = 0
    for (_, nodes) in egraph.classes
        for _ in nodes
            counter += 1
        end
    end
    return counter
end

function get_adjacency_matrix(egraph::EGraph)
    num_nodes = get_nodes_number(egraph)
    adj_matrix = zeros(Int, num_nodes, num_nodes)
    counter = 1
    for (_, nodes) in egraph.classes
        for node in nodes
            for edge in egraph.memo[node]
                adj_matrix[counter, edge] = 1
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


#=
my_rules = @theory a b c begin
    a - a --> 0
    (a + b) + c --> a + (b + c)
end

# Get the one-hot encoding for the rules
rules_encoding = encode_theory_rules(my_rules)
=#
# Get the adjacency matrix for the graph
adjacency_matrix = get_adjacency_matrix(EGraph(ex))

# location /home/rudy/PhdProject/neural-rewriter/data/Halide/train.json
train_data = JSON.parsefile("/home/rudy/PhdProject/neural-rewriter/data/Halide/train.json")
#rules_encoding * (graph_encoding * adjacency_matrix)

for (index, i) in enumerate(train_data[1:1000])
    expression_from_string = Meta.parse(i[1])
    graph_from_expression = EGraph(expression_from_string)
    #println(index, i[1])
    tmp_graph_encoding = encode_egraph_classes(graph_from_expression)
end
#=
model = GNNChain(GCNConv(10 => 64),
                 x -> relu.(x),
                 GCNConv(64 => length(my_rules), relu),
                 GlobalPool(mean),
                 Dense(length(my_rules),1))

lr = 1e-4
opt = Flux.setup(Adam(lr), model)
=#
# training pipelin

#for epoch in 1:100
#    
#end


