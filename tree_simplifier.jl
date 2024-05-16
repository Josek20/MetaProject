include("src/MyModule.jl")
using .MyModule: load_data, preprosses_data_to_expressions
using Metatheory
using Metatheory.Rewriters

theory = @theory a b c begin
    a - a --> 0
    (a - b) - c --> a - (b - c)
    (a - b) - c --> (a - c) - b
end

data = load_data("data/test.json")[1:100]
data = preprosses_data_to_expressions(data)
theory = @theory a b c begin
    a * (b * c) --> (a * b) * c
    a + (b + c) --> (a + b) + c

    a + b --> b + a
    a * (b + c) --> (a * b) + (a * c)
    (a + b) * c --> (a * c) + (b * c)
    (a / b) * c --> (a * c) / b


  #  -a == -1 * a
 #   a - b == a + -b
#    1 * a == a
    a + a --> 2*a

    0 * a --> 0
    a + 0 --> a
    a - a --> 0
    a <= a --> 1
    (a / b) * b --> a

    min(a, min(a, b)) --> min(a, b)
    min(min(a, b), b) --> min(a, b)
    max(a, max(a, b)) --> max(a, b)
    max(max(a, b), b) --> max(a, b)

    max(a, min(a, b)) --> a
    min(a, max(a, b)) --> a
    max(min(a, b), b) --> b
    min(max(a, b), b) --> b
end

mutable struct Node
    ex::Union{Expr, Int}
    rule_index::Int
    children::Vector
    parent
end

exp_size(node::Node) = exp_size(node.ex)
exp_size(ex::Expr) = sum(exp_size.(ex.args))
exp_size(ex::Symbol) = 1
exp_size(ex::Int) = 1

# get all availabel rules to apply
execute(ex, theory) = map(th->Postwalk(Chain([th]))(ex), theory)


# build a tree
depth = 50
nodes_expression_built = Vector()
terminal_nodes = Vector()

function build_next!(parent::Node)
    new_nodes = Vector()
    for (rule_index, new_exp) in enumerate(execute(parent.ex, theory))
        if !(new_exp in nodes_expression_built)
            new_node = Node(new_exp, rule_index, Vector(), parent)
            push!(new_nodes, new_node)
            push!(nodes_expression_built, new_exp)
        end
    end
    if length(new_nodes) == 0
        push!(terminal_nodes, parent)
    end
    #new_nodes = map(tmp->tmp ? Node(tmp[2], tmp[1], Vector(), parent), enumerate(execute(parent.ex, theory)))
    parent.children = new_nodes
    return new_nodes
end

function build_tree!(root::Node, counter::Int) 
    counter += 1
    if counter >= depth
        return
    end
    new_nodes_depth = build_next!(root)
    if length(new_nodes_depth) == 0
        return
    end
    #println(root)
    sorted_nodes_depth = sort(new_nodes_depth, by=exp_size)
    for new_node in sorted_nodes_depth
        _ = build_tree!(new_node, counter)
    end
end

#=
function extract_rules_applied(node::Node, proof_vector::Vector)
    if isnothing(node.parent)
        return
    end
    extract_rules_applied(node.parent, proof_vector) 
    push!(proof_vector, node.rule_index)
    return proof_vector
end
=#
function extract_rules_applied(node::Node) 
    proof_vector = Vector()
    while !isnothing(node.parent)
        push!(proof_vector, node.rule_index)
        node = node.parent
    end
    return reverse(proof_vector)
end

function extract_smallest_terminal_node()
    result = terminal_nodes[1]
    for node in terminal_nodes[2:end]
        if exp_size(result.ex) > exp_size(node.ex)
            result = node.ex
        end
    end
    return result
end

#=
for (index, ex) in enumerate(data[3:3])
    global nodes_expression_built
    global terminal_nodes
    root = Node(ex, 0, Vector(), nothing)
    push!(nodes_expression_built, root.ex)
    build_tree!(root, 1)
    empty!(nodes_expression_built)
    empty!(terminal_nodes)
    println(root)
end
=#
root = Node(data[3], 0, Vector(), nothing)
push!(nodes_expression_built, root.ex)
build_tree!(root, 1)
smallest_node = extract_smallest_terminal_node()
proof_vector = extract_rules_applied(smallest_node)

