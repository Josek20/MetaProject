include("src/MyModule.jl")
using .MyModule: load_data, preprosses_data_to_expressions
using Metatheory
using Metatheory.Rewriters
using DataStructures

theory = @theory a b c begin
    a - a --> 0
    (a - b) - c --> a - (b - c)
    (a - b) - c --> (a - c) - b
end

data = load_data("data/test.json")[1:100]
data = preprosses_data_to_expressions(data)
theory = @theory a b c begin
    a * (b * c) --> (a * b) * c
    (a * b) * c --> a * (b * c)
    a + (b + c) --> (a + b) + c
    (a + b) + c --> a + (b + c)
    a + (b + c) --> (a + c) + b
    (a + b) + c --> (a + c) + b
    (a - b) + b --> a
    (-a + b) + a --> b


    a + b --> b + a
    a * (b + c) --> (a * b) + (a * c)
    (a + b) * c --> (a * c) + (b * c)
    (a / b) * c --> (a * c) / b



    -a --> -1 * a
    a - b --> a + -b
    1 * a --> a
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

mutable struct Node{E, P}
    ex::E
    rule_index::Int
    children::Vector{UInt64}
    parent::P
    depth::Int
    node_id::UInt64
end

function Node(ex::Expr, rule_index, parent, depth)
    Node(ex, rule_index, UInt64[],  parent, depth, hash(ex))
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


function push_to_tree!(soltree::Dict, new_node::Node)
    node_id = new_node.node_id
    if haskey(soltree, node_id)
        old_node = soltree[node_id]
        soltree[node_id] = new_node.depth < old_node.depth ? new_node : old_node
        return (false)
    else
        soltree[node_id] = new_node
        return (true)
    end
end


function expand_node!(parent::Node, soltree::Dict, heuristic::Function, open_list::PriorityQueue)
    ex = parent.ex
    succesors = filter(r -> r[2] != ex, collect(enumerate(execute(ex, theory))))

    for (rule_index, new_exp) in succesors
        new_node = Node(new_exp, rule_index, parent, parent.depth + 1)
        if push_to_tree!(soltree, new_node)
            enqueue!(open_list, new_node, heuristic(new_node.ex))
        else
            push!(terminal_nodes, new_node)
        end
    end
end

function build_tree!(soltree::Dict, heuristic::Function, open_list::PriorityQueue) 
    while length(open_list) > 0
        node = dequeue!(open_list)
        expand_node!(node, soltree, heuristic, open_list)
    end
end

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
            result = node
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
heuristic = exp_size
soltree = Dict{UInt64, Node}()
open_list = PriorityQueue{Node, Int32}()
close_list = Set{UInt64}

#heuristic(node) = model(node.ex)

root = Node(data[3], 0, nothing, 0)
soltree[root.node_id] = root
#open_list[root] = heuristic(root.ex)
enqueue!(open_list, root, heuristic(root.ex))
#push!(nodes_expression_built, root.ex)
build_tree!(soltree, heuristic, open_list)
smallest_node = extract_smallest_terminal_node()
proof_vector = extract_rules_applied(smallest_node)

