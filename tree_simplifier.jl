include("src/MyModule.jl")
include("tree_lstm.jl")
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
#data = [:(a - b + c / 109 * 109)]
#abstract type LikelihoodModel end
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
    #min(a + b, a + c) --> min(b, c)
end

mutable struct Node{E, P}
    ex::E
    rule_index::Int64
    children::Vector{UInt64}
    parent::P
    depth::Int64
    node_id::UInt64
    #expression_encoding::Union{MyNodes, Nothing}
    expression_encoding::MyNodes
end

Node(ex::Int, rule_index, parent, depth, ee) = Node(ex, rule_index, UInt64[],  parent, depth, hash(ex), ee)

function Node(ex::Expr, rule_index::Int64, parent::UInt64, depth::Int64, ee::MyNodes)
    Node(ex, rule_index, UInt64[],  parent, depth, hash(ex), ee)
end

exp_size(node::Node) = exp_size(node.ex)
exp_size(ex::Expr) = sum(exp_size.(ex.args))
exp_size(ex::Symbol) = 1f0
exp_size(ex::Int) = 1f0

# get all availabel rules to apply
execute(ex, theory) = map(th->Postwalk(Metatheory.Chain([th]))(ex), theory)


# build a tree


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


function expand_node!(parent::Node, soltree::Dict{UInt64, Node}, heuristic::LikelihoodModel, open_list::PriorityQueue{Node,Float32, Base.Order.ForwardOrdering}, encodings_buffer::Dict{UInt64, MyNodes}, all_symbols::Vector{Symbol}, symbols_to_index::Dict{Symbol, Int64}, c_init::Vector{Float32}, h_init::Vector{Float32})::Nothing
    ex = parent.ex
    succesors = filter(r -> r[2] != ex, collect(enumerate(execute(ex, theory))))

    for (rule_index, new_exp) in succesors
        if !haskey(encodings_buffer, hash(new_exp))
            encodings_buffer[hash(new_exp)] = expression_encoder!(new_exp, all_symbols, symbols_to_index)
        end
        new_node = Node(new_exp, rule_index, parent.node_id, parent.depth + 1, encodings_buffer[hash(new_exp)])
        if push_to_tree!(soltree, new_node)
            push!(parent.children, new_node.node_id)
            enqueue!(open_list, new_node, heuristic(new_node.expression_encoding, c_init, h_init))
        end
    end
end

function prity_print(arguments)
  for (ind,i) in enumerate(arguments)
    println("================$ind==============")
    println("$i")
  end
end

function get_not_expanded_nodes1(expanded_node_depth, open_list, encodings_buffer)
    result = MyNodes[] 
    for (i, number) in open_list
        if i.depth == expanded_node_depth
            push!(result, encodings_buffer[i.ex]) 
        end
    end
    return result
end

function get_not_expanded_nodes(expanded_node::Node, soltree::Dict{UInt64, Node}, encodings_buffer::Dict{UInt64, MyNodes})
    result = MyNodes[] 
    for i in soltree[expanded_node.parent].children
        if i != expanded_node.node_id
            not_expanded_nodes = soltree[i].node_id
            push!(result, encodings_buffer[not_expanded_nodes])
        end
    end
    return result
end

function build_tree!(soltree::Dict{UInt64, Node}, heuristic::LikelihoodModel, open_list::PriorityQueue{Node,Float32, Base.Order.ForwardOrdering}, pc, optimizer, close_list::Set{UInt64}, encodings_buffer::Dict{UInt64, MyNodes}, all_symbols::Vector{Symbol}, symbols_to_index::Dict{Symbol, Int64}, c_init::Vector{Float32}, h_init::Vector{Float32}) 
    expanded_nodes = MyNodes[]
    not_expanded_nodes = Vector[]
    while length(open_list) > 0
        node = dequeue!(open_list)
        if node.depth >= 5
            println(length(open_list))
            continue
        end
        expand_node!(node, soltree, heuristic, open_list, encodings_buffer, all_symbols, symbols_to_index, c_init, h_init)
        #println(length(open_list))
        push!(close_list, node.node_id)
        if node.depth == 0
            continue
        end
        push!(expanded_nodes, encodings_buffer[node.node_id])
        #push!(not_expanded_nodes, get_not_expanded_nodes1(node.depth, open_list, encodings_buffer))
        push!(not_expanded_nodes, get_not_expanded_nodes(node, soltree, encodings_buffer))
        #heuristic(node.expression_encoding, c_init, h_init)
        loss(heuristic, expanded_nodes, not_expanded_nodes, c_init, h_init)
        #= 
        grad = gradient(pc) do 
          loss(heuristic, expanded_nodes, not_expanded_nodes)
        end
        Flux.update!(optimizer, pc, grad)
        =#
    end
end

function extract_rules_applied(node::Node, soltree::Dict{UInt64, Node}) 
    proof_vector = Vector()
    #while !isnothing(node.parent)
    while node.parent != node.node_id
        push!(proof_vector, node.rule_index)
        node = soltree[node.parent]
    end
    return reverse(proof_vector)
end

function extract_smallest_terminal_node(soltree::Dict{UInt64, Node}, close_list::Set{UInt64})
    #result = soltree[close_list[1]]
    result = nothing
    for node_hash in close_list
        if isnothing(result) || (exp_size(result.ex) >= exp_size(soltree[node_hash].ex) && result.depth > soltree[node_hash].depth)
            result = soltree[node_hash]
        end
    end
    return result
end

heuristic = LikelihoodModel(in_dim, out_dim, out_dim * 2)
c_init = zeros(Float32, out_dim)
h_init = zeros(Float32, out_dim)

#heuristic = exp_size
#
optimizer = ADAM()
depth = 50
#nodes_expression_built = Vector()
#terminal_nodes = Node[] 

pc = Flux.params(heuristic)
ex = data[4]
# Todo: fix issues with && and ||

#for (index, ex) in enumerate(data[1:10])
soltree = Dict{UInt64, Node}()
open_list = PriorityQueue{Node, Float32}()
close_list = Set{UInt64}()
encodings_buffer = Dict{UInt64, MyNodes}()
#encodings_buffer = Dict()
println("Initial expression: $ex")
encoded_ex = expression_encoder!(ex, all_symbols, symbols_to_index)
root = Node(ex, 0, hash(ex), 0, encoded_ex)
soltree[root.node_id] = root
#push!(open_list, root.node_id)
enqueue!(open_list, root, heuristic(root.expression_encoding))

build_tree!(soltree, heuristic, open_list, pc, optimizer, close_list, encodings_buffer, all_symbols, symbols_to_index, c_init, h_init)
println("Have successfuly finished bulding simplification tree!")

smallest_node = extract_smallest_terminal_node(soltree, close_list)
simplified_expression = smallest_node.ex
println("Simplified expression: $simplified_expression")

proof_vector = extract_rules_applied(smallest_node, soltree)
println("Proof vector: $proof_vector")
#end
