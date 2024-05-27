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
end

mutable struct Node{E, P}
    ex::E
    rule_index::Int
    children::Vector{UInt64}
    parent::P
    depth::Int
    node_id::UInt64
    expression_encoding::Union{MyNodes, Nothing}
end

Node(ex::Int, rule_index, parent, depth, ee) = Node(ex, rule_index, UInt64[],  parent, depth, hash(ex), ee)
function Node(ex::Expr, rule_index, parent, depth, ee)
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


function expand_node!(parent::Node, soltree::Dict, heuristic::Union{LikelihoodModel, Function}, open_list::PriorityQueue, encodings_buffer::Dict) 
    ex = parent.ex
    succesors = filter(r -> r[2] != ex, collect(enumerate(execute(ex, theory))))

    for (rule_index, new_exp) in succesors
        if !haskey(encodings_buffer, new_exp)
            encodings_buffer[new_exp] = expression_encoder!(new_exp)
        end
        new_node = Node(new_exp, rule_index, parent, parent.depth + 1, encodings_buffer[new_exp])
        push!(parent.children, new_node.node_id)
        if push_to_tree!(soltree, new_node)
            enqueue!(open_list, new_node, heuristic(new_node.expression_encoding))
        end
    end
end

function prity_print(arguments)
  for (ind,i) in enumerate(arguments)
    println("================$ind==============")
    println("$i")
  end
end

function get_nodes_in_proof(expanded_node::Node)
  buffer = Union{Expr, UInt64}[]
  encodings_buffer = MyNodes[]
  while true
    push!(buffer, expanded_node.node_id)
    push!(encodings_buffer, expanded_node.expression_encoding)
    if expanded_node.depth == 0    
        break
    end
    expanded_node = expanded_node.parent 
  end
  return buffer, encodings_buffer
end

function build_tree!(soltree::Dict, heuristic::Union{LikelihoodModel, Function}, open_list::PriorityQueue, pc, optimizer, close_list::Set{UInt64}, encodings_buffer::Dict) 
    while length(open_list) > 0
        node = dequeue!(open_list)
        if node.depth >= 50
            continue
        end
        expand_node!(node, soltree, heuristic, open_list, encodings_buffer)
        #println(length(open_list))
        push!(close_list, node.node_id)
        in_proof, encodings_in_proof = length(open_list) != 0 ? get_nodes_in_proof(peek(open_list)[1]) : get_nodes_in_proof(node)
        encodings_not_in_proof = map(node->node.expression_encoding, filter(node -> node.node_id ∉ in_proof, collect(values(soltree))))
        #println("===========================================")
        grad = gradient(pc) do 
            loss(heuristic, encodings_in_proof, encodings_not_in_proof)
        end
        Flux.update!(optimizer, pc, grad)
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

function extract_smallest_terminal_node(soltree::Dict{UInt64, Node}, close_list::Set{UInt64})
    #result = soltree[close_list[1]]
    result = nothing
    for node_hash in close_list
        if isnothing(result) || exp_size(result.ex) > exp_size(soltree[node_hash].ex)
            result = soltree[node_hash]
        end
    end
    return result
end

heuristic = LikelihoodModel(in_dim, out_dim, out_dim * 2)
#heuristic = exp_size
#
optimizer = ADAM()
depth = 50
#nodes_expression_built = Vector()
#terminal_nodes = Node[] 

pc = Flux.params(heuristic)


for (index, ex) in enumerate(data[2:2])
  soltree = Dict{UInt64, Node}()
  open_list = PriorityQueue{Node, Float32}()
  close_list = Set{UInt64}()
  encodings_buffer = Dict{Expr, MyNodes}()
  println("Initial expression: $ex")
  encoded_ex = expression_encoder!(ex)
  root = Node(ex, 0, nothing, 0, encoded_ex)
  soltree[root.node_id] = root
  #push!(open_list, root.node_id)
  enqueue!(open_list, root, heuristic(root.expression_encoding))

  build_tree!(soltree, heuristic, open_list, pc, optimizer, close_list, encodings_buffer)
  println("Have successfuly finished bulding simplification tree!")

  smallest_node = extract_smallest_terminal_node(soltree, close_list)
  simplified_expression = smallest_node.ex
  println("Simplified expression: $simplified_expression")
  
  proof_vector = extract_rules_applied(smallest_node)
  println("Proof vector: $proof_vector")
end

