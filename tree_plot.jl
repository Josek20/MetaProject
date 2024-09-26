# include("tree_simplifier.jl")
using Latexify
using MyModule
using MyModule.DataStructures
using BenchmarkTools



preamble1 = """
\\documentclass{standalone}
\\usepackage{tikz}
\\usepackage{verbatim}
\\usepackage{adjustbox}
\\usetikzlibrary{arrows,shapes}


\\begin{document}

\\tikzstyle{nonoptimalnode}=[circle,fill=black!25,minimum size=20pt,inner sep=0pt]
\\tikzstyle{optimalnode} = [nonoptimalnode, fill=red!24]
\\tikzstyle{nonoptimaledge} = [draw,->,thin]
\\tikzstyle{optimaledge} = [draw,thick,->]
\\tikzstyle{hiddennode} = [white]
\\tikzstyle{hiddenedge} = [white]
\\begin{tikzpicture}
"""

preamble = """
\\documentclass{standalone}
\\usepackage{tikz}
\\usetikzlibrary{graphs, graphdrawing, shapes, arrows.meta}

\\begin{document}

\\begin{tikzpicture}[every node/.style={rectangle, draw, fill=blue!20, inner sep=0pt, minimum size=6mm}, node distance=1.5cm]
"""

preamble2 = """
\\documentclass{standalone}
\\usepackage{tikz}
\\usetikzlibrary{trees}

\\begin{document}

\\begin{tikzpicture}
    [
        edge from parent/.style = {draw, -latex},
        level 1/.style = {sibling distance=20cm},
        level 2/.style = {sibling distance=10cm},
        every node/.style = {rectangle, draw, minimum size=3mm}
    ]
    \\tikzstyle{coloredNode} = [fill=blue!20]
    \\tikzstyle{rootNode} = [fill=green!20]
"""

closing = """
\\end{tikzpicture}
\\end{document}
"""
function tree_traversal!(io, node, soltree, expansion_history, proof)
    if isempty(node.children)
        if haskey(expansion_history, node.node_id)
            step_number, prob = expansion_history[node.node_id]
            prob = round(Float64(prob), digits=2)
            step_number = Integer(step_number)
        end
        if node.node_id in proof
            println(io, "node[coloredNode]{$(latexify(node.ex))}")
        else
            println(io, "node{$(latexify(node.ex))}")
        end
        if haskey(expansion_history, node.node_id)
            println(io, "edge from parent node[midway, left] {$((step_number, prob))}")
        end
        return 
    end
    step_number, prob = expansion_history[node.node_id]
    prob = round(Float64(prob), digits=2)
    step_number = Integer(step_number)
    if node.depth == 0
        println(io, "\\node[rootNode] {$(latexify(node.ex))}")
    elseif node.node_id in proof
        println(io, "node[coloredNode] {$(latexify(node.ex))}")
      #println(io, "edge from parent node[midway, left] {$(id, prob)}")
    else
        println(io, "node{$(latexify(node.ex))}")
        #println(io, "edge from parent node[midway, left] {$((step_number, prob))}")
    end
    for id in node.children
        println(io, "child{")
        tree_traversal!(io, soltree[id], soltree, expansion_history, proof)
        println(io, "}")
    end
    if node.depth != 0
        println(io, "edge from parent node[midway, left] {$((step_number, prob))}")
    end
end

function create_latex_tree3(io, root, soltree, smallest_node, preamble, closing, expansion_history, proof, hspace=1, vspace=1)
    println(io, preamble)
    tmp = Dict()
    max_depth = max(map(x->x.depth, values(soltree))...)
    reversed_depth = reverse(collect(1:max_depth + 1))
    #println(io, "\\node {$(root.ex)}")
    tree_traversal!(io, root, soltree, expansion_history, proof)
    println(io, ";")
    println(io, closing)
end

function create_latex_tree2(io, root, soltree, smallest_node, preamble, closing, hspace=1, vspace=1)
    println(io, preamble)
    tmp = Dict()
    max_depth = max(map(x->x.depth, values(soltree))...)
    reversed_depth = reverse(collect(1:max_depth + 1))
    for (k,v) in soltree
        if haskey(tmp, v.depth)
            tmp[v.depth] += 1
        else
            tmp[v.depth] = 1
        end
        println(io, "\\node ($(k)) at ($((tmp[v.depth] - 1) * 10), $(reversed_depth[v.depth + 1])) {$(v.ex)};")
        #println(io, "\\draw ($(k)) -- ($(soltree[v.parent].node_id));")
    end

    for (k,v) in soltree
        println(io, "\\draw ($(k)) -- ($(soltree[v.parent].node_id));")
    end
    println(io, closing)
end

# function plot_tree()
# ex = train_data[3]
# ex = m
ex = :(v2 <= v2 && ((v0 + v1) + 120) - 1 <= (v0 + v1) + 119)
# ex = :(min((((v0 - v1) + 119) / 8) * 8 + v1, v0 + 112) <= v0 + 112)
ex = :((v0 + v1) + 119 <= min(120 + (v0 + v1), v2) && min(((((v0 + v1) - v2) + 127) / 8) * 8 + v2, (v0 + v1) + 120) - 1 <= ((((v0 + v1) - v2) + 134) / 16) * 16 + v2)
ex = train_data[1]
# ex = myex
# ex = :((v0 + v1) * 119 + (v3 + v7) <= (v0 + v1) * 119 + ((v2 * 30 + ((v3 * 4 + v4) + v5)) + v7))
# ex =  :((v0 + v1) * 119 + (v3 + v7) <= (v0 + v1) * 119 + (((v3 * (4 + v2 / (30 / v3)) + v5) + v4) + v7))
# encoded_ex = MyModule.ex2mill(ex, symbols_to_index, all_symbols, variable_names)
encoded_ex = MyModule.single_fast_ex2mill(ex, MyModule.sym_enc)
root = MyModule.Node(ex, (0,0), hash(ex), 0, encoded_ex)

soltree = Dict{UInt64, MyModule.Node}()
open_list = PriorityQueue{MyModule.Node, Float32}()
close_list = Set{UInt64}()
expansion_history = Dict{UInt64, Vector}()
#encodings_buffer = Dict{UInt64, ExprEncoding}()
encodings_buffer = Dict{UInt64, ProductNode}()
println("Initial expression: $ex")
#encoded_ex = expression_encoder(ex, all_symbols, symbols_to_index)
soltree[root.node_id] = root
#push!(open_list, root.node_id)
enqueue!(open_list, root, only(heuristic(root.expression_encoding)))

reached_goal = MyModule.build_tree!(soltree, heuristic, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, max_steps, max_depth, expansion_history, theory, variable_names)
# bmark1 = @benchmark reached_goal = MyModule.build_tree!(soltree, heuristic, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, 1000, 25, expansion_history, theory, variable_names)
# bmark1 = @time reached_goal = MyModule.build_tree!(soltree, heuristic, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, 1000, 25, expansion_history, theory, variable_names)
# soltree = Dict{UInt64, MyModule.Node}()
# open_list = PriorityQueue{MyModule.Node, Float32}()
# close_list = Set{UInt64}()
# expansion_history = Dict{UInt64, Vector}()
# #encodings_buffer = Dict{UInt64, ExprEncoding}()
# encodings_buffer = Dict{UInt64, ProductNode}()
# println("Initial expression: $ex")
# #encoded_ex = expression_encoder(ex, all_symbols, symbols_to_index)
# soltree[root.node_id] = root
# #push!(open_list, root.node_id)
# enqueue!(open_list, root, only(heuristic(root.expression_encoding)))
# # bmark2 = @benchmark reached_goal2 = MyModule.build_tree_new!(soltree, heuristic, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, 1000, 25, expansion_history, theory, variable_names)
# bmark2 = @time reached_goal2 = MyModule.build_tree_new!(soltree, heuristic, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, 1000, 25, expansion_history, theory, variable_names)
# println(bmark1)
# println(bmark2)
println("Have successfuly finished bulding simplification tree!")
smallest_node = MyModule.extract_smallest_terminal_node(soltree, close_list)
# # for (ind, (i, cof)) in enumerate(open_list)
# #     expansion_history[i.node_id] = [length(expansion_history) + ind - 1, cof]
# # end
# # @show expansion_history[smallest_node.node_id]
# simplified_expression = smallest_node.ex
# println("Simplified expression: $simplified_expression")

proof_vector, depth_dict, big_vector, hp, hn, node_proof_vector = MyModule.extract_rules_applied(smallest_node, soltree)

# println("Proof vector: $proof_vector")

# for (ind, (i, cof)) in enumerate(open_list)
#     expansion_history[i.node_id] = [length(expansion_history) + ind - 1, cof]
# end

# test_open_list = PriorityQueue{MyModule.Node, Float32}()
# test_expansion_history = Dict{UInt64, Vector}()
# for (k, n) in soltree
#     enqueue!(test_open_list, n, only(heuristic(n.expression_encoding)))
# end

# for (k,v) in depth_dict
#     tmp = heuristic.(v)
#     println("Depth $k: $tmp")
# end
open(io -> create_latex_tree3(io, root, soltree, smallest_node, preamble2, closing, expansion_history, node_proof_vector), "my_tree.tex", "w")
