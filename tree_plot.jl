include("tree_simplifier.jl")


#ex = test_data[900]
ex = :((((min(v0, 509) + 6) / 8) * 8 + (v1 * 516 + v2)) + 1 <= (((509 + 13) / 16) * 16 + (v1 * 516 + v2)) + 2)
soltree = Dict{UInt64, Node}()
open_list = PriorityQueue{Node, Float32}()
close_list = Set{UInt64}()
#encodings_buffer = Dict{UInt64, ExprEncoding}()
encodings_buffer = Dict{UInt64, ProductNode}()
println("Initial expression: $ex")
#encoded_ex = expression_encoder(ex, all_symbols, symbols_to_index)
encoded_ex = ex2mill(ex, symbols_to_index)
root = Node(ex, 0, hash(ex), 0, encoded_ex)
soltree[root.node_id] = root
#push!(open_list, root.node_id)
enqueue!(open_list, root, only(heuristic(root.expression_encoding)))

build_tree!(soltree, heuristic, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, 1000, 10)
println("Have successfuly finished bulding simplification tree!")

smallest_node = extract_smallest_terminal_node(soltree, close_list)
simplified_expression = smallest_node.ex
println("Simplified expression: $simplified_expression")

proof_vector, depth_dict, big_vector, hp, hn = extract_rules_applied(smallest_node, soltree)

println("Proof vector: $proof_vector")

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
        level 1/.style = {sibling distance=10cm},
        level 2/.style = {sibling distance=12cm},
        every node/.style = {rectangle, draw, fill=blue!20, minimum size=3mm}
    ]
"""

closing = """
\\end{tikzpicture}
\\end{document}
"""
function tree_traversal!(io, node, soltree)
    if isempty(node.children)
        println(io, "node{$(node.ex)}")
        return 
    end
    if node.depth == 0
        println(io, "\\node {$(node.ex)}")
    else
        println(io, "node {$(node.ex)}")
    end
    for id in node.children
        println(io, "child{")
        tree_traversal!(io, soltree[id], soltree)
        println(io, "}")
    end
end

function create_latex_tree3(io, root, soltree, smallest_node, preamble, closing, hspace=1, vspace=1)
    println(io, preamble)
    tmp = Dict()
    max_depth = max(map(x->x.depth, values(soltree))...)
    reversed_depth = reverse(collect(1:max_depth + 1))
    #println(io, "\\node {$(root.ex)}")
    tree_traversal!(io, root, soltree)
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

open(io -> create_latex_tree3(io, root, soltree,smallest_node, preamble2, closing), "my_tree.tex", "w")
