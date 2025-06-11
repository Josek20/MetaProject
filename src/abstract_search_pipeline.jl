generate_children(node::AbstractNode, tree::AbstractDict)
add_child!()
function expand_node!(tree::AbstractTree, state::AbstractSearchState, node::AbstractNode)
    if is_expandable(node)
        children = generate_children(node, tree)
        for child in children
            add_child!(tree, node, child)
            update_state!(state, child)
        end
    end
end

is_finished(tree, state) = true
function build_tree!(tree::AbstractTree, state::AbstractSearchState, root::AbstractNode)
    current_node = root
    while !is_finished(tree, state)
        expand_node!(tree, state, current_node)
        current_node = next_node(tree, state)
    end
end


function initialize_tree(data)
    state = AbstractSearchState()
    tree = AbstractTree(data, state)
    root = AbstractNode(data, state)
    return root, tree, state
end


function tree_search_pipline(data, model)
    root, tree, state = initialize_tree(data)
    build_tree!(tree, state, root)
    return tree, state, root
end