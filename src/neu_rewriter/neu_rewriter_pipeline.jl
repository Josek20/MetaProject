function my_rewrite!(ex::NodeID, pos, new_exp_part::NodeID)
    if isempty(pos)
        return new_exp_part
    end
    node = nc[ex]
    if pos[1] == 1
        new_part = my_rewrite!(node.left, pos[2:end], new_exp_part)
        node.left = new_part
    else        
        new_part = my_rewrite!(node.right, pos[2:end], new_exp_part)
        node.right = new_part
    end
    return get!(nc, node)
end


function push_to_tree1!(soltree, node)
    id = node.node_id
    if haskey(soltree, id)
        return false
    else
        soltree[id] = node
        return true
    end
end


function get_all_subtrees!(ex::Expr, pos, all_subtrees::Set)
    if !isa(ex, Expr)
        return
    end
    push!(all_subtrees, (ex, copy(pos)))
    # tmp = length(ex.args) > 2 ? 2 : 1
    for (ind,i) in enumerate(ex.args)
        push!(pos, ind)
        get_all_subtrees!(i, pos, all_subtrees)
        pop!(pos)
    end
end


function get_all_subtrees!(ex::NodeID, all_subtrees::Vector, parent=[])
    node = MyModule.nc[ex]
    for (ind,i) in enumerate([node.left, node.right])
        if !(MyModule.nc[i].iscall) && MyModule.nc[i].head ∉ [:&&, :||]
            continue
        end
        if i == MyModule.nullid
            continue
        end
        pos = vcat(parent, ind)
        push!(all_subtrees, (i, pos))
        get_all_subtrees!(i, all_subtrees, pos)
    end
end