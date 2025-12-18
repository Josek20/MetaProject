mutable struct NeuralRewriterEnv{SI, SC, T, D} <: AbstractEnvironment
    s_init::SI
    s_current::SC
    t::T
    is_done::D
end
function NeuralRewriterEnv(ex::Expr)
    inex = intern!(ex)
    # inex = ex
    return NeuralRewriterEnv(inex, inex, 1, false)
end
state(env::NeuralRewriterEnv) = env.s_current
function reset!(env::NeuralRewriterEnv)
    env.s_current = env.s_init
    env.t = 1
    # return env
end
MyModule.@my_cache MyModule.LRU(maxsize=10_000) function get_all_subtree(x::NodeID, subtrees=NodeID[])
    x == MyModule.nullid && return subtrees
    only_node = MyModule.nc[x]
    only_node.left == MyModule.nullid && only_node.right == MyModule.nullid && return subtrees
    push!(subtrees, x) 
    get_all_subtree(only_node.left, subtrees)
    get_all_subtree(only_node.right, subtrees)
    return subtrees
end
function get_all_subtree_no_cache(x::NodeID, subtrees=NodeID[])
    x == MyModule.nullid && return subtrees
    only_node = MyModule.nc[x]
    only_node.left == MyModule.nullid && only_node.right == MyModule.nullid && return subtrees
    push!(subtrees, x) 
    get_all_subtree_no_cache(only_node.left, subtrees)
    get_all_subtree_no_cache(only_node.right, subtrees)
    return subtrees
end
function my_rewrite!(ex::NodeID, pos, new_exp_part::NodeID)
    if isempty(pos)
        return new_exp_part
    end
    node = MyModule.nc[ex]

    if pos[1] == 1
        new_part = my_rewrite!(node.left, pos[2:end], new_exp_part)
        # node.left = new_part
        new_node = OnlyNode(node.head, node.iscall, node.v, new_part, node.right)
    else        
        new_part = my_rewrite!(node.right, pos[2:end], new_exp_part)
        # node.right = new_part
        new_node = OnlyNode(node.head, node.iscall, node.v, node.left, new_part)
    end
    return get!(MyModule.nc, new_node)
end
function action_space(env::NeuralRewriterEnv)
    env.t += 1
    subtrees = get_all_subtree_no_cache(env.s_current)
    actions_for_each = map(subtrees) do subtree 
        # can't use all_expand directly because of the main subtree will go and explore everything
        rewritten_subtree, action_index = MyModule.all_expand(subtree, theory)
        tmp = (filter(x->x!=subtree, rewritten_subtree), action_index)
        @assert length(tmp[1]) == length(tmp[2])
        my_rewrite!(env.s_current, pos, new)
        tmp
    end
    all_possible_rewriting, all_possible_index = MyModule.all_expand(env.s_current, theory)
    all_possible_rewriting = filter(x->x!=env.s_current, all_possible_rewriting)
    @assert length(all_possible_rewriting) == length(all_possible_index)
    return subtrees
end
function reward(env::NeuralRewriterEnv, a, s)
    size_current = exp_size(a)
    size_init = exp_size(env.s_init)
    # size_init = exp_size(s)
    return size_init - size_current
end
function reward1(env::NeuralRewriterEnv, a, s)
    size_current = exp_size(a)
    # size_init = exp_size(s)
    return -size_current
end
function act!(env::NeuralRewriterEnv, a)
    env.s_current = a
end
is_terminal(env::NeuralRewriterEnv) = env.is_done