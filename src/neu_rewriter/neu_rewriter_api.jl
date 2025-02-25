mutable struct NeuRewriterEnv<:AbstractEnv
    initial_state
    current_state
    soltree
    close_list
    theory
    max_expansions
    current_expansions
    policy_model
    value_model
end


function NeuRewriterEnv(ex, value_model, policy_model; max_expansions=10)
    soltree = Dict{UInt64, Node}()
    close_list = Set{UInt64}()
    ex = intern!(ex)
    root = Node(ex, (), hash(ex), 0)
    soltree[root.node_id] = root
    return NeuRewriterEnv(root, root, soltree, close_list, theory, max_expansions, 0, policy_model, value_model)
end

function action_space(env::NeuRewriterEnv)
    subtrees = [(env.current_state.ex, [])]
    get_all_subtrees!(env.current_state.ex, subtrees)
    tree_action = []
    for (st, pos) in subtrees
        new_ex = [(intern!(r(st)), pos, ind, st) for (ind,r) in enumerate(env.theory)]
        # filter empty rewrites
        filtered_ex = filter(x->!isnothing(x[1]), new_ex)
        new_ex = map(x->(x..., my_rewrite!(env.current_state.ex, x[2], x[1])), filtered_ex)
        # filter repeated 
        filtered_ex = filter(x->!haskey(env.soltree, hash(x[5])), new_ex)
        append!(tree_action, filtered_ex)
    end
    return tree_action
end

state(env::NeuRewriterEnv) = env.current_state

state_space(env::NeuRewriterEnv) = env.soltree

reward(env::NeuRewriterEnv) = exp_size(env.initial_state.ex) - exp_size(env.current_state.ex)

is_terminated(env::NeuRewriterEnv) = env.current_expansions == env.max_expansions ? true : false

function reset!(env::NeuRewriterEnv)
    empty!(env.soltree)
    env.soltree[env.initial_state.node_id] = env.initial_state
    empty!(env.close_list)
    env.current_expansions = 0
end

function act!(env::NeuRewriterEnv, action)
    rewritten_subtree, pos, rule_id, _, new_ex = action
    next_state = Node(new_ex, (pos, rule_id), hash(new_ex), env.current_state.depth + 1)
    env.soltree[next_state.node_id] = next_state
    env.current_state = next_state
    env.current_expansions += 1
end