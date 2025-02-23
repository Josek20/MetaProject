struct AlphaZeroEnv<:MyEnv
    initial_state
    open_list
    soltree
    close_list
    rolled_out_list
    nodes_stats
    theory
    max_expansions
    current_expansions
end

action_space(env::AlphaZeroEnv) = env.open_list

state(env::AlphaZeroEnv) = first(env.open_list)

state_space(env::AlphaZeroEnv) = values(env.soltree)

function reward(env::AlphaZeroEnv)
    current_state = first(env.open_list)
    return nodes_stats[current_state.node_id].reward
end

is_terminated(env::AlphaZeroEnv) = env.current_expansions == env.max_expansions ? true : false

function reset!(env::AlphaZeroEnv)
    empty!(env.soltree)
    soltree[env.initial_state.node_id] = env.initial_state
    empty!(env.open_list)
    enqueue!(open_list, env.initial_state, 0)
    empty!(env.close_list)
    empty!(env.rolled_out_list)
    empty!(env.nodes_stats)
    env.current_expansions = 0
end

function act!(env::AlphaZeroEnv, action)
    # Todo
end