mutable struct AlphaZeroEnv<:AbstractEnv
    initial_state
    open_list
    soltree
    close_list
    rolled_out_list
    nodes_stats
    theory
    max_expansions
    current_expansions
    policy_model
    value_model
end


function AlphaZeroEnv(ex, policy_model, value_model;max_expansions=10)
    soltree = Dict{UInt64, Node}()
    open_list = PriorityQueue{Node, Float32}(Base.Reverse)
    close_list = Set{UInt64}()
    rolled_out_list = Set{UInt64}()
    nodes_stats = Dict{UInt64, Vector}()
    ex = intern!(ex)
    root = Node(ex, (), hash(ex), 0)
    soltree[root.node_id] = root
    nodes_stats[root.node_id] = [0f0,0f0,0f0]
    push!(close_list, root.node_id)
    expand_node!(root, soltree, open_list, value_model)
    rollout_nodes = keys(open_list)
    for i in rollout_nodes
        push!(rolled_out_list, i.node_id)
    end
    
    trees = [rollout(gs.ex, policy_model, max_expansions=30) for gs in rollout_nodes]
    # trees = [initialize_rollout_tree_search(gs.ex, policy_model, max_expansions=20) for gs in rollout_nodes]
            
    for (rt, (st, sm, cl)) in zip(rollout_nodes, trees)
        reward = exp_size(root.ex) - exp_size(sm.ex)
        # nodes_stats[rt.node_id] = (;visited_count=1,value_sum=value_model(sm.ex),reward=reward)
        nodes_stats[rt.node_id] = [1f0, only(value_model(sm.ex)), reward]
        backpropagate_stats!(rt, nodes_stats, soltree)
    end
    for g_state in rollout_nodes
        v = nodes_stats[g_state.node_id][2]
        n = nodes_stats[g_state.node_id][1]
        N = nodes_stats[g_state.parent][1]
        uct = v ÷ n + 1.2 * √log(N) ÷ n
        open_list[g_state] = uct
    end
    push!(rolled_out_list, root.node_id)
    return AlphaZeroEnv(root, open_list, soltree, close_list, rolled_out_list, nodes_stats, theory, max_expansions, 0, policy_model, value_model)
end


function action_space(env::AlphaZeroEnv)
    actions = keys(env.open_list)
    # filtered_game_states = filter(x->x.node_id ∉ env.rolled_out, actions)
    return actions
end

state(env::AlphaZeroEnv) = first(env.open_list)

state_space(env::AlphaZeroEnv) = values(env.soltree)

function reward(env::AlphaZeroEnv)
    current_state = first(env.open_list)
    return nodes_stats[current_state.node_id].reward
end

is_terminated(env::AlphaZeroEnv) = env.current_expansions == env.max_expansions ? true : false

function reset!(env::AlphaZeroEnv)
    empty!(env.soltree)
    env.soltree[env.initial_state.node_id] = env.initial_state
    empty!(env.open_list)
    enqueue!(env.open_list, env.initial_state, 0)
    empty!(env.close_list)
    empty!(env.rolled_out_list)
    empty!(env.nodes_stats)
    env.current_expansions = 0
    env.nodes_stats[env.initial_state.node_id] = [0f0,0f0,0f0]
    push!(env.close_list, env.initial_state.node_id)
    expand_node!(env.initial_state, env.soltree, env.open_list, env.value_model)
    rollout_nodes = keys(env.open_list)
    for i in rollout_nodes
        push!(env.rolled_out_list, i.node_id)
    end
    
    trees = [rollout(gs.ex, env.policy_model, max_expansions=30) for gs in rollout_nodes]
            
    for (rt, (st, sm, cl)) in zip(rollout_nodes, trees)
        reward = exp_size(env.initial_state.ex) - exp_size(sm.ex)
        # nodes_stats[rt.node_id] = (;visited_count=1,value_sum=value_model(sm.ex),reward=reward)
        env.nodes_stats[rt.node_id] = [1f0, only(value_model(sm.ex)), reward]
        backpropagate_stats!(rt, env.nodes_stats, env.soltree)
    end
    for g_state in rollout_nodes
        v = env.nodes_stats[g_state.node_id][2]
        n = env.nodes_stats[g_state.node_id][1]
        N = env.nodes_stats[g_state.parent][1]
        uct = v ÷ n + 1.2 * √log(N) ÷ n
        env.open_list[g_state] = uct
    end
    push!(env.rolled_out_list, env.initial_state.node_id)
end

function act!(env::AlphaZeroEnv, action)
    monte_carlo_expand!(action, env.soltree, env.open_list)
    new_action = only(filter(x->x ∉ env.rolled_out_list, keys(env.open_list)))
    push!(env.rolled_out_list, new_action.node_id)
    soltree, smallest_node, _ = rollout(new_action, env.policy_model; max_expansions=20)
    reward = env.initial_size - exp_size(smallest_node.ex)
    env.nodes_stats[smallest_node.node_id] = [1f0, only(value_model(smallest_node.ex)), reward]
    backpropagate_stats!(new_action, env.nodes_stats, env.soltree)
    v = env.nodes_stats[smallest_node.node_id][2]
    n = env.nodes_stats[smallest_node.node_id][1]
    N = env.nodes_stats[smallest_node.parent][1]
    uct = v ÷ n + 1.2 * √log(N) ÷ n
    env.open_list[smallest_node] = uct
    env.current_expansions += 1
end