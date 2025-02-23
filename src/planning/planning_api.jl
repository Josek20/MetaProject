struct PlanningEnv<:MyEnv
    initial_state
    soltree
    open_list
    close_list
    theory
    max_expansions
    current_expansions
    model
end

function PlanningEnv(ex, model; max_expansions=2)
    soltree = Dict{UInt64, Node}()
    open_list = PriorityQueue{Node, Float32}()
    close_list = Set{UInt64}()
    ex = intern!(ex)
    root = Node(ex, (), hash(ex), 0)
    soltree[root.node_id] = root
    enqueue!(open_list)
    return PlanningEnv(root, soltree, open_list, close_list, theory, max_expansions, 0, model)
end

action_space(env::PlanningEnv) = env.open_list

state(env::PlanningEnv) = env.soltree

state_space(env::PlanningEnv) = env.soltree

function reward(env::PlanningEnv)
    current_state, _ = first(env.open_list)
    return exp_size(env.initial_state.ex) - exp_size(current_state.ex)
end

is_terminated(env::PlanningEnv) = env.current_expansions == env.max_expansions ? true : false

function reset!(env::PlanningEnv)
    empty!(env.soltree)
    soltree[env.initial_state.node_id] = env.initial_state
    empty!(env.open_list)
    enqueue!(open_list, env.initial_state, 0)
    empty!(env.close_list)
    env.current_expansions = 0
end

function act!(env::PlanningEnv, action)
    current_node = action
    push!(env.close_list, current_node.node_id)
    expand_node!(current_node, env.soltree, env.open_list, env.model)
    env.current_expansions += 1
end