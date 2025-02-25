mutable struct EGraphEnv<:AbstractEnv
    initial_expr
    egraph
    action_sequence
    theory
    max_expansions
    current_expansions
    model
end

function EGraphEnv(ex::Union{Expr, NodeID}, model; max_expansions=2)
    action_sequence = []
    egraph = EGraph(ex)
    return EGraphEnv(ex, egraph, action_sequence, theory, max_expansions, 0, model)
end

# action_space(env::EGraphEnv) = env.theory
function action_space(env::EGraphEnv)
    for (ind, r) in enumerate(env.theory)
        for i in keys(env.egraph.classes)
            r.ematcher!(env.egraph, ind, i)
        end
    end
    actions = []
    for bindings in env.egraph.buffer
        rule_idx, id = bindings[0]
        direction = sign(rule_idx)
        rule_idx = abs(rule_idx)
        rule = env.theory[rule_idx]
        push!(actions, (bindings, rule, id, direction))
    end
    empty!(env.egraph.buffer)
    return actions
end

state(env::EGraphEnv) = env.egraph

state_space(env::EGraphEnv) = env.egraph.classes

reward(env::EGraphEnv) = exp_size(env.initial_expr) - exp_size(extract!(env.egraph, astsize))

is_terminated(env::EGraphEnv) = env.current_expansions == env.max_expansions ? true : false

function reset!(env::EGraphEnv)
    env.egraph = EGraph(env.initial_expr)
    empty!(env.action_sequence)
    env.current_expansions = 0
end

function act!(env::EGraphEnv, action)
    bindings, rule, id, direction = action
    Metatheory.EGraphs.apply_rule!(bindings, env.egraph, rule, id, direction)
    (l, r) = pop!(env.egraph.merges_buffer)
    Metatheory.EGraphs.merge!(env.egraph, l, r)
    push!(env.action_sequence, action)
    env.current_expansions += 1
end