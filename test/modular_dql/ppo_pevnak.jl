using CSV
using DataFrames
using Optimisers
using Distributed
println("Number of workers: ", nworkers())
println("Threads per worker: ", Threads.nthreads())
using Random
@everywhere begin
    using Statistics
    using MyModule
    using MyModule.Flux
    using MyModule.Mill
    using MyModule: all_expand, exp_size, Node, NodeID, intern!, expr, DeduplicatingNode, AbstractModel
    using StatsBase
    using TimerOutputs
    abstract type AbstractEnvironment end
    include("my_env.jl")
end
using Serialization


function get_data(;path="train.json")
    train_data_path = "./data/neural_rewrter/$(path)"
    train_data = load_data(train_data_path)[1:1_000]
    train_data = filter(x->!occursin("select", x[1]), train_data)
    train_data = preprosses_data_to_expressions(train_data)
    sorted_data = sort(train_data, by=x->MyModule.exp_size(x))
    data = sorted_data
    return Vector{Expr}(data)
end

data = get_data()

hidden_size=64
input_size = 64

function ffnn(idim, hidden_size, layers)
    layers == 1 && return Dense(idim, hidden_size, Flux.gelu)
    layers == 2 && return Chain(Dense(idim, hidden_size, Flux.gelu), Dense(hidden_size, hidden_size, Flux.gelu))
end

head_model = ProductModel(
    (;head = ffnn(length(new_all_symbols), hidden_size, 1),
    args = ffnn(hidden_size, hidden_size, 1),  
        ),
    ffnn(2*hidden_size, hidden_size, 1)
    )

args_model = ProductModel(
    (;args = ffnn(hidden_size, hidden_size, 1),  
    position = Dense(2,hidden_size),  
        ),
    ffnn(2*hidden_size, hidden_size, 1)
    )


model = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, 1))
    );
target_model = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, 1))
    );
target_model = deepcopy(model)

value_model = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, 1))
    );

@everywhere struct OnePPOSample
    old_policy::Vector{Float32}
    A::Float32
    G::Float32
    state::Int64
    action::Int64
    actions::Vector{Int64}
end

@everywhere struct PPOMinibatch{X,T}
    x::X
    q::Vector{OnePPOSample}
    ϵ::T
end


Base.show(io::IO, mb::PPOMinibatch) = print(io, "PPOMinibatch (→: $(length(mb.q)))")

function critic_loss(critic, mb::PPOMinibatch)
    o = vec(MyModule.heuristic(critic, mb.x))
    mean(Flux.mse(maximum(o[i] for i in sa.actions), sa.G) for sa in mb.q)
    # mean(Flux.mse(o[sa.state], sa.G) for sa in mb.q)
end

function _actor_loss(log_newπ_all, log_oldπ_all, A, ϵ)
    # Compute for the first action (which is the one taken)
    log_probs_new = logsoftmax(log_newπ_all)
    log_probs_old = logsoftmax(log_oldπ_all)
    r = exp(log_probs_new[1] - log_probs_old[1])  # ratio for taken action
    # @show r, A
    pg_loss1 = A * r
    pg_loss2 = A * clamp(r, 1 - ϵ, 1 + ϵ)
    return -min(pg_loss1, pg_loss2)
end

function actor_loss(actor, target_model, mb::PPOMinibatch)
    o = vec(MyModule.heuristic(actor, mb.x))
    o_old = vec(MyModule.heuristic(target_model, mb.x))
    #mean(_actor_loss(o[sa.actions], sa.old_policy, sa.A, mb.ϵ) for sa in mb.q)
    mean(_actor_loss(o[sa.actions], o_old[sa.actions], sa.A, mb.ϵ) for sa in mb.q)
end

@everywhere function get_solutions_traj_actions!(node::Node, soltree, targets)
    if node.depth == 0
        return
    end
    current_node_id = node.parent
    get_full_tree = UInt64[]
    possible_actions = UInt64[]
    for d in soltree[node.parent].depth:-1:1
        tmp = current_node_id
        current_node_id = soltree[current_node_id].parent
        tmp = filter(x->tmp != x, soltree[current_node_id].children)
        append!(possible_actions, tmp)
        append!(get_full_tree, soltree[current_node_id].children)
    end
    push!(get_full_tree, current_node_id)
    # @show length(possible_actions)
    # @show length(get_full_tree)
    if isa(targets, Dict)
        targets[get_full_tree] = Dict(i=>0 for i in possible_actions)
    else
        push!(targets, (node.parent, possible_actions, get_full_tree))
    end
    get_solutions_traj_actions!(soltree[node.parent], soltree, targets)
end


@everywhere function get_one_batch(env, policy, value_model, target_model; gamma=0.9,max_depth=100, max_steps=100, lambda=0.95)
    soltree, smallest_node, root, tree_history = MyModule.initialize_policy_tree_search(env, policy, max_expansions=max_steps, max_depth=max_depth, gamma=gamma)
    
    targets = []
    get_solutions_traj_actions!(smallest_node, soltree, targets)
    inputs_vec = []
    value_inputs_vec = []
    rews = Float32[]
    v_returns = Float32[]
    softmax_ids = []
    selected_ids = []
    gae = 0
    q = OnePPOSample[]
    #empty!(MyModule.memoize_cache(MyModule.general_expr_cached_inference))
    #empty!(MyModule.memoize_cache(MyModule.general_leaf_cached_inference))
    for (ind, (pos_action, possible_actions, tree)) in enumerate(targets)
        if ind == 1
            rew = minimum(x->exp_size(soltree[x].ex), tree) - exp_size(smallest_node.ex)
        else
            rew = minimum(x->exp_size(soltree[x].ex), tree) - minimum(x->exp_size(soltree[x].ex), targets[ind - 1][3])
        end
        current_values = map(x->only(value_model(soltree[x].ex)), tree)
        if ind == 1
            δ = rew - maximum(current_values)
        else
            next_values = map(x->only(value_model(soltree[x].ex)), targets[ind-1][3])
            δ = rew + gamma * maximum(next_values) - maximum(current_values)
        end
        gae = δ + gamma * lambda * gae
        #Aₜ = δ
        Aₜ = gae
        G = gae + maximum(current_values)
            
        #G = δ
        push!(rews, Aₜ)
        push!(v_returns, G)
        
        if ind == 1
            append!(inputs_vec, vcat(pos_action, possible_actions))
            push!(softmax_ids, collect(1:length(possible_actions) + 1))
            push!(selected_ids, 1)
        else
            push!(softmax_ids, [])
            if pos_action in inputs_vec
                tmp_ind = findfirst(==(pos_action), inputs_vec)
                push!(softmax_ids[end], tmp_ind)
                push!(selected_ids, 1)
            else
                push!(inputs_vec, pos_action)
                push!(softmax_ids[end], length(inputs_vec))
                push!(selected_ids, 1)
            end
            for (ind,a) in enumerate(possible_actions)
                if a in inputs_vec
                    tmp_ind = findfirst(==(a), inputs_vec)
                    push!(softmax_ids[end], tmp_ind)
                else
                    push!(inputs_vec, a)
                    push!(softmax_ids[end], length(inputs_vec))
                end
            end
        end
        #empty!(MyModule.memoize_cache(MyModule.general_expr_cached_inference))
        #empty!(MyModule.memoize_cache(MyModule.general_leaf_cached_inference))
        #push!(q, OnePPOSample([only(target_model(soltree[x].ex)) for x in vcat(pos_action, possible_actions)], Aₜ, G, -1, 1, softmax_ids[end]))
        # push!(q, OnePPOSample(Float32[], Aₜ, G, -1, 1, softmax_ids[end]))
    end
    if length(rews) > 1
        sd = std(rews)
        rews = sd < 1e-6 ? (rews .- mean(rews)) : (rews .- mean(rews)) ./ sd
    end
    @show rews
    for (A, G, ids) in zip(rews, v_returns, softmax_ids)
        push!(q, OnePPOSample(Float32[], A, G, -1, 1, ids))
    end
    actor_inputs = [expr(MyModule.nc, soltree[i].ex) for i in inputs_vec]
    actor_inputs = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(actor_inputs, sym_enc))

    return PPOMinibatch(actor_inputs, q, 0.2)
end
@everywhere function get_one_batch_no_solution(env, policy, value_model; gamma=0.9, max_depth=100, max_steps=100, lambda=0.95)
    soltree, smallest_node, root, tree_history = MyModule.initialize_policy_tree_search(env, policy, max_expansions=max_steps, max_depth=max_depth, gamma=gamma)
    inputs_vec = []
    value_inputs_vec = []
    softmax_ids = []
    selected_ids = []
    rews = Float32[]
    v_returns = Float32[]
    q = OnePPOSample[]
    gae = 0
    reversed_tree_history = reverse(tree_history)
    for (ind, (tree_node_ids, possible_actions, selected_action_index)) in enumerate(reversed_tree_history)
        if ind == 1
            continue
        end
        rew = minimum(x->exp_size(soltree[x].ex), tree_node_ids) - minimum(x->exp_size(soltree[x].ex), reversed_tree_history[ind - 1][1])

        current_values = map(x->only(value_model(soltree[x].ex)), tree_node_ids)

        next_values = map(x->only(value_model(soltree[x].ex)), reversed_tree_history[ind - 1][1])
        δ = rew + gamma * maximum(next_values) - maximum(current_values)
        gae = δ + gamma * lambda * gae
        Aₜ = gae
        G = gae + maximum(current_values)
        push!(value_inputs_vec, tree_node_ids[argmax(current_values)])
        # @show Gₜ
        if ind == 2
            append!(inputs_vec, possible_actions)
            push!(softmax_ids, collect(1:length(possible_actions)))
            push!(selected_ids, selected_action_index)
        else
            push!(softmax_ids, [])
            for (ind,a) in enumerate(possible_actions)
                if ind == selected_action_index && a in inputs_vec
                    tmp_ind = findfirst(==(a), inputs_vec)
                    push!(softmax_ids[end], tmp_ind)
                    push!(selected_ids, selected_action_index)
                elseif ind == selected_action_index && !(a in inputs_vec)
                    push!(inputs_vec, a)
                    push!(softmax_ids[end], length(inputs_vec))
                    push!(selected_ids, selected_action_index)
                elseif ind != selected_action_index && !(a in inputs_vec)
                    push!(inputs_vec, a)
                    push!(softmax_ids[end], length(inputs_vec))
                elseif ind != selected_action_index && a in inputs_vec
                    tmp_ind = findfirst(==(a), inputs_vec)
                    push!(softmax_ids[end], tmp_ind)
                end
            end
        end
        push!(rews, Aₜ)
        push!(v_returns, G)
    end
    # @timeit TO "get nodes for inputs" begin
    #     input_values = [expr(MyModule.nc, soltree[i].ex) for i in inputs_vec]
    #     input_values = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(input_values, sym_enc))
    # end
    # @assert length(rews) == length(softmax_ids)
    # v_inputs = [expr(MyModule.nc, soltree[pa[sid]].ex) for (_,pa,sid) in reversed_tree_history]
    # v_inputs = [expr(MyModule.nc, soltree[i].ex) for i in value_inputs_vec]
    # v_inputs = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(v_inputs, sym_enc))
    # return (;rewards=rews, inputs=(input_values, v_inputs), smallest_node=smallest_node, softmax_ids=softmax_ids, selected_ids=selected_ids)

    # if length(rews) > 1
    #     sd = std(rews)
    #     rews = sd < 1e-6 ? (rews .- mean(rews)) : (rews .- mean(rews)) ./ sd
    # end
    # @show rews
    for (A, G, ids) in zip(rews, v_returns, softmax_ids)
        push!(q, OnePPOSample(Float32[], A, G, -1, 1, ids))
    end
    actor_inputs = [expr(MyModule.nc, soltree[i].ex) for i in inputs_vec]
    actor_inputs = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(actor_inputs, sym_enc))

    return PPOMinibatch(actor_inputs, q, 0.2)
end
@everywhere function linear_get_one_batch(env, policy, value_model; gamma=0.9, max_depth=100, max_steps=100, lambda=0.95)
    env = MyTreeEnv(env, policy)
    inputs_actions = [[env.s_init]]
    traj = []
    for t in 1:max_steps
        possible_actions = action_space(env)
        weights = [only(policy(x)) for x in possible_actions]
        node_index = StatsBase.sample(1:length(weights), Weights(softmax(weights)))
        a = possible_actions[node_index]
        s = state(env)
        act!(env, a)
        ns = state(env)
        r = reward(env, a, s)
        is_done = isempty(action_space(env)) || is_terminal(env)
        push!(traj, (s, a, r, ns, is_done))
        other_indexes = setdiff(1:length(possible_actions), [node_index])
        # @show possible_actions[other_indexes]
        # @show a
        push!(inputs_actions, vcat(a, possible_actions[other_indexes]))
        if is_done
            break
        end
    end

    # @show traj.rewards
    empty!(MyModule.memoize_cache(MyModule.general_leaf_cached_inference))
    empty!(MyModule.memoize_cache(MyModule.general_expr_cached_inference))
    q = OnePPOSample[]
    inputs = []
    rews = Float32[]
    v_returns = Float32[]
    softmax_ids = []
    gae = 0
    for (ind,((_, _, r, _, _), inp_actions)) in enumerate(zip(traj, inputs_actions))
        # ind == 1 && continue
        current_values = map(x->only(value_model(x)), inp_actions)
        if ind == 1
            δ = r - maximum(current_values)
        else
            next_values = map(x->only(value_model(x)), inputs_actions[ind-1])
            δ = r + gamma * maximum(next_values) - maximum(current_values)
        end
        gae = δ + gamma * lambda * gae
        Aₜ = gae
        G = gae + maximum(current_values)
        push!(rews, Aₜ)
        push!(v_returns, G)
        push!(softmax_ids, [])
        for j in inp_actions
            if j in inputs
                tmp_ind = findfirst(==(j), inputs)
                push!(softmax_ids[end], tmp_ind)
            else
                push!(inputs, j)
                push!(softmax_ids[end], length(inputs))
            end
        end
    end
    # if length(rews) > 1
    #     sd = std(rews)
    #     rews = sd < 1e-6 ? (rews .- mean(rews)) : (rews .- mean(rews)) ./ sd
    # end
    for (A, G, ids) in zip(rews, v_returns, softmax_ids)
        push!(q, OnePPOSample(Float32[], A, G, -1, 1, ids))
    end
    actor_inputs = [expr(MyModule.nc, i) for i in inputs]
    actor_inputs = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(actor_inputs, sym_enc))

    return PPOMinibatch(actor_inputs, q, 0.2)
end


function train(model, value_model, target_model, data)
    actor_fval, critic_fval, n = 0.0, 0.0, 0
    actor_opt = Optimisers.setup(ADAM(), model)
    value_opt = Optimisers.setup(ADAM(), value_model)
    MyModule.reset_all_function_caches()
    model_name = "gae_tree_PPO_"
    @everywhere model = $model
    @everywhere target_model = $target_model
    @everywhere value_model = $value_model
    for i in 1:100
        search_time = @elapsed all_d = pmap(data) do d
            empty!(MyModule.nc)
            MyModule.reset_all_function_caches()
            # get_one_batch(intern!(d), model, value_model, target_model; gamma=0.9,max_depth=100, max_steps=100, lambda=0.95)
            # linear_get_one_batch(d, model, value_model; gamma=0.9,max_depth=100, max_steps=100, lambda=0.95)
            get_one_batch_no_solution(intern!(d), model, value_model; gamma=0.9,max_depth=100, max_steps=100, lambda=0.95)
        end
        # search_time = 0
        # all_d = [deserialize("modesl/some_nodes.bin")]
        update_time = @elapsed for _ in 1:1
            for d in all_d
                fᵢ, ∇model = Flux.withgradient(model -> actor_loss(model, target_model, d), model)
                # serialize("models/some_nodes.bin", d)
                !isfinite(fᵢ) && error("Actor loss is $fᵢ on data item")
                actor_fval += fᵢ
                state_tree, model = Optimisers.update(actor_opt, model, only(∇model))

                fᵢ, ∇model = Flux.withgradient(model -> critic_loss(model, d), value_model)
                !isfinite(fᵢ) && error("Critic loss is $fᵢ on data item")
                critic_fval += fᵢ
                state_tree, value_model = Optimisers.update(value_opt, value_model, only(∇model))
            end
        end
        if mod(i, 4) == 0
            target_model = deepcopy(model)
        end
        n += 1
        # serialize("models/ppo_pevnak/trained_parallel_" * model_name * "ep$(i)_gamma09_innep1_for_graph_stats.bin", model)
        # serialize("models/ppo_pevnak/trained_parallel_vl_" * model_name * "ep$(i)_gamma09_innep1_for_graph_stats.bin", value_model)
        println("Ep $(i): actor loss = $(round(actor_fval / n, digits = 3)); critic loss = $(round(critic_fval / n, digits = 3)); trajectory took --> $(round(search_time, digits=2)); update took --> $(round(update_time, digits=2))")
        empty!(MyModule.nc)
        MyModule.reset_all_function_caches()
    end
end

# all_d = [deserialize("modesl/some_nodes.bin")]
# model = deserialize("modesl/ppo_pevnak/trained_parallel_gae_tree_PPO_ep14_gamma09_innep1_for_graph_stats.bin")
# value_model = deserialize("modesl/ppo_pevnak/trained_parallel_vl_gae_tree_PPO_ep14_gamma09_innep1_for_graph_stats.bin")
# target_model = deserialize("modesl/ppo_pevnak/trained_parallel_tr_gae_tree_PPO_ep14_gamma09_innep1_for_graph_stats.bin")
train(model, value_model, target_model, data[1:10])

