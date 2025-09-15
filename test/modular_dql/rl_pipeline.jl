abstract type AbstractEnvironment end
# abstract type AbstractModel end
abstract type AbstractSampler end
abstract type AbstractLearner end

mutable struct RLPipeline{E <: AbstractEnvironment, M <: AbstractModel, S <: AbstractSampler, L <: AbstractLearner}
    env::E
    model::M
    sampler::S
    learner::L
end
mutable struct SimpleRLPipeline{E <: AbstractEnvironment, M <: AbstractModel, S <: AbstractSampler, L <: AbstractLearner, TM <: AbstractModel}
    env::E
    model::M
    sampler::S
    learner::L
    target_model::TM
end
clean_cache(model::AbstractModel) = nothing
function clean_cache(model::ExprModel)
    empty!(MyModule.memoize_cache(MyModule.general_expr_cached_inference))
    empty!(MyModule.memoize_cache(MyModule.general_leaf_cached_inference))
    # MyModule.reset_all_function_caches()
end

function train!(pipeline::RLPipeline; episodes::Int=100)
    MyModule.reset_all_function_caches()
    for episode in 1:episodes
       trajectory_time = @elapsed traj = sample_trajectory(pipeline.sampler, pipeline.env, pipeline.model)
       clean_cache(pipeline.model)
       update_time = @elapsed update_model!(pipeline.learner, pipeline.model, traj)
       update_epsilon!(pipeline.sampler)
    #    results = validation(traj)
    #    results = validation1(pipeline)
       results = validation2(pipeline, traj)
       clean_cache(pipeline.model)
       println("Ep $(episode): lres, tres = $(results);epsilon=$(round(pipeline.sampler.epsilon, digits=2)); update took --> $(round(update_time, digits=2)); trajectory took --> $(round(trajectory_time, digits=2))")
    end
end
function validation3(pipeline, data)
    val_time = @elapsed trajectories = map(data) do d
        pipeline.env.s_init = intern!(d)
        reset!(pipeline.env)
        sampler = pipeline.sampler
        env = pipeline.env
        soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search_epsilon(state(env), pipeline.model; max_expansions=sampler.max_steps, max_depth=sampler.max_depth, epsilon=0.0)
        return(exp_size(root.ex) - exp_size(smallest_node.ex))
    end
    return mean(trajectories), val_time
end
function train!(pipeline::SimpleRLPipeline, data::Vector{Expr}; episodes::Int=100)
    MyModule.reset_all_function_caches()
    trajectories = []
    loss = [(;ex=intern!(e), loss_over_time=Float32[]) for e in data]
    train_validation = [(;ex=intern!(e), val=Vector{Tuple{Float32, Float32}}()) for e in data]
    samples = [(;ds=nothing,rew=[],goal_size=typemax(Int), initial_expr=intern!(i), depth=typemax(Int)) for i in data]
    for episode in 1:episodes
        @timeit TO "sample trajectories" trajectory_time = @elapsed samples = map(samples) do d
            # @show d.initial_expr
            pipeline.env.s_init = d.initial_expr
            reset!(pipeline.env)
            traj = sample_trajectory(pipeline.sampler, pipeline.env, pipeline.model)
            # traj = sample_trajectory(pipeline.sampler, pipeline.env, pipeline.model, pipeline.target_model)

            input_values = get_input_values(traj)
            target = get_target(traj.rewards, pipeline.sampler)
            # @show length(input_values.ii), traj.rewards, 
            # @assert length(traj.rewards) * length(traj.rewards[1]) == length(input_values.ii) * 2 == length(target)
            return(;ds=input_values,rew=target,goal_size=-1, initial_expr=d.initial_expr, depth=traj.pointer)
        end
        # clean_cache(pipeline.model)
        loss_over_time = 0
        @timeit TO "train" update_time = @elapsed for _ in 1:10
            learning_time = 0
            loss = 0
            for (ind, s) in enumerate(samples)
                # @show length(s.rew), length(s.ds.ii)
                learning_time += @elapsed loss += compute_gradient!(s.rew, s.ds, pipeline.model, pipeline.learner)
            end
            loss_over_time += loss / length(samples)
        end
        # serialize("models/trained_DQN_second_DG2_ep$(episode)_for_graph_stats.bin", pipeline.model)
        update_epsilon!(pipeline.sampler)
        if mod(episode, 10) == 0
            pipeline.target_model = deepcopy(pipeline.model)
        end
        clean_cache(pipeline.model)
        @timeit TO "validation" res, val_time = validation3(pipeline, data)
        println("Ep $(episode): lres, tres = $([0, res]); loss = $(loss_over_time / 10);epsilon=$(round(pipeline.sampler.epsilon, digits=2)); update took --> $(round(update_time, digits=2)); trajectory took --> $(round(trajectory_time, digits=2)); validation took --> $(round(val_time, digits=2))")
    end
    return samples, (;loss_stats=loss, val_stats=train_validation)
end

function preprocessing(traj)
    best_traj = argmin(exp_size.(traj.next_states))
    traj.next_states = traj.next_states[1:best_traj]
    traj.actions = traj.actions[1:best_traj]
    traj.states = traj.states[1:best_traj]
    traj.rewards = traj.rewards[1:best_traj]
    traj.is_dones = traj.is_dones[1:best_traj]
    return traj
end
function preprocessing(traj, target_model; γ=0.90)
    # best_traj = argmin(exp_size.(traj.next_states))
    best_traj = length(traj.next_states)
    traj.next_states = traj.next_states[1:best_traj]
    # pushfirst!(traj.next_states, traj.states[1])
    traj.actions = traj.actions[1:best_traj]
    traj.states = traj.states[1:best_traj]
    traj.rewards = traj.rewards[1:best_traj]
    for (ind,(s,sₜ)) in enumerate(zip(traj.states, traj.actions))
        if ind != best_traj
            # @show only.(target_model.(sₜ))
            traj.rewards[ind] = traj.rewards[ind] + γ * maximum(only.(target_model.(sₜ)))
        else
            traj.rewards[ind] = traj.rewards[ind]
        end
    end
    traj.is_dones = traj.is_dones[1:best_traj]
    return traj
end
function train1!(pipeline::SimpleRLPipeline, data::Vector{Expr}; episodes::Int=100)
    MyModule.reset_all_function_caches()
    loss = [(;ex=intern!(e), loss_over_time=Float32[]) for e in data]
    train_validation = [(;ex=intern!(e), val=Vector{Tuple{Float32, Float32}}()) for e in data]
    # trajectories = [Trajectory(1, NodeID[intern!(i)], NodeID[intern!(i)], Float32[0f0], NodeID[intern!(i)], Bool[false]) for i in data]
    samples = [(;ds=nothing,rew=[],goal_size=typemax(Int), initial_expr=intern!(i), depth=typemax(Int)) for i in data]
    for episode in 1:episodes
        trajectory_time = @elapsed samples = map(samples) do d
            pipeline.env.s_init = d.initial_expr
            reset!(pipeline.env)
            # traj = sample_trajectory(pipeline.sampler, pipeline.env, pipeline.model)
            traj = sample_trajectory(pipeline.sampler, pipeline.env, pipeline.model, pipeline.target_model)
            # traj = preprocessing(traj)
            # smallest_node = traj.next_states[argmin(exp_size.(traj.next_states))]
            new_smallest_node_size = minimum(exp_size.(traj.next_states))
            # traj = preprocessing(traj, pipeline.target_model)
            if d.goal_size > new_smallest_node_size
                input_values = get_input_values(traj)
                target = get_target(traj.rewards, pipeline.sampler)
                return(;ds=input_values,rew=target,goal_size=new_smallest_node_size, initial_expr=d.initial_expr, depth=traj.pointer)
            elseif (d.goal_size > new_smallest_node_size && d.depth > traj.pointer)
                target = get_target(traj.rewards, pipeline.sampler)
                input_values = get_input_values(traj)
                return(;ds=input_values,rew=target,goal_size=new_smallest_node_size, initial_expr=d.initial_expr, depth=traj.pointer)
            else
                return(d)
            end
        end
        # serialize("models/trained_DQN_ep$(episode)_for_graph_stats.bin", model)
        loss_over_time = 0
        update_time = @elapsed for _ in 1:10
            learning_time = 0
            loss = 0
            for (ind, s) in enumerate(samples)
                learning_time += @elapsed loss += compute_gradient!(s.rew, s.ds, pipeline.model, pipeline.learner)
            end
            loss_over_time += loss / length(samples)
        end
        # serialize("models/trained_DQN_first_DAG1_ep$(episode)_for_graph_stats.bin", pipeline.model)
        update_epsilon!(pipeline.sampler)
        clean_cache(pipeline.model)
        res, val_time = validation3(pipeline, data)
        println("Ep $(episode): avg_input_size=$(round(mean(map(x->length(x.ds.ii), samples)))); lres, tres = $([0, res]); loss = $(loss_over_time / 10);epsilon=$(round(pipeline.sampler.epsilon, digits=2)); update took --> $(round(update_time, digits=2)); trajectory took --> $(round(trajectory_time, digits=2)); validation took --> $(round(val_time, digits=2)),")
    end
    return samples, (;loss_stats=loss, val_stats=train_validation)
end

function visualization(soltree, online_policy, target_values::Dict)
    empty!(MyModule.memoize_cache(MyModule.general_cached_inference))
    children = Vector[]
    text = []
    link_style = [""]
    style = [""]
    index_to_put = 0
    current_i = 1
    buff = []
    sort_by_depth = sort(collect(values(soltree)), by=x->x.depth)
    i = sort_by_depth[current_i]

    append!(buff, [soltree[j] for j in i.children])
    # push!(children, [soltree[j].ex for j in i.children])
    current_range = 2:length(i.children) + 1
    push!(children, collect(Any, current_range))
    push!(text, string(expr(MyModule.nc, i.ex)))
    # append!(text, [string(expr(MyModule.nc, soltree[j].ex)) for j in i.children])
    while !isempty(buff) 
    # for _ in 1:43
        n = popfirst!(buff)
        # @show n.ex
        # r = exp_size(soltree[n.parent].ex) - exp_size(n.ex)
        r = exp_size(i.ex) - exp_size(n.ex)
        # r = target_values[n.ex]
        push!(text, string(expr(MyModule.nc, n.ex)) * "\nPred:$(round(only(online_policy(n.ex)), digits=4))\nRew:$(r)")
        # push!(text, string(expr(MyModule.nc, n.ex)) * "\nRew:$(r)")
        current_range = current_range.stop + 1:current_range.stop + length(n.children)
        push!(children, collect(Any, current_range))
        append!(buff, [soltree[j] for j in n.children])
    end
    t = D3Tree(children, text=text, init_expand=30)
    # t = D3Tree(children[2:end], text=text, style=style, link_style=link_style, init_expand=2,  svg_node_size=(2020, 2020))
    inbrowser(t, "Mircosoft Edge")
end
function visualization_makie(soltree::Dict, model::ExprModel, mcache::Dict)
    # using Graphs
    # using GraphMakie
    # using CairoMakie
    # using StatsBase
    # using GLMakie

    GLMakie.activate!()
    g = DiGraph(length(soltree))
    id2gid = Dict(n.node_id => ind  for (ind, n) in enumerate(values(soltree)))
    gid2id = Dict(i => ind for (ind, i) in id2gid)
    for node in values(soltree)
        if node.parent != node.node_id
            add_edge!(g, id2gid[node.parent], id2gid[node.node_id])
        end
    end
    depths = countmap([n.depth for n in values(soltree)])
    layout = map(values(soltree)) do node
        c = node.depth
        y = depths[c]
        depths[c] -= 2
        Point2f(12 * c, 12 * y)
    end
    fig = Figure(resolution = (300, 100))  # Super wide
    ax = Axis(fig[1, 1])
    # fig, _, _ = graphplot(ax, g; layout=layout, edge_linestyle=:solid)
    graphplot!(ax, g; layout=layout)
    # fig, ax, plt = graphplot(g, layout=layout, edge_linestyle=:solid)
    # ax.xlimits = (-2, 10)
    # GraphMakie.xlims!(ax, -2, 10)
    for (i, pos) in enumerate(layout)
        text!(ax, string(MyModule.expr(MyModule.nc, soltree[gid2id[i]].ex)); position=pos, align=(:center, :center), fontsize=14, color=:blue)
    end
    display(fig)
    # fig, ax, plt = graphplot(g; layout=layout, nlabels=string.([MyModule.expr(MyModule.nc, soltree[gid2id[i]].ex) for i in 1:nv(g)]))
end
function visualization_makie_in_circles(soltree::Dict, model::ExprModel, mcache::Dict=Dict())
    g = DiGraph(length(soltree))
    id2gid = Dict(n.node_id => ind  for (ind, n) in enumerate(values(soltree)))
    gid2id = Dict(i => ind for (ind, i) in id2gid)
    for node in values(soltree)
        if node.parent != node.node_id
            add_edge!(g, id2gid[node.parent], id2gid[node.node_id])
        end
    end
    # depths = countmap([n.depth for n in values(soltree)])
    depth_dict = Dict(i=>[] for i in 1:10)
    for (k, v) in soltree
        v.depth == 0 && continue
        push!(depth_dict[v.depth], v.node_id)
    end
    for (depth, nodes) in depth_dict
        n = length(nodes)
        for (i, node) in enumerate(nodes)
            θ = 2π * (i - 1) / n
            r = depth * radious_step
            x = r * cos(θ)
            y = r * sin(θ)
            position[node] = x, y
        end
    end
    # layout = map(values(soltree)) do node
    #     c = node.depth
    #     y = depths[c]
    #     depths[c] -= 2
    #     Point2f(12 * c, 12 * y)
    # end
end

# sample_trajectory(s::AbstractSampler, env::AbstractEnvironment, model::AbstractModel)::Trajectory = error("sample_trajectory not inmplemented")
# update_model!(l::AbstractLearner, model::AbstractModel, traj::Trajectory) = error("sample_trajectory not inmplemented")