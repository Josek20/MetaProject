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
    empty!(MyModule.memoize_cache(MyModule.general_cached_inference))
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
    for episode in 1:episodes
        trajectory_time = @elapsed trajectories = map(data) do d
            pipeline.env.s_init = intern!(d)
            reset!(pipeline.env)
            # sample_trajectory(pipeline.sampler, pipeline.env, pipeline.model)
            sample_trajectory(pipeline.sampler, pipeline.env, pipeline.model, pipeline.target_model)
        end
        clean_cache(pipeline.model)
        update_time = @elapsed loss = map(zip(loss, trajectories)) do (ls, traj)
            l = update_model!(pipeline.learner, pipeline.model, traj)
            # (;ex=traj.states[end][end], )
            push!(ls.loss_over_time, l)
            ls 
        end
        update_epsilon!(pipeline.sampler)
        if mod(episode, 10) == 0
            pipeline.target_model = deepcopy(pipeline.model)
        end
        results = [0f0,0f0]
        # train_validation = map(zip(train_validation, trajectories)) do (val, traj)
        #     # tmp = validation2(pipeline, traj)
        #     results[2] += exp_size(val.ex) - traj.pointer
        #     push!(val.val, (0f0, exp_size(val.ex) - traj.pointer))
        #     val
        # end
        clean_cache(pipeline.model)
        res, val_time = validation3(pipeline, data)
        # println("Ep $(episode): lres, tres = $(results / length(trajectories));epsilon=$(round(pipeline.sampler.epsilon, digits=2)); update took --> $(round(update_time, digits=2)); trajectory took --> $(round(trajectory_time, digits=2))")
        println("Ep $(episode): lres, tres = $([0, res]);epsilon=$(round(pipeline.sampler.epsilon, digits=2)); update took --> $(round(update_time, digits=2)); trajectory took --> $(round(trajectory_time, digits=2)); validation took --> $(round(val_time, digits=2))")
    end
    return trajectories, (;loss_stats=loss, val_stats=train_validation)
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
    trajectories = [Trajectory(1, NodeID[intern!(i)], NodeID[intern!(i)], Float32[0f0], NodeID[intern!(i)], Bool[false]) for i in data]
    for episode in 1:episodes
        trajectory_time = @elapsed trajectories = map(trajectories) do d
            pipeline.env.s_init = d.states[1]
            reset!(pipeline.env)
            traj = sample_trajectory(pipeline.sampler, pipeline.env, pipeline.model)
            # traj = preprocessing(traj)
            # traj = preprocessing(traj, pipeline.target_model)
            if exp_size(d.next_states[end]) > exp_size(traj.next_states[end]) || 
                (exp_size(d.next_states[end]) == exp_size(traj.next_states[end]) && length(d.next_states) > length(traj.next_states))
                return traj
            else
                return d
            end
        end
        clean_cache(pipeline.model)
        update_time = @elapsed for traj in trajectories
            update_model!(pipeline.learner, pipeline.model, traj)
        end
        # pipeline.target_model = deepcopy(pipeline.model)
        update_epsilon!(pipeline.sampler)
        results = [0f0,0f0]
        for traj in trajectories
            results .+= validation2(pipeline, traj)
        end
        clean_cache(pipeline.model)
        println("Ep $(episode): lres, tres = $(results / length(trajectories));epsilon=$(round(pipeline.sampler.epsilon, digits=2)); update took --> $(round(update_time, digits=2)); trajectory took --> $(round(trajectory_time, digits=2))")
    end
    return trajectories
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
# sample_trajectory(s::AbstractSampler, env::AbstractEnvironment, model::AbstractModel)::Trajectory = error("sample_trajectory not inmplemented")
# update_model!(l::AbstractLearner, model::AbstractModel, traj::Trajectory) = error("sample_trajectory not inmplemented")