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