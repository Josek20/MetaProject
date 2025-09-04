using Serialization
using Distributed
addprocs()
@everywhere begin
    using MyModule
    using MyModule.Flux
    using MyModule.Mill
    using MyModule: all_expand, exp_size, Node, NodeID, intern!, expr, DeduplicatingNode, AbstractModel
    using TimerOutputs
    const TO = TimerOutput()
    reset_timer!(TO)
    include("rl_pipeline.jl")
    include("sampler.jl")
    include("learner.jl")
end

function get_data()
    train_data_path = "./data/neural_rewrter/train.json"
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
@everywhere Base.only(t::Matrix{Float32}) = size(t)[1] > 1 ? Tuple(vec(t)) : first(t)
model = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, 1))
    );

target = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, 1))
    );
@everywhere function not_boosted(d, sampler, traj)
    input_values = get_input_values(traj)
    target = get_target(traj.rewards, sampler)
    (;ds=input_values,rew=target,goal_size=-1, initial_expr=d.initial_expr, depth=traj.pointer)
end
@everywhere function boosted(d, sampler, traj)
    input_values = get_input_values(traj)
    target = get_target(traj.rewards, sampler)
    new_smallest_node_size = minimum(exp_size.(traj.next_states))
    if d.goal_size > new_smallest_node_size
        input_values = get_input_values(traj)
        target = get_target(traj.rewards, sampler)
        # return(;ds=input_values,rew=target,goal_size=new_smallest_node_size, initial_expr=d.initial_expr, depth=traj.pointer)
        return(;ds=input_values,rew=target,goal_size=-1, initial_expr=d.initial_expr, depth=traj.pointer)
    elseif (d.goal_size > new_smallest_node_size && d.depth > traj.pointer)
        target = get_target(traj.rewards, sampler)
        input_values = get_input_values(traj)
        # return(;ds=input_values,rew=target,goal_size=new_smallest_node_size, initial_expr=d.initial_expr, depth=traj.pointer)
        return(;ds=input_values,rew=target,goal_size=-1, initial_expr=d.initial_expr, depth=traj.pointer)
    else
        return(d)
    end
end
function train(model, data, target_model)
    @everywhere model = $model
    @everywhere target_model = $target_model
    @everywhere is_target = false
    learner = DummyLerner(Flux.mse, model, max_iter=10)
    epochs = 2
    @everywhere is_boosted = false
    @everywhere sampler = TreeSampler(max_steps=100, max_depth=100, epsilon=1.0, eps_decay=0.80, is_directed=false, n_best=-1, batch=128)
    # @everywhere sampler = DAGSampler(max_steps=100, max_depth=100, epsilon=1.0, eps_decay=0.80, is_directed=false, n_best=-1, batch=128)
    # @everywhere sampler = DGSampler(max_steps=100, max_depth=100, epsilon=1.0, eps_decay=0.80, is_directed=false, n_best=-1, batch=128)
    # @everywhere sampler = TreeSampler2Values(max_steps=100, max_depth=100, epsilon=1.0, eps_decay=0.80, is_directed=false, n_best=-1, batch=128)
    samples = [(;ds=nothing,rew=[],goal_size=typemax(Int), initial_expr=i, depth=typemax(Int)) for i in data]
    for e in 1:epochs
        # results = pmap(d->MyModule.initialize_tree_search_epsilon(d, model; max_expansions=1000, max_depth=100, epsilon=1.0), data[1:100])
        search_time = @elapsed samples = pmap(samples) do d
            empty!(MyModule.nc)
            MyModule.reset_all_function_caches()
            if is_target
                traj = sample_trajectory(sampler, d.initial_expr, model, target_model)
            else
                traj = sample_trajectory(sampler, d.initial_expr, model)
            end
            @show MyModule.cache_status()
            if is_boosted
                boosted(d, sampler, model)
            else
                not_boosted(d, sampler, traj)
            end
        end

        loss_over_time = 0
        update_time = @elapsed for _ in 1:10
            learning_time = 0
            loss = 0
            for (ind, s) in enumerate(samples)
                learning_time += @elapsed loss += compute_gradient!(s.rew, s.ds, model, learner)
            end
            loss_over_time += loss / length(samples)
        end
        update_epsilon!(sampler)
        if mod(e, 10) == 0
            target_model = deepcopy(model)
        end
        res = 0.0
        val_time = 0.0
        serialize("models/test_ep$(e).bin", model)
        println("Ep $(e): lres, tres = $([0, res]); loss = $(loss_over_time / 10);epsilon=$(round(sampler.epsilon, digits=2)); trajectory took --> $(round(search_time, digits=2)); update took --> $(round(update_time, digits=2)); validation took --> $(round(val_time, digits=2))")
    end
end
train(model, data[1:10], target)