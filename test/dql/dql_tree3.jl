using CSV
using DataFrames
using Plots
using Optimisers
using Statistics
include("my_replay_buffer.jl")
using MyModule
using MyModule.Flux
using MyModule.Mill
using MyModule: all_expand, exp_size, Node, NodeID, intern!, Node, push_to_tree!, expr
include("tree_env_setup.jl")
using Serialization


gamma = 0.99
# epsilone_decay = 0.9 / 2.5e2
epsilon = eps_start = 1.0
eps_end = 0.01
eps_decay = 0.975
# eps_decay = 0.995
number_of_iterations = 1000
batch_size = 64
input_size = 64

# my_episode_buffer = Vector{Tuple{Array{Float32}, Int, Float32, Array{Float32}, Bool}}()
loss_over_time = []
rewards_over_time = []
update_every = 0
function ffnn(idim, hidden_size, layers)
    layers == 1 && return Dense(idim, hidden_size, Flux.gelu)
    layers == 2 && return Flux.Chain(Dense(idim, hidden_size, Flux.gelu), Dense(hidden_size, hidden_size, Flux.gelu))
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

online_q = Chain(Dense(hidden_size, 1))
target_q = Chain(Dense(hidden_size, 1))
target_q = deepcopy(online_q)

online_embedding_model = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    online_q
    );

target_embedding_model = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    target_q
    );

function soft_update!(online_q, target_q, tau::Float64)
    # Iterate through each layer of the networks
    for (online_layer, target_layer) in zip(online_q.layers, target_q.layers)
        # Perform the soft update on the weights
        target_layer.weight .= tau * online_layer.weight .+ (1.0 - tau) * target_layer.weight
        target_layer.bias .= tau * online_layer.bias .+ (1.0 - tau) * target_layer.bias
    end
end
    
function update_ddqn!(buffer, online_q, target_q, policy_params_optimiser; batch_size=64, gamma=0.995)
    sampled_experiences = sample(buffer, batch_size)
    states = [i[1] for i in sampled_experiences]
    rewards = [i[2] for i in sampled_experiences]
    next_states = [i[3] for i in sampled_experiences]
    

    next_states = hcat(next_states...)
    target_values = rewards + gamma * vec(target_q(next_states))
    
    states = hcat(states...)
    sa, grad = Flux.Zygote.withgradient(online_q) do oq
        expected_values = vec(oq(states))
        loss = mean((target_values - expected_values) .^ 2)
        return loss
    end
    Optimisers.update!(policy_params_optimiser, online_q, grad[1])
    return sa
end

function embed_ex(ex, policy_model)
    inference_type = MyModule.get_inference_type(ex)
    ds = MyModule.general_cached_inference(ex, inference_type, policy_model)
    return ds
end
function update_ddqn2!(buffer, online_embedding, target_embedding, policy_params_optimiser; batch_size=64, gamma=0.99)
    sampled_experiences = sample(buffer, batch_size)
    states = [expr(MyModule.nc, i[1]) for i in sampled_experiences]
    rewards = [i[2] for i in sampled_experiences]
    next_states = [expr(MyModule.nc, i[1]) for i in sampled_experiences]

    states = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(states, sym_enc))
    next_states = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(next_states, sym_enc))

    target_values = rewards + gamma * vec(MyModule.heuristic(target_embedding, next_states))
    
    sa, grad = Flux.Zygote.withgradient(online_embedding) do oq
        expected_values = vec(MyModule.heuristic(oq, next_states))
        loss = mean((target_values - expected_values) .^ 2)
        return loss
    end
    Optimisers.update!(policy_params_optimiser, online_embedding, grad[1])
    return sa
end

function extract_path2buffer2!(soltree, root, my_buffer)
    for v in values(soltree)
        if v.node_id == v.parent
            continue
        end
        ns = v.ex
        s = soltree[v.parent].ex
        size_current = exp_size(ns)
        size_init = exp_size(root.ex)
        if size_current >= size_init
            r = -1
        else
            r = size_init - size_current
        end
        add2buffer!(my_buffer, (s, r, ns))
    end
end

function extract_path2buffer!(soltree, smallest_node, root, my_buffer, embedding_model)
    if smallest_node.node_id == smallest_node.parent
        return
    end
    
    size_current = exp_size(smallest_node.ex)
    size_init = exp_size(root.ex)
    if size_current >= size_init
        r = -1
    else
        r = size_init - size_current
    end
    s = embed_ex(soltree[smallest_node.parent].ex, embedding_model)
    ns = embed_ex(smallest_node.ex, embedding_model)
    add2buffer!(my_buffer, (s, r, ns))
    extract_path2buffer!(soltree, soltree[smallest_node.parent], root, my_buffer, embedding_model) 
end

function train(online_embedding_model, target_embedding_model, data; max_iterations=10, max_steps=100, batch_size=128, lr=0.001, buffer_size=1_000_000)
    my_buffer = MyReplayBuffer(buffer_size)
    pol_optimizer = ADAM(lr)
    online_q_params = Flux.setup(pol_optimizer, online_embedding_model)
    best_solutions = [[intern!(i), intern!(i), exp_size(i)] for i in data]
    MyModule.reset_all_function_caches()
    target_update_freq = 0
    start = time()
    duration = 240 * 60  # 10 minutes in seconds
    iter = 0
    rewards_logs = []
    # while time() - start < duration
    for iter in 1:max_iterations
        best_solutions = map(best_solutions) do i
            [i[1], i[1], exp_size(i[1])]
        end
        for (ind, ex) in enumerate(data)
            search_time = @elapsed soltree, smallest_node, root = MyModule.initialize_tree_search(intern!(ex), online_embedding_model; max_expansions=max_steps, max_depth=100)
            @show search_time
            extract_path2buffer2!(soltree, root, my_buffer)
            target_update_freq += 1
            if length(my_buffer.buffer) >= batch_size
                update_time = @elapsed update_ddqn2!(my_buffer, online_embedding_model, target_embedding_model, online_q_params, batch_size=batch_size)
                @show update_time
                # embedding_model.heuristic = online_q
                if mod(target_update_freq, 10) == 0
                    # soft_update!(online_q, target_q, 0.5)
                    target_embedding_model = deepcopy(online_embedding_model)
                end
            end
            if best_solutions[ind][3] > exp_size(smallest_node.ex)
                best_solutions[ind][3] = exp_size(smallest_node.ex)
                best_solutions[ind][2] = smallest_node.ex
            end
        end
        final_rewards = 0
        for bs in best_solutions
            # println("Initial $(bs[1])")
            # println("Best $(bs[2])")
            # println("Reward $(abs(exp_size(bs[2]) - exp_size(bs[1])))")
            final_rewards += abs(exp_size(bs[2]) - exp_size(bs[1]))
        end
        push!(rewards_logs, final_rewards / length(best_solutions))
        @show rewards_logs[end]
    end
    return rewards_logs
end

max_iterations=10
max_steps=100
batch_size=64
lr=0.001
buffer_size=100_000

rewards_logs = train(online_embedding_model, target_embedding_model, data[end-10:end], max_iterations=max_iterations, max_steps=max_steps, batch_size=batch_size, lr=lr, buffer_size=buffer_size)

plot(rewards_logs, xlabel="Iterations", ylabel="Averege expression simplification")
savefig("stats/tree_ddql_averege_exp_simplification_$(max_iterations)iter_$(max_steps)maxsteps_$(batch_size)batchsize_$(buffer_size)buffsize_$(lr)lr.png")

serialize("models/tree_ddql_embedding.bin", embedding_model)

function validation(data, embedding_model, name)
    MyModule.reset_all_function_caches()
    best_solutions = [[intern!(i), intern!(i), exp_size(i)] for i in data]
    target_update_freq = 0
    for (ind, ex) in enumerate(data)
        soltree, smallest_node, root = MyModule.initialize_tree_search(intern!(ex), embedding_model; max_expansions=max_steps, max_depth=100)
        target_update_freq += 1
        if best_solutions[ind][3] > exp_size(smallest_node.ex)
            best_solutions[ind][3] = exp_size(smallest_node.ex)
            best_solutions[ind][2] = smallest_node.ex
        end
    end
    df = DataFrame(InitEx = Any[], RedEx = Any[], RedDif = Any[])
    final_rewards = 0
    for bs in best_solutions
        println("Initial $(bs[1])")
        println("Best $(bs[2])")
        println("Reward $(abs(exp_size(bs[2]) - exp_size(bs[1])))")
        fin_rew = abs(exp_size(bs[2]) - exp_size(bs[1]))
        final_rewards += fin_rew
        push!(df, (InitEx = MyModule.expr(MyModule.nc, bs[1]), RedEx = MyModule.expr(MyModule.nc, bs[2]), RedDif = fin_rew))
    end
    @show final_rewards / length(best_solutions)
    CSV.write("stats/tree_dql_$(name)_1h.csv", df)
end
validation(data[end-10:end], online_embedding_model, "train")

experiment_name = "tree_dql_test1h"
train_data_path = "./data/neural_rewrter/test.json"
train_data = load_data(train_data_path)[1:1_000]
train_data = filter(x->!occursin("select", x[1]), train_data)
train_data = preprosses_data_to_expressions(train_data)
sorted_data = sort(train_data, by=x->MyModule.exp_size(x))
data = sorted_data


validation(data, embedding_model, "test")