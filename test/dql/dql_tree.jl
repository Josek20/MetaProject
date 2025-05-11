using Plots
using Optimisers
using Statistics
include("my_replay_buffer.jl")
using MyModule
using MyModule.Flux
using MyModule.Mill
using MyModule: all_expand, exp_size, Node, NodeID, intern!, Node, push_to_tree!
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

# if ARGS[6] == 1
online_q = Chain(Dense(hidden_size, 1))
target_q = Chain(Dense(hidden_size, 1))
target_q = deepcopy(online_q)
# elseif ARGS[6] == 2
#     online_q = Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, 1))
#     target_q = Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, 1))
#     target_q = deepcopy(online_q)
# else
#     online_q = Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, hidden_size, relu), Dense(hidden_size, 1))
#     target_q = Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, hidden_size, relu), Dense(hidden_size, 1))
#     target_q = deepcopy(online_q)
# end
embedding_model = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    online_q
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

update_epsilon(epsilon; eps_decay=0.995, eps_min=0.1) = max(eps_min, eps_decay * epsilon)

mutable struct MyTreeEnv
    s_init
    s_current
    t
    policy_model
    soltree
end
function MyTreeEnv(ex::Expr, policy_model)
    inex = intern!(ex)

    soltree = Dict{UInt64, Node}()
    root = Node(inex, (), hash(inex), 0)
    soltree[root.node_id] = root
    return MyTreeEnv(root, root, 1, policy_model, soltree)
end
function state(env::MyTreeEnv)
    inference_type = MyModule.get_inference_type(env.s_current.ex)
    ds = MyModule.general_cached_inference(env.s_current.ex, inference_type, env.policy_model)
    return vec(ds)
end
# function reset!(env::MyTreeEnv, ex::NodeID)
#     env.t = 1
#     soltree = Dict{UInt64, Node}()
#     root = Node(inex, (), hash(inex), 0)
#     soltree[root.node_id] = root
    
# end
function reset!(env::MyTreeEnv)
    env.t = 1
    soltree = Dict{UInt64, Node}()
    root = Node(env.s_init, (), hash(env.s_init), 0)
    soltree[root.node_id] = root
    env.s_init = root
    env.s_current = root
    env.soltree = soltree
    return env
end
function action_space(env::MyTreeEnv)
    env.t += 1
    # new_ex, _ = all_expand(env.s_current, theory)
    # return new_ex
    all_actions, rules_applied = all_expand(env.s_current.ex, theory)
    tmp = filter(x->x!=env.s_current, all_actions)
    new_nodes = map(x->Node(x[1], x[2], env.s_current.node_id, env.s_current.depth + 1), zip(tmp, rules_applied))
    new_nodes = filter(x->push_to_tree!(env.soltree, x), new_nodes)
    return new_nodes
end
function reward(env::MyTreeEnv)
    size_current = exp_size(env.s_current.ex)
    size_init = exp_size(env.s_init.ex)
    if size_current >= size_init
        return -1
    else
        return size_init - size_current
    end
end
function act!(env::MyTreeEnv, a)
    env.s_current = a
end

function train(embedding_model, online_q, target_q, data; max_iterations=10, max_steps=100, epsilon=1.0, batch_size=128, lr=0.001, buffer_size=1_000_000)
    env = MyTreeEnv(data[1], embedding_model)
    pol_optimizer = ADAM(lr)
    online_q_params = Flux.setup(pol_optimizer, online_q)
    MyModule.reset_all_function_caches()
    target_update_freq = 0
    my_buffer = MyReplayBuffer(buffer_size)
    best_solutions = [[intern!(i), intern!(i), exp_size(i)] for i in data]
    progress_log = []
    start = time()
    duration = 10 * 60  # 10 minutes in seconds

    while time() - start < duration
    # t1 = @elapsed for iter in 1:max_iterations
        t2 = @elapsed for (ind, ex) in enumerate(data)
            env.s_init = intern!(ex)
            reset!(env)
            # @show env.s_init
            # debug_vec = [(intern!(ex), 0)]

            rewards = 0
            t3 = @elapsed for i in 1:max_steps
                if epsilon <= rand()
                    possible_actions = action_space(env)
                    o = map(possible_actions) do i
                        only(env.policy_model(i))
                    end
                    a = possible_actions[argmax(o)]
                else
                    a = rand(action_space(env))
                end
                s = state(env)
                act!(env, a)
                r = reward(env)
                # push!(debug_vec, (a, r))
                rewards += r
                ns = state(env)
                add2buffer!(my_buffer, (s, r, ns))
                target_update_freq += 1
                if length(my_buffer.buffer) >= batch_size
                    update_ddqn!(my_buffer, online_q, target_q, online_q_params, batch_size=batch_size)
                    if mod(target_update_freq, 10) == 0
                        # soft_update!(online_q, target_q, 0.5)
                        target_q = deepcopy(online_q)
                    end
                end
                if best_solutions[ind][3] > exp_size(env.s_current)
                    best_solutions[ind][3] = exp_size(env.s_current)
                    best_solutions[ind][2] = env.s_current
                end
                if isempty(action_space(env))
                    break
                end
            end
            # @show t3
        end
        @show t2
        epsilon = update_epsilon(epsilon)
        @show epsilon
        full_rewards = 0
        for bs in best_solutions
            println("Iter $(iter):")
            println("Initial $(bs[1])")
            println("Best $(bs[2])")
            full_rewards += abs(exp_size(bs[2]) - exp_size(bs[1]))
        end
        push!(progress_log, full_rewards / length(data))
        @show full_rewards
        # println("Iter $(iter): rewards=$(rewards)")
    end
    @show t1
    final_rewards = 0
    for bs in best_solutions
        println("Initial $(bs[1])")
        println("Best $(bs[2])")
        println("Reward $(abs(exp_size(bs[2]) - exp_size(bs[1])))")
        final_rewards += abs(exp_size(bs[2]) - exp_size(bs[1]))
    end
    @show final_rewards
    return best_solutions, progress_log
end
function train2(embedding_model, online_q, target_q, data; max_iterations=10, max_steps=100, epsilon=1.0, batch_size=128, lr=0.001, buffer_size=1_000_000)
    env = MyTreeEnv(data[1], embedding_model)
    pol_optimizer = ADAM(lr)
    online_q_params = Flux.setup(pol_optimizer, online_q)
    MyModule.reset_all_function_caches()
    target_update_freq = 0
    my_buffer = MyReplayBuffer(buffer_size)
    best_solutions = [[intern!(i), intern!(i), exp_size(i)] for i in data]
    progress_log = []
    start = time()
    duration = 10 * 60  # 10 minutes in seconds
    iter = 0
    while time() - start < duration
    # t1 = @elapsed for iter in 1:max_iterations
        iter += 1
        t2 = @elapsed for (ind, ex) in enumerate(data)
            env.s_init = intern!(ex)
            reset!(env)
            # @show env.s_init
            # debug_vec = [(intern!(ex), 0)]

            rewards = 0
            t3 = @elapsed for i in 1:max_steps
                possible_actions = action_space(env)
                if isempty(possible_actions)
                    break
                end
                if epsilon <= rand()
                    o = map(possible_actions) do i
                        only(env.policy_model(i))
                    end
                    a = possible_actions[argmax(o)]
                else
                    a = rand(possible_actions)
                end
                s = state(env)
                act!(env, a)
                r = reward(env)
                # push!(debug_vec, (a, r))
                rewards += r
                ns = state(env)
                add2buffer!(my_buffer, (s, r, ns))
                target_update_freq += 1
                if length(my_buffer.buffer) >= batch_size
                    update_ddqn!(my_buffer, online_q, target_q, online_q_params, batch_size=batch_size)
                    if mod(target_update_freq, 10) == 0
                        # soft_update!(online_q, target_q, 0.5)
                        target_q = deepcopy(online_q)
                    end
                end
                if best_solutions[ind][3] > exp_size(env.s_current.ex)
                    best_solutions[ind][3] = exp_size(env.s_current.ex)
                    best_solutions[ind][2] = env.s_current.ex
                end
                
            end
            # @show t3
        end
        @show t2
        epsilon = update_epsilon(epsilon)
        @show epsilon
        full_rewards = 0
        for bs in best_solutions
            println("Iter $(iter):")
            println("Initial $(bs[1])")
            println("Best $(bs[2])")
            full_rewards += abs(exp_size(bs[2]) - exp_size(bs[1]))
        end
        push!(progress_log, full_rewards / length(data))
        @show full_rewards
        # println("Iter $(iter): rewards=$(rewards)")
    end
    @show t1
    final_rewards = 0
    for bs in best_solutions
        println("Initial $(bs[1])")
        println("Best $(bs[2])")
        println("Reward $(abs(exp_size(bs[2]) - exp_size(bs[1])))")
        final_rewards += abs(exp_size(bs[2]) - exp_size(bs[1]))
    end
    @show final_rewards
    return best_solutions, progress_log
end
function train3(embedding_model, online_q, target_q, data; max_iterations=10, max_steps=100, epsilon=1.0, batch_size=128, lr=0.001, buffer_size=1_000_000)
    # env = MyTreeEnv(data[1], embedding_model)
    env = EGraphEnv(data[1], online_q; max_expansions=max_steps)
    pol_optimizer = ADAM(lr)
    online_q_params = Flux.setup(pol_optimizer, online_q)
    MyModule.reset_all_function_caches()
    target_update_freq = 0
    my_buffer = MyReplayBuffer(buffer_size)
    progress_log = []
    t1 = @elapsed for iter in 1:max_iterations
        best_solutions = [[intern!(i), intern!(i), exp_size(i)] for i in data]
        t2 = @elapsed for (ind, ex) in enumerate(data)
            env.s_init = ex
            reset!(env)
            # @show env.s_init
            # debug_vec = [(intern!(ex), 0)]

            rewards = 0
            t3 = @elapsed for i in 1:max_steps
                possible_actions = action_space(env)
                if isempty(possible_actions)
                    break
                end
                if epsilon <= rand()
                    o = map(possible_actions) do i
                        only(env.policy_model(i))
                    end
                    a = possible_actions[argmax(o)]
                else
                    a = rand(possible_actions)
                end
                s = state(env)
                act!(env, a)
                r = reward(env)
                # push!(debug_vec, (a, r))
                rewards += r
                ns = state(env)
                add2buffer!(my_buffer, (s, r, ns))
                target_update_freq += 1
                if length(my_buffer.buffer) >= batch_size
                    update_ddqn!(my_buffer, online_q, target_q, online_q_params, batch_size=batch_size)
                    if mod(target_update_freq, 10) == 0
                        # soft_update!(online_q, target_q, 0.5)
                        target_q = deepcopy(online_q)
                    end
                end
                if best_solutions[ind][3] > exp_size(env.s_current.ex)
                    best_solutions[ind][3] = exp_size(env.s_current.ex)
                    best_solutions[ind][2] = env.s_current.ex
                end
                
            end
            # @show t3
        end
        @show t2
        epsilon = update_epsilon(epsilon)
        @show epsilon
        full_rewards = 0
        for bs in best_solutions
            println("Iter $(iter):")
            println("Initial $(bs[1])")
            println("Best $(bs[2])")
            full_rewards += abs(exp_size(bs[2]) - exp_size(bs[1]))
        end
        push!(progress_log, full_rewards / length(data))
        @show full_rewards / length(best_solutions)
        # println("Iter $(iter): rewards=$(rewards)")
    end
    @show t1
    # final_rewards = 0
    # for bs in best_solutions
    #     println("Initial $(bs[1])")
    #     println("Best $(bs[2])")
    #     println("Reward $(abs(exp_size(bs[2]) - exp_size(bs[1])))")
    #     final_rewards += abs(exp_size(bs[2]) - exp_size(bs[1]))
    # end
    # @show final_rewards
    return best_solutions, progress_log
end
max_iterations=100
max_steps=100
batch_size=128
lr=0.0001
buffer_size=1_00_000

best_solutions, progress_log = train2(embedding_model, online_q, target_q, data, max_iterations=max_iterations, max_steps=max_steps, epsilon=1.0, batch_size=batch_size, lr=lr, buffer_size=buffer_size)
# best_solutions, progress_log = train(embedding_model, online_q, target_q, data, max_iterations=max_iterations, max_steps=max_steps, epsilon=1.0, batch_size=batch_size, lr=lr, buffer_size=buffer_size)
# plot(progress_log, xlabel="Iterations", ylabel="Averege expression simplification")
# savefig("stats/ddql_averege_exp_simplification_$(max_iterations)iter_$(max_steps)maxsteps_$(batch_size)batchsize_$(buffer_size)buffsize_$(lr)lr.png")
# serialize("models/ddql_online_network_$(max_iterations)iter_$(max_steps)maxsteps_$(batch_size)batchsize_$(buffer_size)buffsize_$(lr)lr.bin", online_q)
# serialize("models/ddql_target_network_$(max_iterations)iter_$(max_steps)maxsteps_$(batch_size)batchsize$(buffer_size)buffsize_$(lr)lr.bin", target_q)
