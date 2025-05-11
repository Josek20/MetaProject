using CSV
using DataFrames
using Plots
using Optimisers
using Statistics
include("my_replay_buffer.jl")
using MyModule
using MyModule.Flux
using MyModule.Mill
using MyModule: all_expand, exp_size, Node, NodeID, intern!
include("tree_env_setup.jl")
using Serialization


# hidden_size=Meta.parse(ARGS[7])
hidden_size=64
gamma = 0.99
# epsilone_decay = 0.9 / 2.5e2
epsilon = eps_start = 1.0
eps_end = 0.01
eps_decay = 0.95
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
# if ARGS[6] == "1"
online_q = Chain(Dense(input_size, 1))
target_q = Chain(Dense(input_size, 1))
target_q = deepcopy(online_q)
# elseif ARGS[6] == "2"
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
    mean_sa = 0
    for k in 1:10
        sa, grad = Flux.Zygote.withgradient(online_q) do oq
            expected_values = vec(oq(states))
            loss = mean((target_values - expected_values) .^ 2)
            return loss
        end
        Optimisers.update!(policy_params_optimiser, online_q, grad[1])
        mean_sa += sa
        # println("TEp $(k): $(sa)")
    end
    return mean_sa / 10
end

update_epsilon(epsilon; eps_decay=0.95, eps_min=0.1) = max(eps_min, eps_decay * epsilon)

mutable struct MyTreeEnv
    s_init
    s_current
    t
    is_done
    policy_model
end
function MyTreeEnv(ex::Expr, policy_model)
    inex = intern!(ex)
    return MyTreeEnv(inex, inex, 1, false, policy_model)
end
function state(env::MyTreeEnv)
    inference_type = MyModule.get_inference_type(env.s_current)
    ds = MyModule.general_cached_inference(env.s_current, inference_type, env.policy_model)
    return vec(ds)
end
function reset!(env::MyTreeEnv)
    env.s_current = env.s_init
    env.t = 1
    return env
end
function action_space(env::MyTreeEnv)
    env.t += 1
    #new_ex, _ = all_expand(env.s_current, theory)
    all_actions = first(MyModule.all_expand(env.s_current, theory))
    tmp = filter(x->x!=env.s_current, all_actions)
    return tmp
end
function reward(env::MyTreeEnv)
    size_current = exp_size(env.s_current)
    size_init = exp_size(env.s_init)
    if size_current >= size_init
        return -1
    else
        return size_init - size_current
    end
end
function act!(env::MyTreeEnv, a)
    env.s_current = a
end
is_terminal(env::MyTreeEnv) = env.is_done
function validation(data, embedding_model, name)
    env = MyTreeEnv(data[1], embedding_model)
    MyModule.reset_all_function_caches()
    best_solutions = [[intern!(i), intern!(i), exp_size(i)] for i in data]
    t2 = @elapsed for (ind, ex) in enumerate(data)
        env.s_init = intern!(ex)
        reset!(env)
        # @show env.s_init
        debug_vec = [(intern!(ex), 0)]

        rewards = 0
        t3 = @elapsed for i in 1:max_steps
            possible_actions = action_space(env)
            o = map(possible_actions) do i
                only(env.policy_model(i))
            end
            a = possible_actions[argmax(o)]
            s = state(env)
            act!(env, a)
            r = reward(env)
            rewards += r
            ns = state(env)
            is_done = env.is_done
            if best_solutions[ind][3] > exp_size(env.s_current)
                best_solutions[ind][3] = exp_size(env.s_current)
                best_solutions[ind][2] = env.s_current
            end
            if isempty(action_space(env)) || is_done
                break
            end
        end
    end
	if name == ""
		final_rew = 0
		for bs in best_solutions
		    final_rew += abs(exp_size(bs[2]) - exp_size(bs[1]))
		end
		@show final_rew / length(best_solutions)
	else
		df = DataFrame(InitEx = Any[], RedEx = Any[], RedDif = Any[])
		for bs in best_solutions
		    println("Initial $(bs[1])")
		    println("Best $(bs[2])")
		    println("Reward $(abs(exp_size(bs[2]) - exp_size(bs[1])))")
		    fin_rew = abs(exp_size(bs[2]) - exp_size(bs[1]))
		    push!(df, (InitEx = MyModule.expr(MyModule.nc, bs[1]), RedEx = MyModule.expr(MyModule.nc, bs[2]), RedDif = fin_rew))
		end
		CSV.write("stats/dql_$(name)_1h.csv", df)
	end
end
function train(embedding_model, online_q, target_q, data; max_iterations=100, max_steps=100, epsilon=1.0, batch_size=128, lr=0.001, buffer_size=1_000_000, target_update=10)
    env = MyTreeEnv(data[1], embedding_model)
    pol_optimizer = ADAM(lr)
    online_q_params = Flux.setup(pol_optimizer, online_q)
    best_solutions = [[intern!(i), intern!(i), exp_size(i)] for i in data]

    MyModule.reset_all_function_caches()
    target_update_freq = 0
    my_buffer = MyReplayBuffer(buffer_size)
    progress_log = []
    start = time()
    duration = 240 * 60  # 10 minutes in seconds
    iter = 0
    #while time() - start < duration
    t1 = @elapsed for iter in 1:max_iterations
	    best_solutions = map(best_solutions) do i
            [i[1], i[1], exp_size(i[1])]
        end
    	#iter += 1
        t2 = @elapsed for (ind, ex) in enumerate(data)
            env.s_init = intern!(ex)
            reset!(env)
            # @show env.s_init
            debug_vec = []
            
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
                    a = rand(action_space(env))
                end
                s = state(env)
                s0 = env.s_current
                act!(env, a)
                # @show env.s_current
                r = reward(env)
                # push!(debug_vec, (a, r))
                rewards += r
                ns = state(env)
		        is_done = env.is_done
                # add2buffer!(my_buffer, (s, r, ns))
                push!(debug_vec, (s, r, ns, s0))
                target_update_freq += 1
                if length(my_buffer.buffer) >= batch_size
                    update_ddqn!(my_buffer, online_q, target_q, online_q_params, batch_size=batch_size)
                    if mod(target_update_freq, target_update) == 0
                        # soft_update!(online_q, target_q, 0.5)
                        target_q = deepcopy(online_q)
			            target_update_freq = 0
                    end
                end
                if best_solutions[ind][3] > exp_size(env.s_current)
                    best_solutions[ind][3] = exp_size(env.s_current)
                    best_solutions[ind][2] = env.s_current
                end
                if isempty(action_space(env)) || is_done
                    break
                end
            end
            _, min_ind = findmin(x->exp_size(x[end]) - exp_size(env.s_init), debug_vec)
            for (ind, i) in enumerate(debug_vec)
                if ind == min_ind
                    break
                end
                s, r, ns, _ = i
                add2buffer!(my_buffer, (s, r, ns))
            end
        end
        @show t2
        epsilon = update_epsilon(epsilon)
        @show epsilon
        full_rewards = 0
        for bs in best_solutions
            # println("Iter $(iter):")
            # println("Initial $(bs[1])")
            # println("Best $(bs[2])")
            full_rewards += abs(exp_size(bs[2]) - exp_size(bs[1]))
        end
        push!(progress_log, full_rewards / length(data))
        @show full_rewards / length(best_solutions)
        # println("Iter $(iter): rewards=$(rewards)")
    end
    return progress_log, env.policy_model
end
# max_iterations=Meta.parse(ARGS[1])
# max_steps=Meta.parse(ARGS[2])
# batch_size=Meta.parse(ARGS[3])
# buffer_size=Meta.parse(ARGS[4])
# lr = Meta.parse(ARGS[5])
max_iterations=1
max_steps=50
batch_size=128
buffer_size=100_000
lr = 0.001
println("maxiter=$(max_iterations),maxsteps=$(max_steps),batchsize=$(batch_size),buffersize=$(buffer_size),lr=$(lr)")

progress_log, new_embedding_model = train(embedding_model, online_q, target_q, data, max_iterations=max_iterations, max_steps=max_steps, epsilon=1.0, batch_size=batch_size, lr=lr, buffer_size=buffer_size)
plot(progress_log, xlabel="Iterations", ylabel="Averege expression simplification")
# savefig("stats/new_ddql_averege_exp_simplification_$(max_iterations)iter_$(max_steps)maxsteps_$(batch_size)batchsize_$(buffer_size)buffsize_$(lr)lr.png")



# experiment_name = "dql_new_$(max_iterations)"
# train_data_path = "./data/neural_rewrter/test.json"
# train_data = load_data(train_data_path)[1:1_000]
# train_data = filter(x->!occursin("select", x[1]), train_data)
# train_data = preprosses_data_to_expressions(train_data)
# # sorted_data = sort(train_data, by=x->MyModule.exp_size(x))
# test_data = sorted_data


# validation(data, new_embedding_model, "train")
# validation(test_data, new_embedding_model, "test")
