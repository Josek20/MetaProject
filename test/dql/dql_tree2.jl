using CSV
# using D3Trees
using DataFrames
using Plots
using Optimisers
using Statistics
include("my_replay_buffer.jl")
using MyModule
using MyModule.Flux
using MyModule.Mill
using MyModule: all_expand, exp_size, Node, NodeID, intern!, expr
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
# online_q = Chain(Dense(input_size, 1))
# target_q = Chain(Dense(input_size, 1))
# target_q = deepcopy(online_q)
# elseif ARGS[6] == "2"
online_q = Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, 1))
target_q = Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, 1))
target_q = deepcopy(online_q)
# else
#     online_q = Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, hidden_size, relu), Dense(hidden_size, 1))
#     target_q = Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, hidden_size, relu), Dense(hidden_size, 1))
#     target_q = deepcopy(online_q)
# end

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
function update_ddqn1!(buffer, online_q, target_q, policy_params_optimiser; batch_size=64, gamma=0.995, max_iter=1)
    # sampled_experiences = sample(buffer, batch_size)
    sampled_experiences = buffer
    if length(sampled_experiences) < 2
        return
    end
    states = [i[1] for i in sampled_experiences[1:end-1]]
    rewards = [i[2] for i in sampled_experiences[1:end-1]]
    next_states = [i[3] for i in sampled_experiences[1:end-1]]
    next_next_states = [i[3] for i in sampled_experiences[2:end]]
    # @show length(states), length(next_states), length(rewards), length(next_next_states)
    @assert length(states) == length(next_states) == length(rewards) == length(next_next_states)
    
    next_states = hcat(next_states...)
    next_next_states = hcat(next_next_states...)
    ns = vcat(next_states, next_next_states)
    
    states = hcat(states...)
    s = vcat(states, next_states)
    target_values = rewards + gamma * vec(target_q(ns))
    
    mean_sa = 0
    for k in 1:max_iter
        sa, grad = Flux.Zygote.withgradient(online_q) do oq
            expected_values = vec(oq(s))
            loss = mean((target_values - expected_values) .^ 2)
            return loss
        end
        Optimisers.update!(policy_params_optimiser, online_q, grad[1])
        mean_sa += sa
        # println("TEp $(k): $(sa)")
    end
    return mean_sa / max_iter
end 
function update_ddqn!(buffer, online_q, target_q, policy_params_optimiser; batch_size=64, gamma=0.995, max_iter=1)
    sampled_experiences = sample(buffer, batch_size)
    # sampled_experiences = buffer
    # sampled_experiences = vcat(values(buffer)...)
    states = [expr(MyModule.nc, i[1]) for i in sampled_experiences]
    rewards = [i[2] for i in sampled_experiences]
    next_states = [expr(MyModule.nc, i[3]) for i in sampled_experiences]
    next_possible_states = [[expr(MyModule.nc, j) for j in i[5]] for i in sampled_experiences]
    # is_dones = [i[4] for i in sampled_experiences]
    states = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(states, sym_enc))
    # next_states = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(next_states, sym_enc))
    next_states = MyModule.deduplicate.(MyModule.no_reduce_multiple_fast_ex2mill.(next_possible_states))
    next_q = [maximum(MyModule.heuristic(target_q, i)) for i in next_states]
    target_values = rewards + gamma * next_q
    # target_values = rewards + gamma * vec(MyModule.heuristic(target_q, next_states))# .* (1 .- is_dones) 
    
    # states = hcat(states...)
    mean_sa = 0
    for k in 1:max_iter
        sa, grad = Flux.Zygote.withgradient(online_q) do oq
            expected_values = vec(MyModule.heuristic(oq, states))
            loss = sum((target_values - expected_values) .^ 2)
            return loss
        end
        Optimisers.update!(policy_params_optimiser, online_q, grad[1])
        mean_sa += sa
    end
    return mean_sa / max_iter
end
function update_ddqn3!(buffer, online_q, target_q, policy_params_optimiser; batch_size=64, gamma=0.995, max_iter=1)
    # sampled_experiences = sample(buffer, batch_size)
    sampled_experiences = buffer
    # sampled_experiences = vcat(values(buffer)...)
    states = [expr(MyModule.nc, i[1]) for i in sampled_experiences]
    rewards = [i[2] for i in sampled_experiences]
    next_states = [expr(MyModule.nc, i[3]) for i in sampled_experiences]
    possible_next_states = [expr(MyModule.nc, i[5]) for i in sampled_experiences]
    # is_dones = [i[4] for i in sampled_experiences]
    states = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(states, sym_enc))
    next_states = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(next_states, sym_enc))
    possible_next_states = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(possible_next_states, sym_enc))

    # target_values = rewards + gamma * vec(MyModule.heuristic(target_q, next_states))# .* (1 .- is_dones)
    next_q = maximum(MyModule.heuristic(target_q, possible_next_states))
    # target_values = rewards + gamma * vec(MyModule.heuristic(target_q, next_states[2:end]))# .* (1 .- is_dones)
    target_values = rewards + gamma * next_q
    # x->a->w
    # states = hcat(states...)
    mean_sa = 0
    for k in 1:max_iter
        sa, grad = Flux.Zygote.withgradient(online_q) do oq
            expected_values = vec(MyModule.heuristic(oq, next_states))
            # loss = mean((target_values - expected_values) .^ 2)
            loss = Flux.mse(target_values, expected_values)
            return loss
        end
        Optimisers.update!(policy_params_optimiser, online_q, grad[1])
        mean_sa += sa
    end
    return mean_sa / max_iter
end
update_epsilon(epsilon; eps_decay=0.95, eps_min=0.1) = max(eps_min, eps_decay * epsilon) == eps_min ? 0.0 : max(eps_min, eps_decay * epsilon)

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
    # inference_type = MyModule.get_inference_type(env.s_current)
    # ds = MyModule.general_cached_inference(env.s_current, inference_type, env.policy_model)
    # return vec(ds)
    return env.s_current
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
    # if size_current >= size_init
    #     return -1
    # else
    return size_init - size_current
    # end
end
function reward(env::MyTreeEnv, a, s)
    size_current = exp_size(a)
    size_init = exp_size(s)
    return size_init - size_current
end
function reward1(env::MyTreeEnv, a, s)
    size_current = exp_size(a)
    # size_init = exp_size(s)
    return -size_current
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
# function train(online_q, target_q, data; max_iterations=100, max_steps=100, epsilon=1.0, batch_size=128, lr=0.001, buffer_size=1_000_000, target_update=20)
function train12(embedding_model, online_q, target_q, data; max_iterations=100, max_steps=100, epsilon=1.0, batch_size=128, lr=0.001, buffer_size=1_000_000, target_update=10)
    # env = MyTreeEnv(data[1], online_q)
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
    tmp = []
    best_buffer = Dict(i=>Tuple[] for i in 1:length(data))
    new_best_buffer = Dict(i=>Tuple[] for i in 1:length(data))
    # best_buffer[1] = [(intern!(:(116 - 100 <= 1024)), 2, intern!(:(16 <= 1024)), false), (intern!(:(16 <= 1024)), 2, intern!(:(1)), true)]
    visulization_buffer = []
    #while time() - start < duration
    avg_solutionv = []
    avg_nodes = []
    avg_solutionsl = []
    t1 = @elapsed for iter in 1:max_iterations
	    best_solutions = map(best_solutions) do i
            [i[1], i[1], exp_size(i[1])]
        end
    	#iter += 1
        solutionv = []               
        nodes = []                   
        solutionsl = []
        t2 = @elapsed for (ind, ex) in enumerate(data)
            env.s_init = intern!(ex)
            reset!(env)
            # @show env.s_init
            debug_vec = []
            stmp = []
            vis_buff = []
            rewards = 0
            t3 = @elapsed for i in 1:max_steps
                possible_actions = action_space(env)
                if isempty(possible_actions)
                    break
                end
                o = zeros(Float32, length(possible_actions))
                max_ind = 0
                if epsilon <= rand()
                    o = map(possible_actions) do i
                        only(env.policy_model(i))
                        # current_emb = state(env)
                        # inference_type = MyModule.get_inference_type(i)
                        # next_emb = MyModule.general_cached_inference(i, inference_type, env.policy_model)
                        # q = online_q(vcat(current_emb, vec(next_emb)))
                        # return only(q)
                    end
                    max_ind = argmax(o)
                    a = possible_actions[max_ind]
                else
                    max_ind = rand(1:length(possible_actions))
                    o[max_ind] = 99.99
                    a = possible_actions[max_ind]
                end
                s = state(env)
                s0 = env.s_current
                act!(env, a)
                # @show env.s_current
                r = reward(env, a, s0)
                # r = reward(env)
                # push!(debug_vec, (a, r))
                rewards += r
                ns = state(env)
		        is_done = env.is_done || i == max_steps || isempty(action_space(env))
                add2buffer!(my_buffer, (s, r, ns, is_done, possible_actions))
                push!(debug_vec, (s, a, r, ns))
                push!(vis_buff, (s0, a, possible_actions, max_ind, r, o))
                # push!(new_best_buffer[ind], (s, r, ns))
                # push!(new_best_buffer[ind], (s, r, ns))
                target_update_freq += 1
                if length(my_buffer.buffer) >= batch_size
                    # best_buffer
                    # ml = update_ddqn1!(my_buffer, online_q, target_q, online_q_params, batch_size=batch_size)
                    # ml = update_ddqn!(my_buffer.buffer, online_q, target_q, online_q_params, batch_size=batch_size)
                    ml = update_ddqn!(my_buffer, online_q, target_q, online_q_params, batch_size=batch_size,max_iter=10)
                    push!(tmp, ml)
                    if mod(target_update_freq, target_update) == 0
                        # soft_update!(online_q, target_q, 0.5)
                        target_q = deepcopy(online_q)
			            target_update_freq = 0
                    end
                end
                # update_time = @elapsed update_ddqn3!(best_buffer, online_embedding_model, target_embedding_model, online_q_params, target_update_freq)
	            # @show update_time
                if best_solutions[ind][3] > exp_size(env.s_current)
                    best_solutions[ind][3] = exp_size(env.s_current)
                    best_solutions[ind][2] = env.s_current
                end
                if isempty(action_space(env)) || is_done
                    break
                end
            end
            push!(visulization_buffer, vis_buff)
            _, min_ind = findmin(x->exp_size(x[2]) - exp_size(env.s_init), debug_vec)
            push!(nodes, length(Set(debug_vec)))                    
            push!(solutionsl, length(debug_vec[1:min_ind]))         
            push!(solutionv, exp_size(env.s_init) - exp_size(debug_vec[1:min_ind][end][2]))
           
            if length(best_buffer[ind]) == 0
                best_buffer[ind] = debug_vec[1:min_ind]
            elseif exp_size(best_buffer[ind][end][3]) > exp_size(debug_vec[1:min_ind][end][3])
                best_buffer[ind] = debug_vec[1:min_ind]
            elseif exp_size(best_buffer[ind][end][3]) == exp_size(debug_vec[1:min_ind][end][3]) && length(debug_vec[1:min_ind]) < length(best_buffer[ind])
                best_buffer[ind] = debug_vec[1:min_ind]
            end
            # q = vcat(vcat(values(best_buffer)...), debug_vec[1:min_ind])
            # q = debug_vec[1:min_ind]
            # q = vcat(values(best_buffer)...)
            # target_update_freq += 1
            # ml = update_ddqn!(q, online_q, target_q, online_q_params, batch_size=batch_size, max_iter=1)
            # push!(tmp, ml)
            # if mod(target_update_freq, target_update) == 0
            #     target_q = deepcopy(online_q)
            #     target_update_freq = 0
            # end
        end
        push!(avg_nodes, mean(nodes))
        push!(avg_solutionsl, mean(solutionsl))
        push!(avg_solutionv, mean(solutionv))
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
    p1 = plot(avg_nodes, xlabel="Expression ID", ylabel="Number of unique explored nodes", legend=false)
    p2 = plot(avg_solutionsl, xlabel="Expression ID", ylabel="Best solution legnth", legend=false)
    p3 = plot(avg_solutionv, xlabel="Expression ID", ylabel="Best solution simplification", legend=false)
    plot(p1, p2, p3, layout=(3,1))
    savefig("trained_dqn_stats_avg1.png")
    return progress_log, env.policy_model, tmp, best_solutions, best_buffer, visulization_buffer, new_best_buffer
end
# max_iterations=Meta.parse(ARGS[1])
# max_steps=Meta.parse(ARGS[2])
# batch_size=Meta.parse(ARGS[3])
# buffer_size=Meta.parse(ARGS[4])
# lr = Meta.parse(ARGS[5])
max_iterations=100
max_steps=100
batch_size=50000
buffer_size=1_00_000
lr = 0.003
println("maxiter=$(max_iterations),maxsteps=$(max_steps),batchsize=$(batch_size),buffersize=$(buffer_size),lr=$(lr)")

# progress_log, new_embedding_model, tmp, best_solutions, best_buffer, visulization_buffer = train(online_embedding_model, online_q, target_q, data, max_iterations=max_iterations, max_steps=max_steps, epsilon=1.0, batch_size=batch_size, lr=lr, buffer_size=buffer_size)
# progress_log, new_embedding_model, tmp, best_solutions, best_buffer, visulization_buffer, new_best_buffer = train(online_embedding_model, online_embedding_model, target_embedding_model, data[1:10], max_iterations=max_iterations, max_steps=max_steps, epsilon=1.0, batch_size=batch_size, lr=lr, buffer_size=buffer_size)
progress_log, new_embedding_model, tmp, best_solutions, best_buffer, visulization_buffer, new_best_buffer = train12(online_policy, online_policy, target_policy, data[1:1], max_iterations=1, max_steps=10, epsilon=0.0, batch_size=batch_size, lr=lr, buffer_size=buffer_size)
# avg_solutionv = []               
# avg_nodes = []                   
# avg_solutionsl = []
# for _ in 1:10
#     solutionv = []               
#     nodes = []                   
#     solutionsl = []
#     for i in 1:length(data)
#         push!(nodes, length(Set(new_best_buffer[i])))                    
#         short_sol = []               
#         _, min_ind = findmin(x->exp_size(x[end]) - exp_size(intern!(data[i])), new_best_buffer[i])           
#         push!(solutionsl, length(new_best_buffer[i][1:min_ind]))         
#         push!(solutionv, exp_size(intern!(data[i])) - exp_size(new_best_buffer[i][1:min_ind][end][end]))     
#     end
#     push!(avg_nodes, nodes)
#     push!(avg_solutionv, solutionv)
#     push!(avg_solutionsl, solutionsl)
# end
# p1 = plot(mean(avg_nodes), xlabel="Expression ID", ylabel="Number of unique explored nodes", legend=false)
# p2 = plot(mean(avg_solutionsl), xlabel="Expression ID", ylabel="Best solution legnth", legend=false)
# p3 = plot(mean(avg_solutionv), xlabel="Expression ID", ylabel="Best solution simplification", legend=false)
# plot(p1, p2, p3, layout=(3,1))
# savefig("random_dqn_stats_avg10.png")
function visulization(visulization_buffer)
    # using D3Trees
    single_sample = visulization_buffer[end]
    children = Vector[]
    text = []
    link_style = [""]
    style = [""]
    index_to_put = 0
    current_range = 1:1
    for (s0, a, possible_actions, max_ind, r, o) in single_sample
        if length(children) == 0
            push!(text, string(expr(MyModule.nc, s0)))
            current_range = 2:length(possible_actions) + 1
            push!(children, collect(Any, current_range))
        else
            current_range = current_range.stop + 1:length(possible_actions) + current_range.stop
            children[index_to_put] = collect(Any, current_range)
        end
        append!(text, [string(expr(MyModule.nc, i)) * "\nPred:$(round(k, digits=4))\nRew:$(ind == max_ind ? r : 0)" for (ind,(i, k)) in enumerate(zip(possible_actions, o))])
        index_to_put = length(children) + max_ind
        for k in 1:length(possible_actions)
            push!(children, [])
            if k == max_ind
                push!(link_style, "stroke:blue")
                push!(style, "fill:green")
            else
                push!(link_style, "")
                push!(style, "")
            end
        end
    end
    t = D3Tree(children, text=text, style=style, link_style=link_style, init_expand=2,  svg_node_size=(2020, 2020))
    inbrowser(t, "Mircosoft Edge")


    # io = IOBuffer()
    # show(io, MIME"text/html"(), tree)
    # html_string = String(take!(io))

    # # Save it to a file
    # open("stats/dql_tree_vis/tree_output.html", "w") do f
    #     write(f, html_string)
    # end
    return t
end
visulization(visulization_buffer)
# p1 = plot(progress_log, xlabel="Iterations", ylabel="Averege expression simplification")
# p2 = plot(tmp)
# plot(p1, p2, layout=(2,1))
# savefig("stats/new_ddql_averege_exp_simplification_$(max_iterations)iter_$(max_steps)maxsteps_$(batch_size)batchsize_$(buffer_size)buffsize_$(lr)lr.png")
# visulization(visulization_buffer)


# experiment_name = "dql_new_$(max_iterations)"
# train_data_path = "./data/neural_rewrter/test.json"
# train_data = load_data(train_data_path)[1:1_000]
# train_data = filter(x->!occursin("select", x[1]), train_data)
# train_data = preprosses_data_to_expressions(train_data)
# # sorted_data = sort(train_data, by=x->MyModule.exp_size(x))
# test_data = sorted_data


# validation(data, new_embedding_model, "train")
# validation(test_data, new_embedding_model, "test")


function test_manual_path()
    ex = intern!(:(116 - 100 <= 1024))
    new_nodes = action_space(env)
    buff = []
    for i in new_nodes[1:end-1]
        size_current = exp_size(i)
        size_init = exp_size(ex)
        r = size_init - size_current
        push!(buff, (ex, r, i, 0))
    end
    size_current = exp_size(new_nodes[end])
    size_init = exp_size(ex)
    r = size_init - size_current
    push!(buff, (ex, r, new_nodes[end], 0))
    pol_optimizer = ADAM(lr)
    online_q_params = Flux.setup(pol_optimizer, online_embedding_model)
    update_ddqn!(buff, online_embedding_model, target_embedding_model, online_q_params; batch_size=64, gamma=0.995, max_iter=1)
end


# planning_model = ExprModel(
#     head_model,
#     Mill.SegmentedSum(hidden_size),
#     args_model,
#     Flux.Chain(Dense(hidden_size, hidden_size, Flux.gelu), Dense(hidden_size, hidden_size, Flux.gelu), Dense(hidden_size, 1)),
#     );
# planning_model = deserialize("models/trained_heuristic_test_heuristic_boosted_1h_ep10_hidden64.bin")
function test_planning_solution()
    # get the planning solution
    pol_optimizer = ADAM(lr)
    online_q_params = Flux.setup(pol_optimizer, online_embedding_model)
    update_ddqn!(buff, online_embedding_model, target_embedding_model, online_q_params; batch_size=64, gamma=0.995, max_iter=1)
end