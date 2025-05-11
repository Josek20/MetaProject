using ReinforcementLearning
using Flux
using Plots
using Optimisers
using Statistics

# env = CartPoleEnv()
# env = PendulumEnv()
# env = MountainCarEnv()
input_size = 3
output_size = 3
hidden_size = 32
online_q = Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, hidden_size*2, relu), Dense(hidden_size*2, output_size))
target_q = Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, hidden_size*2, relu), Dense(hidden_size*2, output_size))
target_q = deepcopy(online_q)
mutable struct MyReplayBuffer
    capacity::Int
    buffer::Vector{Tuple{Array{Float32}, Int, Float32, Array{Float32}, Bool}}
    pointer::Int
end

MyReplayBuffer(capacity) = MyReplayBuffer(capacity, Vector{Tuple{Array{Float32}, Int, Float32, Array{Float32}, Bool}}(), 1)

function add2buffer!(buffer::MyReplayBuffer, experience)
    if buffer.capacity <= length(buffer.buffer)
        buffer.buffer[buffer.pointer] = experience
        buffer.pointer = (buffer.pointer % buffer.capacity) + 1
    else
        push!(buffer.buffer, experience)
    end
end

function sample(buffer::MyReplayBuffer, batch_size::Int)
    sampled_indices = rand(1:length(buffer.buffer), batch_size)
    # return [buffer.buffer[i] for i in sampled_indices]
    return buffer.buffer[sampled_indices]
end

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
    actions = [i[2] for i in sampled_experiences]
    rewards = [i[3] for i in sampled_experiences]
    next_states = [i[4] for i in sampled_experiences]
    dones = [i[5] for i in sampled_experiences]
    
    
    # next_states = hcat(next_states...)
    # target_values = rewards + gamma * (vec(maximum(target_q(next_states), dims=1)) .* (1 .- dones))
    # target_values = rewards + gamma * (vec(maximum(target_q(next_states), dims=1)))
    
    target_values = []
    for (ns, r, dn) in zip(next_states, rewards, dones)
        next_action = argmax(online_q(ns))
        # target = r + (gamma * target_q(ns)[next_action]) * (1.0 - dn)
        # MountainCarEnv
        target = r + (gamma * target_q(ns)[next_action])
        push!(target_values, target)
    end
    # for (s, a, ns, r, dn) in zip(states, actions, next_states, rewards, dones)
    #     # next_action = argmax(online_q(ns))
        
    #     target = r + (gamma * maximum(target_q(ns))) * (1.0 - dn)
    #     push!(target_values, target)
    # end
    # Convert target values to a vector
    states = hcat(states...)
    # Calculate the loss for the batch
    sa, grad = Flux.Zygote.withgradient(online_q) do oq
        # loss = mean((target_values .- oq(states)[actions]) .^ 2)  # MSE loss
        expected_values = oq(states)
        extracted_values = [expected_values[actions[i], i] for i in 1:batch_size]
        loss = mean((target_values - extracted_values) .^ 2)  # MSE loss
        return loss
    end
    Optimisers.update!(policy_params_optimiser, online_q, grad[1])
    return sa
end

function test(model, env)
    is_done = false
    reset!(env)
    rewards = 0
    while !is_done
        a = argmax(model(state(env)))
        if typeof(env) == PendulumEnv{true, Float64}
            if a == 1
                a = -2.0
            elseif a == 2
                a = 0.0
            else
                a = 2.0
            end
        end
        act!(env, a)
        r = reward(env)
        is_done = is_terminated(env)
        rewards += r
    end
    return rewards
end


pol_optimizer = ADAM(0.0005)
online_qp = Flux.setup(pol_optimizer, online_q)


gamma = 0.99
# epsilone_decay = 0.9 / 2.5e2
epsilon = eps_start = 1.0
eps_end = 0.01
eps_decay = 0.975
# eps_decay = 0.995
number_of_iterations = 1000
batch_size = 64
max_steps = 200

# my_episode_buffer = Vector{Tuple{Array{Float32}, Int, Float32, Array{Float32}, Bool}}()
my_buffer = MyReplayBuffer(1_000) # 1_000_000
loss_over_time = []
rewards_over_time = []
update_every = 0
for iter in 1:number_of_iterations
    is_done = false
    reset!(env)
    rewards = 0
    
    # while !is_done
    for _ in 1:max_steps
        s = Float32.(state(env))
        
        # Epsilon-greedy action selection
        # if rand() <= epsilon
        #     a = rand(1:length(action_space(env)))  # Random action
        # else
        #     a = argmax(online_q(s))  # Action with max Q-value
        # end
        a = argmax(online_q(s))
        if typeof(env) == PendulumEnv{true, Float64}
            if a == 1
                a = -2.0
            elseif a == 2
                a = 0.0
            else
                a = 2.0
            end
        end
        act!(env, a)
        if typeof(env) == MountainCarEnv{Float64, Int64}
            if env.state[1] >= 0.5
                r = 100
            elseif env.state[1] >= 0.46 || env.state[1] < -1.15
                r = 5
            elseif env.state[1] >= 0.3
                r = 2
            elseif env.state[1] >= 0 || env.state[1] < -0.95
                r = 0.2
            else
                r = -1
            end
            if abs(env.state[2]) <= 0.01
                r = -2
            end
        else
            r = reward(env)
        end
        rewards += r
        is_done = is_terminated(env)
        ns = Float32.(state(env))
        
        # Select action for the next state using online Q network
        na = argmax(online_q(ns))
        add2buffer!(my_buffer, (s, a, r, ns, is_done))
        update_every += 1
        # if length(my_buffer.buffer) >= batch_size && mod(update_every, 4) == 0
        if length(my_buffer.buffer) >= batch_size &&  typeof(env) == MountainCarEnv{Float64, Int64}
            ma_loss = update_ddqn!(my_buffer, online_q, target_q, online_qp, batch_size=batch_size, gamma=gamma)
            push!(loss_over_time, ma_loss)
            # target_q = deepcopy(online_q)
        end
        if length(my_buffer.buffer) >= batch_size && mod(update_every, 20) == 0
            # soft_update!(online_q, target_q, 0.05)
            target_q = deepcopy(online_q)
        end
        if is_done
            break
        end
    end
    epsilon = max(eps_end, eps_decay * epsilon)
    # @show rewards
    test_rew = test(online_q, env)
    # if mod(iter, 20) == 0
    #     # soft_update!(online_q, target_q, 0.05)
    #     target_q = deepcopy(online_q)
    # end
    push!(rewards_over_time, test_rew)
    # println("Iter $(iter); test resutls: epsilon=$(round(epsilon, digits=3)), reward=$(mean(last(rewards_over_time, 100)))")
    println("Iter $(iter); test resutls: epsilon=$(round(epsilon, digits=3)), reward=$(test_rew)")
end

plot(loss_over_time)