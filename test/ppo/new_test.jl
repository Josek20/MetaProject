# https://github.com/vballoli/PPO.jl/blob/master/src/PPO.jl

using Flux, Statistics
using Flux: Chain, Dense, Conv, onehot, ADAM, mse
using ReinforcementLearning
using ReinforcementLearning: reward, is_terminated

LOSS_CLIPPING = 0.3
ENTROPY_LOSS = 1e-4
GAMMA = 0.899
env = CartPoleEnv()  # Changed to CartPole
NUM_ACTIONS = 2
DUMMY_ACTION = zeros(1, NUM_ACTIONS)
DUMMY_VALUE = zeros(1, 1)

println("Actions: ", NUM_ACTIONS)

function exponential_average(old, new, b1)
    return old * b1 + (1-b1) * new
end

function proximal_policy_optimization_loss(adv, old_pred)
    function loss(y_true, y_pred)
        prob = y_true .* y_pred
        old_prob = y_true .* old_pred
        r = prob ./ (old_prob .+ 1e-10)
        clipped = clamp.(r, 1-LOSS_CLIPPING, 1+LOSS_CLIPPING)
        entropy = -(prob .* log.(prob .+ 1e-10))
        # @show size(entropy), size(r), size(clipped), size(adv)
        # @show size(clipped .* adv), size(r .* adv)
        # @show minimum.([r .* adv clipped .* adv], dims=1)
        return -mean(minimum([r .* adv clipped .* adv], dims=1)) + ENTROPY_LOSS * mean(entropy)
    end
    return loss
end

function build_actor(num_layers, input_dims, hidden_dims, num_actions)
    layers = []
    push!(layers, Dense(input_dims[1], hidden_dims, relu))  # CartPole uses 4D input
    for _ in 1:num_layers-1
        push!(layers, Dense(hidden_dims, hidden_dims, relu))
    end
    push!(layers, Dense(hidden_dims, num_actions))
    return Chain(layers..., softmax)
end

function build_critic(num_layers, input_dims, hidden_dims)
    layers = []
    push!(layers, Dense(input_dims[1], hidden_dims, relu))
    for _ in 1:num_layers-1
        push!(layers, Dense(hidden_dims, hidden_dims, relu))
    end
    push!(layers, Dense(hidden_dims, 1))
    return Chain(layers...)
end

function get_action(actor, obs)
    probs = actor(obs)
    # action_idx = sample(1:NUM_ACTIONS, Weights(vec(probs)))
    action_idx = argmax(probs)
    return action_idx, onehot(action_idx, 1:NUM_ACTIONS), probs
end

function compute_advantages(rewards, values, gamma=GAMMA)
    advantages = similar(rewards)
    last_advantage = 0
    for t in length(rewards):-1:1
        delta = rewards[t] + (t < length(rewards) ? gamma * values[t+1] : 0) - values[t]
        advantages[t] = delta + gamma * last_advantage
        last_advantage = advantages[t]
    end
    return advantages
end

function main(episodes=500, batch_size=32, epochs=50)  # Reduced episodes
    actor = build_actor(3, 4, 64, NUM_ACTIONS)  # Smaller network
    critic = build_critic(3, 4, 64)            # Smaller network
    
    actor_opt = ADAM(0.0003)
    critic_opt = ADAM(0.01)
    
    reward_history = Float64[]
    
    for episode in 1:episodes
        states = []
        actions = []
        old_probs = []
        rewards = []
        
        reset!(env)
        state = deepcopy(env.state)
        episode_reward = 0
        done = false
        
        while !done
            action, action_onehot, probs = get_action(actor, state)
            act!(env, action)
            next_state = deepcopy(env.state)
            rew = reward(env)
            done = env.done
            # next_state, reward, done, _ = step!(env, action)
            
            push!(states, state)
            push!(actions, action_onehot)
            push!(old_probs, probs)
            push!(rewards, rew)
            
            state = next_state
            episode_reward += rew
        end
        
        states = hcat(states...)
        actions = hcat(actions...)
        old_probs = hcat(old_probs...)
        values = critic(states)
        
        advantages = compute_advantages(rewards, vec(values))
        returns = advantages + vec(values)
        
        advantages = (advantages .- mean(advantages)) ./ (std(advantages) .+ 1e-8)
        advantages = reshape(advantages, 1, :)
        for _ in 1:epochs
            actor_opt_state = Flux.setup(actor_opt, actor)
            sa, grad = Flux.Zygote.withgradient(actor) do ac
                new_probs = ac(states)
                loss_fn = proximal_policy_optimization_loss(advantages, old_probs)
                loss_fn(actions, new_probs)
            end
            Optimisers.update!(actor_opt_state, actor, grad[1])

            critic_opt_state = Flux.setup(critic_opt, critic)
            sa, grad = Flux.Zygote.withgradient(critic) do cr
                pred_values = vec(cr(states))
                mse(pred_values, returns)
            end
            Optimisers.update!(critic_opt_state, critic, grad[1])
        end
        
        push!(reward_history, episode_reward)
        if episode % 10 == 0
            println("Episode: $episode, Avg Reward: $(mean(reward_history[end-9:end]))")
        end
    end
    
    return actor, critic, reward_history
end

actor, critic, rewards = main()