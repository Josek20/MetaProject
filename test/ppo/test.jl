using Flux  # For neural networks
using Statistics
using Random
using ReinforcementLearning
using ReinforcementLearning: reward, is_terminated
using Flux: mean, std
using Optimisers

# Define the Actor-Critic network
mutable struct ActorCritic
    policy::Chain    # Policy network
    value::Chain     # Value network
end

function ActorCritic(input_dim::Int, hidden_dim::Int, action_dim::Int)
    policy = Chain(
        Dense(input_dim, hidden_dim, relu),
        Dense(hidden_dim, hidden_dim, relu),
        Dense(hidden_dim, action_dim), softmax
    )
    
    value = Chain(
        Dense(input_dim, hidden_dim, relu),
        Dense(hidden_dim, hidden_dim, relu),
        Dense(hidden_dim, 1)
    )
    
    ActorCritic(policy, value)
end

# PPO Agent
mutable struct PPOAgent
    model::ActorCritic
    optimizer
    clip_epsilon::Float32
    gamma::Float32
    lambda::Float32
end

function PPOAgent(input_dim::Int, hidden_dim::Int, action_dim::Int)
    model = ActorCritic(input_dim, hidden_dim, action_dim)
    optimizer = ADAM(0.0003)
    PPOAgent(model, optimizer, 0.2f0, 0.99f0, 0.95f0)
end

function get_action(agent::PPOAgent, state)
    probs = agent.model.policy(state)
    action = rand(Categorical(probs))
    return action
end

function compute_gae(rewards, values, next_value, dones, gamma, lambda)
    advantages = similar(rewards)
    returns = similar(rewards)
    gae = 0.0
    
    for t in length(rewards):-1:1
        delta = rewards[t] + gamma * (1 - dones[t]) * next_value - values[t]
        gae = delta + gamma * lambda * (1 - dones[t]) * gae
        advantages[t] = gae
        returns[t] = gae + values[t]
        next_value = values[t]
    end
    
    return advantages, returns
end

function update!(agent::PPOAgent, states, actions, old_probs, returns, advantages)
    # Normalize advantages
    advantages = (advantages .- mean(advantages)) ./ (std(advantages) .+ 1e-8)
    
    # Convert to proper types
    states = Float32.(states)
    actions = Int.(actions)
    old_probs = Float32.(old_probs)
    returns = Float32.(returns)
    advantages = Float32.(advantages)
    
    # Compute losses
    # ps = params(agent.model.policy, agent.model.value)
    ps_policy = Flux.setup(agent.optimizer, agent.model.policy)
    ps_value = Flux.setup(agent.optimizer, agent.model.value)
    sa, gs = Flux.Zygote.withgradient(agent.model.policy, agent.model.value) do pm, vm
        # Policy loss
        new_probs = pm(states)
        action_probs = [new_probs[a, i] for (i, a) in enumerate(actions)]
        ratios = action_probs ./ old_probs
        surr1 = ratios .* advantages
        surr2 = clamp.(ratios, 1-agent.clip_epsilon, 1+agent.clip_epsilon) .* advantages
        policy_loss = -mean(min.(surr1, surr2))
        
        # Value loss
        values = vm(states)[:]
        value_loss = mean((returns .- values).^2)
        
        # Total loss
        policy_loss + 0.5 * value_loss
    end
    
    # Update parameters
    Flux.update!(ps_policy, agent.model.policy, gs[1])
    Flux.update!(ps_value, agent.model.value, gs[2])
end

function train_ppo(env; episodes=1000, max_steps=500, update_freq=20, epochs=10)
    # Assuming CartPole has 4 observations and 2 actions
    agent = PPOAgent(4, 64, 2)
    total_rewards = Float32[]
    
    states = []
    actions = []
    rewards = []
    old_probs = []
    dones = []
    
    for episode in 1:episodes
        reset!(env)
        episode_reward = 0.0
        
        for t in 1:max_steps
            # Get action probabilities
            st = deepcopy(env.state)

            probs = agent.model.policy(st)
            action = argmax(probs)
            
            # Take step in environment
            # next_state, reward, done, _ = step!(env, action-1)  # -1 because actions are 0,1
            act!(env, action)
            rew = reward(env)
            done = is_terminated(env)
            # Store transition
            push!(states, st)
            push!(actions, action)
            push!(rewards, rew)
            push!(old_probs, probs[action])
            push!(dones, done)
            
            episode_reward += rew
            next_state = deepcopy(env.state)
            
            # Update if we have enough samples
            if length(states) >= update_freq
                # Compute returns and advantages
                values = agent.model.value(hcat(states...))[:]
                next_value = agent.model.value(next_state)[1]
                advantages, returns = compute_gae(rewards, values, next_value, dones, 
                                                agent.gamma, agent.lambda)
                
                # Multiple epochs of updates
                for _ in 1:epochs
                    update!(agent, hcat(states...), actions, old_probs, returns, advantages)
                end
                
                # Clear buffers
                empty!(states)
                empty!(actions)
                empty!(rewards)
                empty!(old_probs)
                empty!(dones)
            end
            
            done && break
        end
        
        push!(total_rewards, episode_reward)
        println("Episode $episode: Reward = $episode_reward")
    end
    
    return agent, total_rewards
end

# Example usage:
env = CartPoleEnv()  # Assuming this exists
agent, rewards = train_ppo(env)