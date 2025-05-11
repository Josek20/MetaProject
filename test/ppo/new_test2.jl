using Flux
using Distributions
using Zygote
using CircularArrays

mutable struct RoboticArmEnv
    position::Float64
    target::Float64
    done::Bool
end

function step!(env::RoboticArmEnv, action)
    env.position += action
    reward = -abs(env.position - env.target)
    env.done = reward >= -0.1
    return reward
end

struct PPOAgent
    policy::Chain
    old_policy::Chain
    value::Chain
    opt
end

function train_ppo(agent::PPOAgent, env::RoboticArmEnv, episodes::Int)

    buffer = CircularArray(Float64, 1000)
    policy_opt_state = Flux.setup(agent.opt, agent.policy)
    value_opt_state = Flux.setup(agent.opt, agent.value)
    update_frequencey = 0
    max_update_frequencey = 100
    for i in 1:episodes
        state = env.position
        rewards = []

        while !env.done
            action = rand(Normal(agent.policy(Float32[state]), 1.0))
            reward = step!(env, action)
            push!(rewards, reward)
            push!(buffer, reward)
            state = env.position
        end

        if length(buffer) > 100
            batch = sample(buffer, 32)
            grads = Flux.gradient(agent.policy, agent.value) do pl, vl
                pol_loss, val_loss = ppo_loss(batch, pl, vl, agent.old_policy)
            end
            Flux.update!(policy_opt_state, agent.policy, grads[1])
            Flux.update!(value_opt_state, agent.value, grads[2])
        end
        if update_frequencey % max_update_frequencey == 0
            # copy weights from policy to old_policy
            agent.old_policy = agent.policy
        end
    end
end

function ppo_loss(batch, policy, value, old_policy)
    # Clipped surrogate objective
    rewards = [b[1] for b in batch] 
    states = [b[2] for b in batch]
    len_batch = length(batch)
    actions = [b[3] + len_batch * (ind - 1) for (ind,b) in enumerate(batch)]

    advantages = rewards .- value.(states)
    ratios = policy.(states)[actions] / old_policy.(states)[actions]

    clipfracs = [clamp(ratio, 1-ε, 1+ε) for ratio in ratios]
    pol_loss = -mean(min.(ratios.*advantages, clipfracs.*advantages))

    val_loss = mean((value.(states) .- rewards).^2)
    return pol_loss + val_loss
end

policy = Chain(Dense(1, 32, relu), Dense(32, 2))
old_policy = Chain(Dense(1, 32, relu), Dense(32, 2))
value = Chain(Dense(1, 32, relu), Dense(32, 1))

opt = ADAM(0.001)
agent = PPOAgent(policy, old_policy, value, opt)
env = RoboticArmEnv(0.0, 5.0, false)

train_ppo(agent, env, 100)

# using Flux
# using Distributions
# using Flux.Zygote
# using CircularArrays
# using ReinforcementLearning
# using ReinforcementLearning: reward, is_terminated


# struct PPOAgent
#     policy
#     value
#     opt
# end

# function train_ppo(agent::PPOAgent, env, episodes::Int)

#     buffer = CircularArray(Float64, 1000)

#     for i in 1:episodes
#         reset!(env)
#         state = env.state
#         rewards = []

#         while !env.done
#             # action = rand(Normal(agent.policy(state), 1.0))
#             action = argmax(agent.policy(state))
#             # reward = step!(env, action)
#             act!(env, action)
#             rew = reward(env)
#             push!(rewards, rew)
#             push!(buffer, typeof(rew))
#             state = env.state
#         end

#         if length(buffer) > 100
#             batch = sample(buffer, 32)
#             @show batch
#             pol_loss, val_loss = ppo_loss(batch, agent.policy, agent.value)
#             Flux.train!(pol_loss, agent.opt)
#             Flux.train!(val_loss, agent.opt)
#         end
#     end
# end

# function ppo_loss(batch, policy, value)
#     # Clipped surrogate objective
#     rewards = [b[1] for b in batch] 
#     states = [b[2] for b in batch]
#     actions = [b[3] for b in batch]

#     advantages = rewards .- value(states)
#     ratios = policy.(states, actions)

#     clipfracs = [clamp(ratio, 1-ε, 1+ε) for ratio in ratios]
#     pol_loss = -mean(min.(ratios.*advantages, clipfracs.*advantages))

#     val_loss = mean((value.(states) .- rewards).^2)
#     return pol_loss, val_loss
# end

# policy = Chain(Dense(4, 32, relu), Dense(32, 2), softmax)
# value = Chain(Dense(4, 32, relu), Dense(32, 1))

# opt = ADAM(0.01)
# agent = PPOAgent(policy, value, opt)
# env = CartPoleEnv()

# train_ppo(agent, env, 100)