using ReinforcementLearning
using ReinforcementLearning: reward, is_terminated
using Flux
using Flux: mean, std, mse, OptimiserChain, ClipGrad
using Optimisers
using Distributions
using Serialization

function step!(env::CartPoleEnv, action)
    act!(env, action)
    return copy(env.state), reward(env)
end


mutable struct Agent
    critic
    actor
end

function Agent(env::CartPoleEnv, hidden_size=64)
    input_size = length(state_space(env).a)
    @assert isinteger(input_size) 
    output_size = length(action_space(env))
    @assert isinteger(output_size) 

    actor = Chain(Dense(input_size, hidden_size, tanh), Dense(hidden_size, hidden_size, tanh), Dense(hidden_size, output_size))
    critic = Chain(Dense(input_size, hidden_size, tanh), Dense(hidden_size, hidden_size, tanh), Dense(hidden_size, 1))
    return Agent(critic, actor)
end

function get_action_value(x, agent::Agent, action=nothing)
    logits = softmax(agent.actor(x))
    probs = Categorical(logits)
    if isnothing(action)
        action = rand(probs)
    end
    shannon_entropy = -sum(probs.p .* log.(probs.p))
    return action, logpdf(probs, action), shannon_entropy, agent.critic(x) 
end

env = CartPoleEnv()
agent = Agent(env)
eps = 1e-5
lr = 2.5e-4
optim = OptimiserChain(ClipGrad(eps), Adam(lr))

num_step = 128
batch_size = num_step

total_timesteps = 25_000
num_updates = total_timesteps // batch_size


states = zeros((num_step, length(state_space(env).a)))
actions = zeros(num_step)
log_probs = zeros(num_step)
rewards = zeros(num_step)
dones = zeros(num_step)
values = zeros(num_step)

episode_return = 0
global_step = 0
for update in 1:num_updates + 1
    reset!(env)
    state = copy(env.state)    
    # Todo: implement lr annealing
    done_step = 0
    for step in 1:num_step
        global_step += 1
        states[step, :] = state
        action, log_prob, _, value = get_action_value(state, agent)
        values[step] = only(value)
        actions[step] = action
        log_probs[step] = log_prob
        act!(env, action)
        next_state = copy(env.state)
        rewards[step] = ReinforcementLearning.reward(env)
        done = is_terminated(env)
        dones[step] = done
        if mod(global_step, 100) == 0
            @show global_step, sum(rewards)
        end
        if done
            done_step = step
            break
        end
    end
    G = []
end