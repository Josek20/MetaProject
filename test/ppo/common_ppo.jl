using ReinforcementLearning
using ReinforcementLearning: reward, is_terminated
using Flux
using Flux: mean, std, mse
using Optimisers
using Distributions
using Serialization

function step!(env::CartPoleEnv, action)
    act!(env, action)
    return copy(env.state), reward(env)
end

function step!(env::PendulumEnv, action)
    @show action
    act!(env, action)
    return copy(env.state), reward(env)
end

function get_episodes(env, action_picker, policy_model, value_model, policy_opt_state, value_opt_state; number_of_episodes=1, epsilon=0.2)
    episodes = []
    for i in 1:number_of_episodes
        reset!(env)
        state = copy(env.state)
        # episode = []
        while !env.done
            if rand() >= epsilon
                action = action_picker(state)
            else
                action = sample(action_space(env))
                # action = rand(action_space(env))
            end
            next_state, rew = step!(env, action)
            # push!(episode, (;state=state, action=action, reward=rew, next_state=next_state, is_terminal=env.done))
            push!(episodes, (;state=state, action=action, reward=rew, next_state=next_state, is_terminal=env.done))
            update!(episodes, policy_model, value_model, epsilon, policy_opt_state, value_opt_state)
            state = next_state
        end
        # push!(episodes, episode)
    end
    return episodes
end

dum_action_picker(env::CartPoleEnv) = 1

function update!(my_episodes, policy_model, value_model, epsilone, policy_opt_state, value_opt_state)
    episodes = []
    for i in my_episodes
        append!(episodes, i)
    end
    states = [i.state for i in episodes]
    stacked_states = stack(states)
    # len_batch = length(episodes)
    actions = [i.action + 2 * (ind - 1) for (ind,i) in enumerate(episodes)]
    rew = [i.reward for i in episodes]
    is_terminal = [i.is_terminal for i in episodes]
    discounted_rewards = []
    discounted_reward = 0
    for (r, ist) in zip(reverse(rew), reverse(is_terminal))
        if ist
            discounted_reward = 0
        end
        discounted_reward = r + gamma * discounted_reward 
        push!(discounted_rewards, discounted_reward)
    end
    
    normalized_dicounterd_rewards = (discounted_rewards .- mean(discounted_rewards)) / (std(discounted_rewards) + 1e-7)
    advantages = normalized_dicounterd_rewards - vec(value_model(stacked_states))
    actions_log_prob = policy_model(stacked_states)[actions]
    for k in 1:K
        sa, grad = Flux.Zygote.withgradient(policy_model, value_model) do pm, vm
            probs = pm(stacked_states)[actions]
            state_values = vec(vm(stacked_states))
            ratios = probs ./ actions_log_prob
            surrogate1 = ratios .* advantages
            surrogate2 = clamp.(ratios, 1 - epsilone, 1 + epsilone) .* advantages
            loss = -min(surrogate1, surrogate2) .+ 0.5 * mse(state_values, normalized_dicounterd_rewards)
            tmp = mean(loss)
            tmp
        end
    # @show sa
        Flux.update!(policy_opt_state, policy_model, grad[1])
        Flux.update!(value_opt_state, value_model, grad[2])
        if isnan(sa)
            break
        end
    end
end

function train(;max_update_frequencey=100, iterations=500, number_of_episodes=100,  gamma=0.99, epsilone=0.2, K=40)
    policy_model = Chain(Dense(4, 32, relu), Dense(32, 2), softmax)
    old_policy_model = Chain(Dense(4, 32, relu), Dense(32, 2), softmax)
    old_policy_model = policy_model
    value_model = Chain(Dense(4, 32, relu), Dense(32, 1))
    pol_optimizer = ADAM(0.0003)
    val_optimizer = ADAM(0.001)
    value_opt_state = Flux.setup(val_optimizer, value_model)
    policy_opt_state = Flux.setup(pol_optimizer, policy_model)
    policy_action_picker = (st)->argmax(policy_model(st))
    # policy_action_picker = (st)->only(policy_model(st))
    my_episodes = []
    for i in 1:iterations
        @show i
        # if i == 1
        my_episodes = get_episodes(env, policy_action_picker, number_of_episodes=number_of_episodes)
        # else
        #     new_episodes = get_episodes(env, policy_action_picker, number_of_episodes=1)
        #     pop!(my_episodes)
        #     append!(my_episodes, new_episodes)
        #     # push!(my_episodes, new_episodes)
        # end
        @show length(my_episodes)
        # @assert length(my_episodes) == number_of_episodes
        update!(my_episodes, policy_model, value_model, epsilone)
        # if i % max_update_frequencey == 0
        #     # Todo: copy weights
        #     old_policy_model = policy_model
        # end        
    end
    hidden_size = 32
    env_name = "cart_pole_env"
    experiment_name = "$(env_name)_ppo_hds=$(hidden_size)_gamma=$(gamma)_eps=$(epsilone)_iter=$(iterations)_k=$(K)_buffsize=$(number_of_episodes)"
    serialize("models/trained_policy_model_$(experiment_name)", pl)
    serialize("models/trained_value_model_$(experiment_name)", vl)
    return policy_model, value_model
end


function plot_stats(rew_per_iter, loss_per_iter)
    p1 = plot(1:length(loss_per_iter), loss_per_iter, xlabel="Iterations", ylabel="Value", label="loss")
    p2 = plot(1:length(loss_per_iter), rew_per_iter, xlabel="Iterations", ylabel="Value", label="reward")
    plot(p1, p2, layout=(1,2))
    savefig("stats/ppo_loss_reward.png")
end


env = CartPoleEnv()
policy_model = Chain(Dense(4, 32, relu), Dense(32, 2), softmax)
old_policy_model = Chain(Dense(4, 32, relu), Dense(32, 2), softmax)
old_policy_model = policy_model
value_model = Chain(Dense(4, 32, relu), Dense(32, 1))
pol_optimizer = ADAM(0.0003)
val_optimizer = ADAM(0.001)
value_opt_state = Flux.setup(val_optimizer, value_model)
policy_opt_state = Flux.setup(pol_optimizer, policy_model)
# pl, vl = train()
policy_action_picker = (st)->argmax(pl(st))
new_episodes = get_episodes(env, policy_action_picker, number_of_episodes=1)
@show sum([i.reward for i in new_episodes[1]])
