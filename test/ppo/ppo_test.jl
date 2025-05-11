using ReinforcementLearning
using ReinforcementLearning: reward, is_terminated
using Flux
using Flux: mean, std, mse
using Optimisers
using Distributions

env = CartPoleEnv()

hidden_size = 64
policy_model = Chain(Dense(4, hidden_size, relu), Dense(hidden_size, hidden_size, relu), Dense(hidden_size, 2), softmax)
old_policy_model = Chain(Dense(4, hidden_size, relu), Dense(hidden_size, hidden_size, relu), Dense(hidden_size, 2))
value_model = Chain(Dense(4, hidden_size, tanh), Dense(hidden_size, hidden_size), Dense(hidden_size, 1))


function my_ppo_loss(old_policy, new_policy_model, value_model, states, actions, rewards; ϵ = 0.25)
    # old_policy = [old_policy_model(state)[action] for (state, action) in zip(states, actions)]
    # @show new_policy_model(states[1])[actions[1]]
    stacked_states = stack(states)
    new_policy = new_policy_model(stacked_states)
    # all_values = [only(value_model(state)) for state in states]
    all_values = value_model(stacked_states)
    policy_loss = 0
    # for t in length(rewards):-1: 1
    entropy = 0
    for t in 1:length(rewards)
        r0 = new_policy[actions[t], t] / old_policy[t]
        Aₜ = advantages_estimation(all_values, rewards, t)
        policy_loss += min(r0 * Aₜ, clamp(r0, 1 - ϵ, 1 + ϵ) * Aₜ)
        entropy += (new_policy[actions[t], t] * log(new_policy[actions[t], t] + 1e-10))
    end
    policy_loss /= length(rewards)
    value_loss = mean((all_values[1:end-1] .- rewards).^2)
    entropy /= length(rewards)
    return policy_loss + 0.5 * value_loss
end

function advantages_estimation(values, rewards, t=1; γ = 0.99, λ = 0.9)
    t == length(rewards) && return rewards[t] + γ * values[t + 1] - values[t]
    δₜ = rewards[t] + γ * values[t + 1] - values[t]
    # Aₜ = δₜ + γ * λ * advantages_estimation(values, rewards, t + 1)
    Aₜ = δₜ + (γ * λ) * advantages_estimation(values, rewards, t + 1)
    return Aₜ
end


function new_advatage_estimation(rewards, values; γ = 0.99, λ = 0.95)
    @assert length(rewards) + 1 == length(values)
    Aₜ = 0
    for t in length(rewards):-1:1
        Aₜ += γ ^ (t - 1) * λ ^ (t - 1) * (rewards[t] + γ * values[t + 1] - values[t]) 
    end
    return Aₜ
end

function ppo_loss(policy_model, value_model, old_probs, states, actions, advantages, returns; clip_ϵ=0.1)
    # Get current policy probabilities
    # @show states, size(actions)
    curr_probs = [policy_model(s)[a] for (s, a) in zip(states, actions)]
    # stacked_states = stack(states)
    # curr_probs = policy_model(stacked_states)[actions]
    # Get old policy probabilities
    # old_probs = [old_policy_model(s)[a] for (s, a) in zip(states, actions)]
    
    ratios = curr_probs ./ (old_probs .+ 1e-10)
    pred_values = [only(value_model(s)) for s in states]
    # pred_values = value_model(stacked_states)
    value_loss = mean((pred_values[1:end-1] .- returns).^2)
    
    # Calculate policy loss with clipping
    advantages_norm = (advantages .- mean(advantages)) ./ (std(advantages) .+ 1e-10)
    surr1 = ratios .* advantages_norm
    surr2 = clamp.(ratios, 1 - clip_ϵ, 1 + clip_ϵ) .* advantages_norm
    policy_loss = -mean(min.(surr1, surr2))
    
    # Total loss
    return policy_loss + 0.5 * value_loss
end
function new_train_ppo(; steps=env.params.max_steps, epochs=100_000, γ = 0.99, K=40, ϵ = 0.2, epsilon=1.0, eps_decay=0.995, min_epsilon=0.01)
    # pol_optimizer = Optimiser(ClipNorm(1f-4), Adam(0.0003))
    # val_optimizer = Optimiser(ClipNorm(1f-4), Adam(0.001))
    pol_optimizer = ADAM(0.0003)
    val_optimizer = ADAM(0.001)
    value_opt_state = Flux.setup(val_optimizer, value_model)
    policy_opt_state = Flux.setup(pol_optimizer, policy_model)
    # val_pr = Flux.params(value_model)
    # pol_pr = Flux.params(policy_model)
    states = []
    actions = []
    actions_log_prob = []
    state_values = []
    rewards = []
    my_ranges = []
    for epoch in 1:epochs 
        
        reset!(env)
        total_reward = 0
        state = copy(env.state)
        for _ in 1:steps
            # if rand() > epsilon
            probs = policy_model(state)
            dist = Categorical(probs)
            action = argmax(dist.p)
            # else
            #     action = rand(action_space(env))
            # end
            act!(env, action)
            rew = reward(env)
            push!(states, state)
            push!(state_values, only(value_model(state)))
            push!(actions_log_prob, logpdf(dist, action))
            push!(actions, action)
            push!(rewards, rew)
            total_reward += rew
            
            state = copy(env.state)
            done = is_terminated(env)
            done && break
        end
        epsilon = max(min_epsilon, epsilon * eps_decay)
        if isempty(my_ranges)
            push!(my_ranges, 1:length(states))
        else
            start_range = my_ranges[end].stop
            push!(my_ranges, start_range + 1:length(states))
        end
        @assert length(rewards) == length(states)
        # @show epoch, length(states)
        if length(states) >= 400
            stacked_old_states = stack(states)
            # states = stacked_old_states
            # stacked_old_states_values = stack()
            old_states_values = state_values
            discounted_rewards = Float32[]
            # discounted_reward = 0
            # for reward in reverse(rewards)
            #     discounted_reward = reward + γ * discounted_reward
            #     push!(discounted_rewards, discounted_reward)
            # end
            for rng in my_ranges
                discounted_reward = 0
                for reward in reverse(rewards[rng])
                    discounted_reward = reward + γ * discounted_reward
                    push!(discounted_rewards, discounted_reward)
                end
            end
            reverse!(discounted_rewards)
            normalized_dicounterd_rewards = (discounted_rewards .- mean(discounted_rewards)) / (std(discounted_rewards) + 1e-7)
            advantages = normalized_dicounterd_rewards - old_states_values
            for k in 1:K
                # @show k
                sa, grad = Flux.Zygote.withgradient(value_model, policy_model) do vm, pm
                # grad = Flux.gradient() do v, 
                    probs = pm.(states)
                    # reshaped_probs = reshape(probs, :)
                    # reshaped_probs = [reshaped_probs[i:i+1] for i in 1:2:length(reshaped_probs)]
                    dist = Categorical.(probs)
                    log_probs = logpdf.(dist, actions)
                    dist_entropy = entropy.(dist)
                    state_values = vm(stacked_old_states)
                    ratios = exp.(log_probs - actions_log_prob)
                    surrogate1 = ratios .* advantages
                    surrogate2 = clamp.(ratios, 1 - ϵ, 1 + ϵ) .* advantages
                    # @show size(surrogate1), size(surrogate2)
                    # @show size(state_values), size(normalized_dicounterd_rewards)
                    # state_values = stack(state_values, dims=1)
                    # @show size(normalized_dicounterd_rewards), size(state_values)
                    loss = -min(surrogate1, surrogate2) .+ 0.5 * mse(state_values[1,:], normalized_dicounterd_rewards) - 0.01 * dist_entropy
                    tmp = mean(loss)
                    
                    tmp
                end
                # @show sa
                Optimisers.update!(value_opt_state, value_model, grad[1])
                Optimisers.update!(policy_opt_state, policy_model, grad[2])
            end
            reset!(env)
            eval_reward = 0
            while !is_terminated(env)
                probs = policy_model(env.state)
                dist = Categorical(probs)
                action = argmax(dist.p)
                act!(env, action)
                rew = reward(env)
                eval_reward += rew
            end
            println("Epoch $epoch - Eval Reward: $eval_reward")
            states = []
            state_values = []
            actions_log_prob = []
            actions = []
            rewards = []
            my_ranges = []
        end
    end
end
function train_ppo(; steps=400, epochs=4000)
    pol_optimizer = ADAM(3e-4)
    val_optimizer = ADAM(0.001)
    previous_policy_models = [deepcopy(policy_model)]
    value_opt_state = Flux.setup(val_optimizer, value_model)
    policy_opt_state = Flux.setup(pol_optimizer, policy_model)
    for epoch in 1:epochs 
        states = []
        actions = []
        rewards = []
        
        reset!(env)
        total_reward = 0
        state = deepcopy(env.state)
        for _ in 1:steps
            probs = policy_model(state)
            action = argmax(probs)
            act!(env, action)
            rew = reward(env)
            push!(states, state)
            push!(actions, action)
            push!(rewards, rew)
            total_reward += rew
            
            state = deepcopy(env.state)
            done = is_terminated(env)
            done && break
        end
        push!(states, state)
        @assert length(rewards) + 1 == length(states)

        # values = [only(value_model(s)) for s in states]
        # returns = Float32[]
        # G = 0
        # for r in reverse(rewards)
        #     G = r + 0.99 * G
        #     push!(returns, G)
        # end
        # reverse!(returns)
        # advantages = [advantages_estimation(values, rewards, t) for t in 1:length(rewards)]
        
        if epoch == 1
            old_pm = previous_policy_models[end]
        else
            push!(previous_policy_models, deepcopy(policy_model))
            old_pm = previous_policy_models[end - 1]
        end
        stacked_states = stack(states)
        # old_probs = [old_pm(state)[action] for (state, action) in zip(states, actions)]
        old_probs = old_pm(stacked_states)
        old_probs = [old_probs[actions[t], t] for t in 1:length(rewards)]
        for _ in 1:40
            sa, grad = Flux.Zygote.withgradient(value_model, policy_model) do vm, pm
                # ppo_loss(pm, vm, old_probs, states, actions, advantages, rewards)
                my_ppo_loss(old_probs, pm, vm, states, actions, rewards)
            end
            # @show length(grad)
            @show sa
            Optimisers.update!(value_opt_state, value_model, grad[1])
            Optimisers.update!(policy_opt_state, policy_model, grad[2])
        end
        
        println("Epoch $epoch - Total Reward: $total_reward")
    end
end

# Run training
# train_ppo()
new_train_ppo()