using MyModule
using MyModule: all_expand, NodeID, exp_size, intern!
using POMDPs, Crux, Flux, POMDPGym
function training_data(n=typemax(Int))
    train_data_path = "data/neural_rewrter/test.json"
    train_data = load_data(train_data_path)[1:1000]
    train_data = filter(x->!occursin("select", x[1]), train_data)
    train_data = preprosses_data_to_expressions(train_data)
    train_data = sort(train_data, by=x->MyModule.exp_size(x))
    last(train_data, min(length(train_data), n))
end
trn_data = training_data(1000)
struct MyEnv <: POMDPs.MDP{NodeID, NodeID}
  t
  s0
  s
  max_t
end
MyEnv(ex::NodeID, max_t=50) = MyEnv(1, ex, ex, max_t)
POMDPs.states(env::MyEnv) = nothing
function POMDPs.actions(env::MyEnv)
  # all_expand(env.s, theory)
  tmp = filter(x->x!=s, all_actions)
  length(tmp) == 0 && return NodeID[s]
  return tmp
end
POMDPs.initialstate(env::MyEnv) = env.s0
function POMDPs.transition(env::MyEnv, s, a)
  env.s = a
  env.t += 1
  return a
end
function POMDPs.reward(env::MyEnv, s, a)
  initial_size = exp_size(s0)
  current_size = exp_size(s)
  next_size = exp_size(a)
  if initial_size <= next_size
    return -1
  else
    return initial_size - next_size
  end
end
POMDPs.isterminal(env::MyEnv, s) = env.t == env.max_t
POMDPs.discount(env::MyEnv) = 0.99
function POMDPs.gen(env::MyEnv, s, a, rng)
    sp = transition(env, s, a)
    r = reward(env, s, a)
    terminal = isterminal(env, s)
    return (sp=sp, r=r, terminal=terminal)
end
POMDPs.initialstate(env, rng) = env.s0
## Cartpole - V0
# mdp = GymPOMDP(:CartPole, version = :v1)
my_mdp = MyEnv(intern!(trn_data[end]))
mdp = GymPOMDP(:MountainCar, version = :v0)
as = 1
S = 64
A() = DiscreteNetwork(Chain(Dense(Crux.dim(S)..., 64, relu), Dense(64, 64, relu), Dense(64, length(as))), as)
V() = ContinuousNetwork(Chain(Dense(Crux.dim(S)..., 64, relu), Dense(64, 64, relu), Dense(64, 1)))
# Solve with REINFORCE (~2 seconds)
# 𝒮_reinforce = REINFORCE(π=A(), S=S, N=10000, ΔN=500, a_opt=(epochs=5,), interaction_storage=[])
# @time π_reinforce = solve(𝒮_reinforce, mdp)
#
# # Solve with A2C (~8 seconds)
# 𝒮_a2c = A2C(π=ActorCritic(A(), V()), S=S, N=10000, ΔN=500)
# @time π_a2c = solve(𝒮_a2c, mdp)
#
# Solve with PPO (~15 seconds)
# m_ppo = PPO(π=ActorCritic(A(), V()), S=S, N=10000, ΔN=500)
# @time π_ppo = solve(m_ppo, mdp)
#
# # Solve with DQN (~12 seconds)
# 𝒮_dqn = DQN(π=A(), S=S, N=10000, interaction_storage=[])
# @time π_dqn = solve(𝒮_dqn, mdp)
# Solve with SoftQLearning w/ varying α (~12 seconds)
# 𝒮_sql = SoftQ(π=A(), α=Float32(0.1), S=S, N=10000,
#     ΔN=1, c_opt=(;epochs=5), interaction_storage=[])
# @time π_sql = solve(𝒮_sql, mdp)
#
# # Plot the learning curve
# p = plot_learning([𝒮_reinforce, 𝒮_a2c, 𝒮_ppo, 𝒮_dqn, 𝒮_sql], title = "CartPole-V0 Training Curves",
#     labels = ["REINFORCE", "A2C", "PPO", "DQN", "SoftQ" ])
# Crux.savefig(p, "cartpole_training.pdf")
#
# # Produce a gif with the final policy
# gif(mdp, π_ppo, "cartpole_policy.gif", max_steps=100)
## Optional - Save data for imitation learning
# using BSON
# s = Sampler(mdp, 𝒮_dqn.agent, max_steps=100, required_columns=[:t])
#
# data = steps!(s, Nsteps=10000)
# sum(data[:r])/100
# data[:expert_val] = ones(Float32, 1, 10000)
# data[:a]
#
# data = ExperienceBuffer(data)
# BSON.@save "examples/il/expert_data/cartpole.bson" data