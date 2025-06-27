mutable struct DummyLerner{LF, MI, P} <: AbstractLearner
    loss_func::LF
    max_iter::MI
    params::P
end

function DummyLerner(loss_func::Function, model::ExprModel; lr=0.001, max_iter=10)
    pol_optimizer = ADAM(lr)
    policy_params = Flux.setup(pol_optimizer, model)
    DummyLerner(loss_func, max_iter, policy_params)
end
function compute_gradient!(target_values, input_values, model, l::AbstractLearner)
    sa, grad = Flux.Zygote.withgradient(model) do oq
        expected_values = vec(MyModule.heuristic(oq, input_values))
        l.loss_func(target_values, expected_values)
    end
    Optimisers.update!(l.params, model, grad[1])
    return sa
end


function sample(traj::Trajectory)
    return [(1, traj)]
end
function sample(traj::TreeTrajectory)
    all_traj = [(i, traj) for i in 1:length(traj.states)]
    return all_traj
end


function update_model!(l::AbstractLearner, model::AbstractModel, traj::AbstractTrajectory)
    sampled_trajectory = sample(traj)
    for inner_ep in 1:l.max_iter
        for trajectory in sampled_trajectory
            target_values, input_values = preprocess_trajectory(trajectory)
            loss = compute_gradient!(target_values, input_values, model, l)
        end
    end
end