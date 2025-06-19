struct DummyLerner <: AbstractLearner
    loss_func::Function
    max_iter::Int
    params
end

function DummyLerner(loss_func::Function, max_iter::Int, model::ExprModel; lr=0.001)
    pol_optimizer = ADAM(lr)
    policy_params = Flux.setup(pol_optimizer, model)
    DummyLerner(loss_func, max_iter, policy_params)
end
(m::ExprModel)(input_values::DeduplicatingNode) = MyModule.heuristic(m, input_values)
function compute_gradient!(target_values, input_values, model, l::AbstractLearner)
    sa, grad = Flux.Zygote.withgradient(model) do oq
        expected_values = oq(input_values)
        l.loss_func(target_values, expected_values)
    end
    Optimisers.update!(l.params, model, grad[1])
    return sa
end

function update_model!(l::AbstractLearner, model::AbstractModel, traj::AbstractTrajectory)
    sampled_trajectory = sample(traj)
    for inner_ep in 1:l.max_iter
        for trajectory in sampled_trajectory
            target_values, input_values = preprocess_trajectory(trajectory, type)
            loss = compute_gradient!(target_values, input_values, model, l)
        end
    end
end