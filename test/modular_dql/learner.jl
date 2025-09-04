mutable struct DummyLerner{LF, MI, P} <: AbstractLearner
    loss_func::LF
    max_iter::MI
    params::P
end

mutable struct HeadLerner{LF, MI, P, FH, SH} <: AbstractLearner
    loss_func::LF
    max_iter::MI
    model_params::P
    first_head::FH
    second_head::SH
end

# mutable struct PolicyLerner{LF, MI, P} <: AbstractLearner
#     loss_func::LF
#     max_iter::MI
#     params::P
# end

function DummyLerner(loss_func::Function, model::ExprModel; lr=0.001, max_iter=10)
    pol_optimizer = ADAM(lr)
    policy_params = Flux.setup(pol_optimizer, model)
    DummyLerner(loss_func, max_iter, policy_params)
end
# function HeadLerner(loss_func::Function, model::DoubleHeadedModel; lr=0.001, max_iter=10)
#     pol_optimizer1 = ADAM(lr)
#     pol_optimizer2 = ADAM(lr)
#     policy_params1 = Flux.setup(pol_optimizer1, model.first_head)
#     policy_params2 = Flux.setup(pol_optimizer2, model.second_head)
#     HeadLerner(loss_func, max_iter, [policy_params1, policy_params2], model.first_head, model.second_head)
# end
# function PolicyLerner(loss_func::Function, model::ExprModel; lr=0.001, max_iter=10)
#     pol_optimizer = ADAM(lr)
#     policy_params = Flux.setup(pol_optimizer, model)
#     DummyLerner(loss_func, max_iter, policy_params)
# end
Flux.mse(x::Tuple) = Flux.mse(x...)
Flux.crossentropy(x::Tuple) = Flux.crossentropy(reverse(x)...)
my_reinforce_loss(x::Tuple) = mean(- log.(x[2]) .* x[1])
# function my_reinforce_loss(x::Tuple)
#     # @show x[2]
#     tmp1 = - log.(x[2])
#     # @show tmp1
#     mean(tmp1 .* x[1])
# end
function compute_gradient!(target_values, input_values::Tuple, model::ExprModel, l::DummyLerner)
    sa, grad = Flux.Zygote.withgradient(model) do oq
        expected_values = vec(MyModule.heuristic(oq, input_values[1]))
        expected_values = expected_values .* input_values[2][1]
        # expected_values[expected_values .== 0] = -Inf
        expected_values = ifelse.(input_values[2][1], expected_values, -Inf)
        expected_values = softmax(expected_values)
        # @show size(expected_values)
        # @show input_values[2][2]
        # @show expected_values[input_values[2][2]]
        loss = l.loss_func((target_values, expected_values[input_values[2][2]]))
        return loss
    end
    Optimisers.update!(l.params, model, grad[1])
    return sa
end
function compute_gradient!(target_values, input_values, model::ExprModel, l::DummyLerner)
    sa, grad = Flux.Zygote.withgradient(model) do oq
        expected_values = vec(MyModule.heuristic(oq, input_values))
        # @show target_values, expected_values
        l.loss_func((target_values, expected_values))
    end
    Optimisers.update!(l.params, model, grad[1])
    return sa
end

# function compute_gradient!(target_values, input_values, model::DoubleHeadedModel, l::HeadLerner)
#     values1 = first.(target_values)
#     values2 = map(x->x[2], target_values)
#     embeddings = model.main_body(input_values)
#     # @show input_values[1
#     sa1, grad = Flux.Zygote.withgradient(model.first_head) do oq
#         expected_values = vec(oq(embeddings))
#         l.loss_func(values1, expected_values)
#     end
#     Optimisers.update!(l.model_params[1], model.first_head, grad[1])
#     sa2, grad = Flux.Zygote.withgradient(model.second_head) do oq
#         expected_values = vec(oq(embeddings))
#         l.loss_func(values2, expected_values)
#     end
#     Optimisers.update!(l.model_params[2], model.second_head, grad[1])
#     return sa1 + sa2
# end


function sample(traj::AbstractTrajectory)
    return [(1, traj)]
end
function sample(traj::TreeTrajectory)
    all_traj = [(i, traj) for i in 1:length(traj.states)]
    return all_traj
end


function update_model!(l::AbstractLearner, model::AbstractModel, traj::AbstractTrajectory)
    sampled_trajectory = sample(traj)
    loss = 0
    input_processing_time = 0
    learning_time = 0
    for inner_ep in 1:l.max_iter
        for trajectory in sampled_trajectory
            # @show trajectory
            # error("")
            input_processing_time += @elapsed target_values, input_values = preprocess_trajectory(trajectory)
            learning_time += @elapsed loss += compute_gradient!(target_values, input_values, model, l)
            # @show loss
        end
    end
    final_loss = loss / (l.max_iter * length(sampled_trajectory))
    @show final_loss, input_processing_time, learning_time 
    return final_loss
end