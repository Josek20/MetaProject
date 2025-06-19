abstract type AbstractEnvironment end
# abstract type AbstractModel end
abstract type AbstractSampler end
abstract type AbstractLearner end

mutable struct RLPipeline
    env::AbstractEnvironment
    model::AbstractModel
    sampler::AbstractSampler
    learner::AbstractLearner
end

function train!(pipeline::RLPipeline; episodes::Int=100)
    for episode in 1:episodes
       traj = sample_trajectory(pipeline.sampler, pipeline.env, pipeline.model)
       update_model!(pipeline.learner, pipeline.model, traj)
       update_epsilon!(pipeline.sampler)
    end
end


# sample_trajectory(s::AbstractSampler, env::AbstractEnvironment, model::AbstractModel)::Trajectory = error("sample_trajectory not inmplemented")
# update_model!(l::AbstractLearner, model::AbstractModel, traj::Trajectory) = error("sample_trajectory not inmplemented")