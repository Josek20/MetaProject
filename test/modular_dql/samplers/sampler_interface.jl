abstract type AbstractSampler end


function sample_trajectory(sampler::AbstractSamplerPipeline, env::NodeID, model)
    raw = run_search(sampler, env, model)
    processed = preprocess(sampler, raw)
    return finalize_trajectory(sampler, processed)
end
