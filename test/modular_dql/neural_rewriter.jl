function sample_trajectory(value_model, policy_model, env, sampler)
    reset!(env)
    soltree = Dict()
    traj = Trajectory()
    for t in 1:sampler.max_steps
        possible_subtree = action_space(env)
        if rand() >= sampler.epsilon
            # how to choose similar subtrees from different branches
            # we dont care if they are the same the first one wins
            o = [vec(value_model(pa)) for pa in possible_subtree]
            choosen_subtree = possible_subtree[argmax(o)]
        else
            choosen_subtree = rand(possible_actions)
        end
        s = state(env)
        a = (choosen_subtree, )
        act!(env, a)
        ns = state(env)
        r = reward(env, a, s)
        is_done = isempty(action_space(env)) || is_terminal(env)
        push2traj!(traj, (s,a,r,ns,is_done))
        if is_done
            break
        end
    end
end