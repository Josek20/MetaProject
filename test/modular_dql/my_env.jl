update_epsilon(epsilon; eps_decay=0.95, eps_min=0.1) = max(eps_min, eps_decay * epsilon) == eps_min ? 0.0 : max(eps_min, eps_decay * epsilon)

mutable struct MyTreeEnv{SI, SC, T, D, PM} <: AbstractEnvironment
    s_init::SI
    s_current::SC
    t::T
    is_done::D
    policy_model::PM
end
function MyTreeEnv(ex::Expr, policy_model)
    inex = intern!(ex)
    return MyTreeEnv(inex, inex, 1, false, policy_model)
end
function state(env::MyTreeEnv)
    # inference_type = MyModule.get_inference_type(env.s_current)
    # ds = MyModule.general_cached_inference(env.s_current, inference_type, env.policy_model)
    # return vec(ds)
    return env.s_current
end
function reset!(env::MyTreeEnv)
    env.s_current = env.s_init
    env.t = 1
    # return env
end
function action_space(env::MyTreeEnv)
    env.t += 1
    #new_ex, _ = all_expand(env.s_current, theory)
    all_actions = first(MyModule.all_expand(env.s_current, theory))
    tmp = filter(x->x!=env.s_current, all_actions)
    return tmp
end
function reward(env::MyTreeEnv, a, s)
    size_current = exp_size(a)
    size_init = exp_size(s)
    return size_init - size_current
end
function reward1(env::MyTreeEnv, a, s)
    size_current = exp_size(a)
    # size_init = exp_size(s)
    return -size_current
end
function act!(env::MyTreeEnv, a)
    env.s_current = a
end
is_terminal(env::MyTreeEnv) = env.is_done