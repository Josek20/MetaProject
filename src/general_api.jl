abstract type MyEnv end

action_space(env::MyEnv)
state(env::MyEnv)
state_space(env::MyEnv)
reward(env::MyEnv)
is_terminated(env::MyEnv)
reset!(env::MyEnv)
act!(env::MyEnv, action)