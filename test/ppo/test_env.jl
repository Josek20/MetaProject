using Test
using Random
using ReinforcementLearning
using ReinforcementLearning: reward, is_terminated

@testset "CartPole Tests" begin
    # Test 1: Initialization
    @testset "Initialization" begin
        env = CartPoleEnv()  # Replace with your constructor
        
        # Check initial state dimensions (typically [cart_pos, cart_vel, pole_angle, pole_vel])
        @test length(env.state) == 4
        
        # Check initial state is within reasonable bounds
        @test abs(env.state[1]) <= 0.05  # cart position
        @test abs(env.state[2]) <= 0.05  # cart velocity
        @test abs(env.state[3]) <= 0.05  # pole angle
        @test abs(env.state[4]) <= 0.05  # pole velocity
    end

    # Test 2: Step function
    @testset "Step Function" begin
        env = CartPoleEnv()
        initial_state = copy(env.state)
        action = rand((1,2))  # Assuming 0 = left, 1 = right
        # next_state, reward, done, info = step!(env, action)
        act!(env, action)
        next_state = copy(env.state)
        rew = reward(env)
        done = is_terminated(env)
        # Check return types
        @test isa(next_state, Vector{Float64})
        @test isa(rew, Float64)
        @test isa(done, Bool)
        # @test isa(info, Dict)
        
        # Check state changed
        @test env.state != initial_state
        
        # Check reward is reasonable (typically 1.0 for survival)
        @test rew > 0.0
    end

    # Test 3: Reset function
    @testset "Reset" begin
        env = CartPoleEnv()
        # Modify state
        env.state = [1.0, 1.0, 1.0, 1.0]
        reset!(env)
        
        # Check state is reset to small values
        @test all(abs.(env.state) .< 0.1)
    end

    # Test 4: Termination conditions
    @testset "Termination" begin
        env = CartPoleEnv()
        
        # Test cart position boundary
        env.state = [2.5, 0.0, 0.0, 0.0]  # Beyond typical 2.4 limit
        act!(env, 1)
        done = is_terminated(env)
        # _, _, done, _ = 
        @test done == true
        
        # Test pole angle boundary
        env = CartPoleEnv()
        env.state = [0.0, 0.0, 0.5, 0.0]  # Beyond typical ~0.209 radians (12 degrees)
        # _, _, done, _ = step!(env, 0)
        act!(env, 1)
        @test done == true
    end

    # Test 5: Physics consistency
    @testset "Physics" begin
        env = CartPoleEnv()
        initial_pos = env.state[1]
        initial_vel = env.state[2]
        
        # Apply right force
        act!(env, 2)
        @test env.state[2] > initial_vel  # Velocity should increase
        @test env.state[1] > initial_pos  # Position should increase
        
        # Apply left force
        env = CartPoleEnv()
        initial_pos = env.state[1]
        initial_vel = env.state[2]
        act!(env, 1)
        @test env.state[2] < initial_vel  # Velocity should decrease
        @test env.state[1] < initial_pos  # Position should decrease
    end

    # Test 6: Action space
    @testset "Action Space" begin
        env = CartPoleEnv()
        @test_throws AssertionError act!(env, 3)  # Invalid action
        @test_throws AssertionError act!(env, -1) # Invalid action
    end
end

# Example minimal CartPole implementation for reference
# You can remove this if you have your own implementation
# mutable struct CartPoleEnv
#     state::Vector{Float64}
#     gravity::Float64
#     masscart::Float64
#     masspole::Float64
#     length::Float64
    
#     CartPoleEnv() = new(
#         zeros(4),    # [x, x_dot, theta, theta_dot]
#         9.81,        # gravity
#         1.0,         # masscart
#         0.1,         # masspole
#         0.5          # length
#     )
# end

# function reset!(env::CartPoleEnv)
#     env.state = randn(4) .* 0.05
#     return env.state
# end

# function step!(env::CartPoleEnv, action::Int)
#     # Simplified physics - replace with your actual implementation
#     force = (action == 1 ? 10.0 : -10.0)
#     env.state[2] += force * 0.01  # Simple velocity update
#     env.state[1] += env.state[2] * 0.01  # Simple position update
    
#     done = abs(env.state[1]) > 2.4 || abs(env.state[3]) > 0.209
#     reward = 1.0
#     info = Dict()
    
#     return env.state, reward, done, info
# end