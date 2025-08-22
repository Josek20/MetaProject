using Test
using MyModule

function get_data()
    train_data_path = "./data/neural_rewrter/train.json"
    train_data = load_data(train_data_path)[1:1_000]
    train_data = filter(x->!occursin("select", x[1]), train_data)
    train_data = preprosses_data_to_expressions(train_data)
    sorted_data = sort(train_data, by=x->MyModule.exp_size(x))
    data = sorted_data
    return data
end


function ffnn(idim, hidden_size, layers)
    layers == 1 && return Dense(idim, hidden_size, Flux.gelu)
    layers == 2 && return Chain(Dense(idim, hidden_size, Flux.gelu), Dense(hidden_size, hidden_size, Flux.gelu))
end
function create_test_model(;hidden_size=64, input_size=64)
    head_model = ProductModel(
    (;head = ffnn(length(new_all_symbols), hidden_size, 1),
      args = ffnn(hidden_size, hidden_size, 1),  
        ),
    ffnn(2*hidden_size, hidden_size, 1)
    )

    args_model = ProductModel(
        (;args = ffnn(hidden_size, hidden_size, 1),  
        position = Dense(2,hidden_size),  
            ),
        ffnn(2*hidden_size, hidden_size, 1)
        )

    model = ExprModel(
        head_model,
        Mill.SegmentedSum(hidden_size),
        args_model,
        Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, 1))
        );
    return model
end
@testset "check model inference" begin
    model = create_test_model()
    @testset "check inference cache" begin
        a = model(:(111 - 10 <= 1221))
        @assert length(MyModule.memoize_cache(MyModule.general_leaf_cached_inference)) == 3
    end
    @testset "check inference values collision" begin
        o1 = only(model(:(111 - 10 <= 1221)))
        o2 = only(model(:(111 - 10 <= 1221)))
        @assert o1 != o2
    end
end
@testset "DQL Pipeline Test" begin
    ex = get_data()[1]
    @testset "linear trajectory" begin
        sampler = LinearSampler(max_steps=50, epsilon=0.0, eps_decay=0.95)
        model = create_test_model()
        env = MyEnv(ex, model)
        trj = sample_trajectory(sampler, env, model)
        @assert !isempty(trj)
        # @assert 
    end
    @testset "tree trajectory" begin
        sampler = TreeSampler(max_steps=50, max_depth=100, epsilon=0.0, eps_decay=0.95, is_directed=false)
        model = create_test_model()
        env = MyTreeEnv(ex, model)
        trj = sample_trajectory(sampler, env, model)
        @assert !isempty(trj)
    end
    @testset "derected graph trajectory" begin
        sampler = TreeSampler(max_steps=50, max_depth=100, epsilon=0.0, eps_decay=0.95, is_directed=true)
        model = create_test_model()
        env = MyTreeEnv(ex, model)
        trj = sample_trajectory(sampler, env, model)
        @assert !isempty(trj)
    end
    @test "preprocess linear trajectory" begin
        sampler = LinearSampler(max_steps=50, epsilon=0.0, eps_decay=0.95)
        model = create_test_model()
        env = MyEnv(ex, model)
        trj = sample_trajectory(sampler, env, model) 
        preprocess(trj)
    end
    @test "preprocess tree trajectory" begin
        sampler = TreeSampler(max_steps=50, max_depth=100, epsilon=0.0, eps_decay=0.95, is_directed=false)
        model = create_test_model()
        env = MyEnv(ex, model)
        trj = sample_trajectory(sampler, env, model) 
    end
    @test "preprocess directed graph trajectory" begin
        sampler = TreeSampler(max_steps=50, max_depth=100, epsilon=0.0, eps_decay=0.95, is_directed=true)
        model = create_test_model()
        env = MyEnv(ex, model)
        trj = sample_trajectory(sampler, env, model) 
    end
end