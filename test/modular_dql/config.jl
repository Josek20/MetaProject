function TreeDQN()
    hidden_size=64
    input_size = 64

    function ffnn(idim, hidden_size, layers)
        layers == 1 && return Dense(idim, hidden_size, Flux.gelu)
        layers == 2 && return Chain(Dense(idim, hidden_size, Flux.gelu), Dense(hidden_size, hidden_size, Flux.gelu))
    end

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

    sampler = TreeSampler(max_steps=100, max_depth=100, epsilon=1.0, eps_decay=0.80, is_directed=false, n_best=5, batch=64)

    model = ExprModel(
        head_model,
        Mill.SegmentedSum(hidden_size),
        args_model,
        Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, 1))
        );
    target_model = ExprModel(
        head_model,
        Mill.SegmentedSum(hidden_size),
        args_model,
        Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, 1))
        );
    target_model = deepcopy(model)
    learner = DummyLerner(Flux.mse, model, max_iter=1)
    env = MyTreeEnv(data[300], model)
    pipeline = SimpleRLPipeline(env, model, sampler, learner, target_model)
    return pipeline
end

