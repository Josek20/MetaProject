using MyModule.Flux
using Optimisers


number_of_tokens = 10000

random_tokens = collect(1:number_of_tokens)
rand_y = 
encoding_size = 128
tokens_encodings = rand(Float32, (encoding_size, number_of_tokens))

tok2enc = Dict(t=>tokens_encodings[:,ind] for (ind,t) in enumerate(random_tokens))


n = 100
chosen_tokens = rand(random_tokens, n)

hidden_size = 128

model = Chain(
    Dense(encoding_size, hidden_size, gelu),
    Dense(hidden_size, hidden_size, gelu),
    Dense(hidden_size, 1),
)

Heaviside(x) = x > 0 ? 1 : 0
function loss(model, d, pos, neg, agg=mean, surrogate=softplus)
    o = vec(model(d))
    diff = o[pos] - 
    return agg(surrogate.(diff))
end


function preprocess_data(chosen_tokens, tok2enc, tokens_encodings, n=n)
    filtered_tok = filter(x->x ∉ chosen_tokens, keys(tok2enc))
    filtered_tok = [i for i in filtered_tok]
    # adding inconsistency
    push!(filtered_tok, rand(chosen_tokens))
    ######################
    ftk_len = length(filtered_tok)
    @show ftk_len
    split_step = div(ftk_len, n)
    splited_tok = [filtered_tok[i: (i + split_step > ftk_len ? ftk_len : i + split_step)] for i in 1:split_step:ftk_len]
    samples = map(enumerate(chosen_tokens)) do (ind, i)
        d = tokens_encodings
        pos = fill(i, length(splited_tok[ind]))
        neg = splited_tok[ind]
        (;d=d, pos=pos, neg=neg) 
    end
    return samples
end


function train(model, samples, epochs=100)
    optimizer=ADAM()
    opt_state = Flux.setup(optimizer, model)
    for ep in 1:epochs
        @show ep
        for (i, (d, pos, neg)) in enumerate(samples)
            sa, grad = Flux.Zygote.withgradient(model) do h
                loss(h, d, pos, neg)
            end
            # sum_loss += sa
            @show sa
            Optimisers.update!(opt_state, model, grad[1])
        end
        valid = [loss(model, d, pos, neg, sum, Heaviside) for (d, pos, neg) in samples]
        @show valid
        @show sum(valid)
        if sum(valid) == 0
            break
        end
    end
end

samples = preprocess_data(chosen_tokens, tok2enc, tokens_encodings)

train(model, samples)