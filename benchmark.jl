using MyModule
using MyModule.Flux
using MyModule.Mill
using Serialization


exp_data = Vector()
open("data/training_data/benchmarking.bin", "r") do file
    data = deserialize(file)
    append!(exp_data, data)
end


hidden_size = 128
heuristic = ExprModel(
    Flux.Chain(Dense(length(symbols_to_index) + 1 + 13, hidden_size, relu), Dense(hidden_size,hidden_size)),
    Mill.SegmentedSumMax(hidden_size),
    Flux.Chain(Dense(3*hidden_size + 2, hidden_size,relu), Dense(hidden_size, hidden_size)),
    Flux.Chain(Dense(hidden_size, hidden_size,relu), Dense(hidden_size, 1))
    )



function compare_two_methods(data, model)
    cache = Dict()
    for ex in data[1:1000] 
        @show ex
        @time begin
            m1 = MyModule.cached_inference(ex, cache, model, new_all_symbols, sym_enc)
            o1 = MyModule.embed(model, m1)
        end
        @time begin
            m2 = MyModule.single_fast_ex2mill(ex, sym_enc)
            o2 = model(m2)
        end
        @show o1
        @show o2
        @assert abs(only(o1) - only(o2)) <= 0.000001
    end
end


function compare_two_methods2(data, model)
    cache = Dict()
    @time begin
        tmp = map(ex->MyModule.cached_inference(ex, cache, model, new_all_symbols, sym_enc), data)
        map(x->MyModule.embed(model, x), tmp)
    end
    @time begin
        m2 = MyModule.multiple_fast_ex2mill(data, sym_enc)
        ds = reduce(catobs, m2)
        o2 = model(ds)
    end
end
# compare_two_methods(exp_data, heuristic)
compare_two_methods2(exp_data, heuristic)
