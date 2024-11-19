using Distributed
using Revise

number_of_workers = nworkers()
addprocs(number_of_workers)
# @everywhere using MyModule
# @everywhere using MyModule.LRUCache
# @everywhere using MyModule.SimpleChains
# @everywhere using MyModule.Mill

# using MyModule
# using MyModule.Metatheory
# using MyModule.Flux
# using MyModule.Mill
# using MyModule.DataFrames
# using MyModule.LRUCache
# using MyModule.SimpleChains
# using StatsBase
# using CSV
# using BSON
# using JLD2


# train_data_path = "./data/neural_rewrter/train.json"
# test_data_path = "./data/neural_rewrter/test.json"


# train_data = isfile(train_data_path) ? load_data(train_data_path)[1:100] : load_data(test_data_path)[1:1000]
# test_data = load_data(test_data_path)[1:100]
# train_data = preprosses_data_to_expressions(train_data)
# test_data = preprosses_data_to_expressions(test_data)

# @everywhere hidden_size = 128
# @everywhere simple_heuristic = ExprModelSimpleChains(ExprModel(
#     SimpleChain(static(length(new_all_symbols)), TurboDense{true}(SimpleChains.relu, hidden_size),TurboDense{true}(identity, hidden_size)),
#     Mill.SegmentedSum(hidden_size),
#     SimpleChain(static(2 * hidden_size + 2), TurboDense{true}(SimpleChains.relu, hidden_size),TurboDense{true}(identity, hidden_size)),
#     SimpleChain(static(hidden_size), TurboDense{true}(SimpleChains.relu, hidden_size),TurboDense{true}(identity, 1)),
# ))


# heuristic = ExprModel(
#     Flux.Chain(Dense(length(new_all_symbols), hidden_size, Flux.leakyrelu), Dense(hidden_size,hidden_size)),
#     Mill.SegmentedSum(hidden_size),
#     Flux.Chain(Dense(2*hidden_size + 2, hidden_size,Flux.leakyrelu), Dense(hidden_size, hidden_size)),
#     Flux.Chain(Dense(hidden_size, hidden_size,Flux.leakyrelu), Dense(hidden_size, 1))
#     )
# heuristic = MyModule.simple_to_flux(simple_heuristic, heuristic)

# epochs = 10
# optimizer = Adam()
# global training_samples = [Vector{TrainingSample}() for _ in 1:number_of_workers]
# # training_samples = Vector{TrainingSample}()
# max_steps = 1000
# max_depth = 60
# n = 10
# stats = []
# loss_stats = []
# proof_stats = []
# stp = div(n, number_of_workers)
# batched_train_data = [train_data[i:i + stp - 1] for i in 1:stp:n]
# dt = 1
# @everywhere exp_cache = LRU(maxsize=100_000)
# @everywhere cache = LRU(maxsize=1_000_000)
# @everywhere size_cache = LRU(maxsize=100_000)
# @everywhere expr_cache = LRU(maxsize=100_000)

# # exp_cache = LRU(maxsize=100_000)
# # cache = LRU(maxsize=1_000_000)
# # size_cache = LRU(maxsize=100_000)
# # expr_cache = LRU(maxsize=100_000)
# # global some_alpha = 1
# some_alpha = 0.05
# for ep in 1:epochs 
#     empty!(cache)
    
    
#     # train_heuristic!(heuristic, train_data, training_samples, max_steps, max_depth)
#     println("Epoch $ep")
#     println("Training===========================================")
#     global training_samples = pmap(dt -> train_heuristic!(simple_heuristic, batched_train_data[dt], training_samples[dt], max_steps, max_depth, all_symbols, theory, variable_names, cache, exp_cache, size_cache, expr_cache, ep - 1 + some_alpha), collect(1:number_of_workers))
#     # train_heuristic!(simple_heuristic, test_data[5:5], training_samples, max_steps, max_depth, new_all_symbols, theory, variable_names, cache, exp_cache, size_cache, expr_cache, some_alpha)
#     global simple_heuristic = MyModule.flux_to_simple(simple_heuristic, heuristic)
    
#     @assert 0 == 1
# end

# @everywhere using MyModule.Flux
# using Distributed

# # Define a simple model
# @everywhere model = Chain(Dense(2, 10, relu), Dense(10, 1))

# # Define a function to be run on workers
# @everywhere function some_function(model, data)
#     # Do some computation on model with data
#     return model(data)
# end

# # Define a function to update the model
# function update_model(model)
#     # Update logic for the model (e.g., train or modify the model)
#     # For example, let's just change the weights slightly here:
#     for l in layers(model)
#         Flux.reset!(l)
#     end
#     return model
# end

# # Example data
# data = [rand(2) for _ in 1:10]

# # Loop to perform the computation and update the model
# for ep in 1:10
#     # Perform parallel computation using pmap
#     result = pmap(d -> some_function(model, d), data)

#     # After pmap, update the model
#     global model = update_model(model)

#     # Optionally, print or inspect intermediate results
#     println("Epoch $ep complete.")
# end


# using SharedVector, LinearAlgebra

# Define a function that performs some computation and returns a vector
@everywhere function compute_function(x)
    # Here, just return a vector based on the input for illustration
    return [x^2, x^3, x^4]
end


# Function to use pmap and preserve the order of results
function parallel_computation_with_order(input_data)
    # Step 1: Pair each input with its index
    indexed_input = [(i, x) for (i, x) in enumerate(input_data)]
    
    # Step 2: Perform parallel computation using pmap
    results = pmap(pair -> compute_function(pair[2]), indexed_input)

    # Step 3: Sort the results by the original index
    # sorted_results = sortperm([pair[1] for pair in indexed_input])
    # ordered_results = [results[i] for i in sorted_results]

    return results
end

# Example usage
input_data = [1, 2, 3, 4, 5]
results = parallel_computation_with_order(input_data)

# Display the results
println("Ordered results:")
for result in results
    println(result)
end
