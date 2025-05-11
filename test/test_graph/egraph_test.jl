using Flux
using GeometricFlux
using Graphs
using Random

# Set random seed for reproducibility
Random.seed!(42)

# 1. Create a sample graph (e.g., a simple undirected graph with 5 nodes)
g = SimpleGraph(5)
add_edge!(g, 1, 2)
add_edge!(g, 1, 3)
add_edge!(g, 2, 3)
add_edge!(g, 3, 4)
add_edge!(g, 4, 5)

# 2. Generate synthetic node features (e.g., 4 features per node)
num_nodes = nv(g)
num_features = 4
features = rand(Float32, num_features, num_nodes)  # Feature matrix: 4 x 5

# 3. Define ground-truth labels (e.g., binary classification for nodes)
labels = rand(Bool, num_nodes)  # Random binary labels for demonstration

# 4. Define a GCN model
model = Chain(
    GCNConv(num_features => 16, relu),  # GCN layer: 4 input features -> 16 hidden
    GCNConv(16 => 2),                   # GCN layer: 16 hidden -> 2 output classes
    softmax                            # Softmax for classification
)

# 5. Convert graph txo FeaturedGraph (GeometricFlux format)
input_dim = 4
hidden1_dim = 16
hidden2_dim = 32
encoder = Chain(
    WithGraph(fg, GCNConv(input_dim=>hidden1_dim, relu)),
    WithGraph(fg, GCNConv(hidden1_dim=>hidden2_dim)),
)

fg = FeaturedGraph(g)

# 6. Define loss function
loss(x, y) = Flux.crossentropy(model(x), Flux.onehotbatch(y, [false, true]))

# 7. Set up optimizer
opt = Flux.Adam(0.01)

# 8. Training loop (single epoch for simplicity)
ps = Flux.params(model)
data = [(fg, labels)]
for epoch in 1:10
    Flux.train!(loss, ps, data, opt)
    @info "Epoch $epoch" loss=loss(fg, labels)
end

# 9. Make predictions
predictions = model(fg)
predicted_labels = [argmax(predictions[:, i]) - 1 for i in 1:num_nodes]  # 0 or 1
println("Predicted labels: ", predicted_labels)


# parse string from log files to get the best
# content = read("tmp.txt", String)
# sections = split(content, "---")
# filtered_sections = filter(section -> occursin("final_rewards", section), sections)
# df = DataFrame()
# for i in filtered_sections
#     tmp = split(i,"\n")
#     # @show tmp[2], tmp[4]
#     pairs = split(tmp[3], ",")
#     # @show pairs
#     # Convert each pair into a NamedTuple
#     data = Dict()
#     for pair in pairs
#         key, val = split(pair, "=")
#         # Try to parse as number (Float64 or Int)
#         if occursin(".", val) || occursin("e", val)
#             data[key] = parse(Float64, val)
#         else
#             data[key] = parse(Int, val)
#         end
#     end
#     data["final_reward"] = parse(Float32, split(tmp[5], "= ")[end][1:end-3])
#     if isempty(df)
#         df = DataFrame(; (Symbol(k) => v for (k, v) in data)...)
#     else
#         push!(df, data) 
#     end
# end
# println(df)