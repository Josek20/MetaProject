using MyModule

experiment_name = "ppo"
train_data_path = "./data/neural_rewrter/train.json"
train_data = load_data(train_data_path)[1:1_000]
train_data = filter(x->!occursin("select", x[1]), train_data)
train_data = preprosses_data_to_expressions(train_data)
sorted_data = sort(train_data, by=x->MyModule.exp_size(x))
data = sorted_data

input_dim = 512
hidden_dim = 256
max_steps = 50
hidden_size = 64