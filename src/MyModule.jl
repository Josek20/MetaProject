module MyModule

using Flux
using JSON
using Metatheory
using .Metatheory.EGraphs: SaturationReport, eqsat_search!, eqsat_apply!, extract!
using GraphNeuralNetworks: GNNGraph, GCNConv

include("small_data_loader.jl")
include("egraph_processor.jl")
import .get_number_of_enods
include("graph_neural_network_module.jl")


end
