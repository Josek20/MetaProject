using CSV
using DataFrames
using Statistics
using Plots
using StatsBase
using StatsPlots

function get_res(paths;start_epoch=1, final_epoch=20)
    tmp = map(paths) do path_name
        res = map(start_epoch:final_epoch) do e
            path = replace(path_name, "ep\$(ep)" => "ep$(e)")
            df = CSV.read(path, DataFrame)
            mean(df[!,1] .- df[!,2])
        end
    end
    return tmp
end
function get_res1(paths;start_epoch=1, final_epoch=20)
    tmp = map(paths) do path_name
        res = map(start_epoch:final_epoch) do e
            path = replace(path_name, "\$(ep)" => "$(e)")
            df = CSV.read(path, DataFrame)
            mean(df[!,1] .- df[!,2])
        end
    end
    return tmp
end
function get_res2(paths;start_epoch=1, final_epoch=20)
    tmp = map(paths) do path_name
        res = map(start_epoch:final_epoch) do e
            path = replace(path_name, "\$(ep)" => "$(e)")
            df = CSV.read(path, DataFrame)
            sum((df[!,1] .- df[!,2]) .!= 0)
        end
    end
    return tmp
end

function get_conv_res(paths;start_epoch=1, final_epoch=20)
    tmp = map(paths) do path_name
        res = map(start_epoch:final_epoch) do e
            path = replace(path_name, "ep\$(ep)" => "ep$(e)")
            df = CSV.read(path, DataFrame)
            # df[!,2] = isnan.(df[!,2]) ? 0.0 : df[!,2]
            mean(df[!,2])
        end
    end
    return tmp
end

function get_res3(paths;start_epoch=1, final_epoch=20)
    tmp = map(paths) do path_name
        res = map(start_epoch:final_epoch) do e
            path = replace(path_name, "ep\$(ep)" => "ep$(e)")
            df = CSV.read(path, DataFrame)
            # df[!,2] = isnan.(df[!,2]) ? 0.0 : df[!,2]
            mean(df[!,5])
        end
    end
    return tmp
end

function get_res4(paths;start_epoch=1, final_epoch=20)
    tmp = map(paths) do path_name
        res = map(start_epoch:final_epoch) do e
            path = replace(path_name, "ep\$(ep)" => "ep$(e)")
            df = CSV.read(path, DataFrame)
            # df[!,2] = isnan.(df[!,2]) ? 0.0 : df[!,2]
            mean(df[!,6])
        end
    end
    return tmp
end

function get_res5(paths;start_epoch=1, final_epoch=20)
    tmp = map(paths) do path_name
        res = map(start_epoch:final_epoch) do e
            path = replace(path_name, "ep\$(ep)" => "ep$(e)")
            df = CSV.read(path, DataFrame)
            # df[!,2] = isnan.(df[!,2]) ? 0.0 : df[!,2]
            mean(df[!,7])
        end
    end
    return tmp
end


function process_model(paths::Vector{String}; get_res::Function=get_res, final_epoch=20)
    res = hcat(get_res(paths, final_epoch=final_epoch)...)
    means = vec(mean(res, dims=2))
    stds = vec(std(res, dims=2))
    return means, stds
end


function plot_all_results()
    models_results = Dict(
        "Planning" => (
            train=["stats/planning_4th_20ep/results_of_test_heuristic_boosted_1h_heuristic_ep\$(ep)_hidden64.csv",
                "stats/planning_3rd_20ep/results_of_test_heuristic_boosted_1h_heuristic_ep\$(ep)_hidden64.csv",
                "stats/planning_2nd_20ep/results_of_test_heuristic_boosted_1h_heuristic_ep\$(ep)_hidden64.csv"],
            test=["stats/planning_4th_20ep/results_of_test_heuristic_boosted_1h_test_heuristic_ep\$(ep)_hidden64.csv",
                "stats/planning_3rd_20ep/results_of_test_heuristic_boosted_1h_test_heuristic_ep\$(ep)_hidden64.csv",
                "stats/planning_2nd_20ep/results_of_test_heuristic_boosted_1h_test_heuristic_ep\$(ep)_hidden64.csv"]
        ),
        # VL 1 and 2 on Tree
        "VL 1 Tree" => (
            train=["stats/dqn_first_2nd_20ep/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_3rd_20ep/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_4th_20ep/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            test=["stats/dqn_first_2nd_20ep/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_3rd_20ep/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_4th_20ep/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"]
        ),
        "VL 2 Tree" => (
            train=["stats/dqn_first_2nd_20ep/results_of_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_3rd_20ep/results_of_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_4th_20ep/results_of_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            test=["stats/dqn_first_2nd_20ep/results_of_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_3rd_20ep/results_of_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_4th_20ep/results_of_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"]
        ),
        # VL 1 and 2 on DAG
        "VL 1 DAG" => (
            train=["stats/dqn_first_2nd_20ep/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_3rd_20ep/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_4th_20ep/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            test=["stats/dqn_first_2nd_20ep/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_3rd_20ep/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_4th_20ep/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"]
        ),
        "VL 2 DAG" => (
            train=["stats/dqn_first_2nd_20ep/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_3rd_20ep/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_4th_20ep/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            test=["stats/dqn_first_2nd_20ep/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_3rd_20ep/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_4th_20ep/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"]
        ),
        # VL 1 and 2 on DG
        "VL 1 DG" => (
            train=["stats/dqn_first_2nd_20ep/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_3rd_20ep/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_4th_20ep/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            test=["stats/dqn_first_2nd_20ep/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_3rd_20ep/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_4th_20ep/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"]
        ),
        "VL 2 DG" => (
            train=["stats/dqn_first_2nd_20ep/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_3rd_20ep/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_4th_20ep/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            test=["stats/dqn_first_2nd_20ep/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_3rd_20ep/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_4th_20ep/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"]
        ),
    )

    plot(size=(800, 600), xlabel="Epochs", ylabel="Mean simplification", title="Model Comparison on Test Set")

    for (label, data) in models_results
        means_test, _ = process_model(data.train)
        plot!(1:length(means_test), means_test, label="$label Mean", lw=2)
    end
    plot!()
end



function plot_all_results_gamma()
    models_results = Dict(
        "Planning" => (
            train=["stats/planning_4th_20ep/results_of_test_heuristic_boosted_1h_heuristic_ep\$(ep)_hidden64.csv",
                "stats/planning_3rd_20ep/results_of_test_heuristic_boosted_1h_heuristic_ep\$(ep)_hidden64.csv",
                "stats/planning_2nd_20ep/results_of_test_heuristic_boosted_1h_heuristic_ep\$(ep)_hidden64.csv"],
            test=["stats/planning_4th_20ep/results_of_test_heuristic_boosted_1h_test_heuristic_ep\$(ep)_hidden64.csv",
                "stats/planning_3rd_20ep/results_of_test_heuristic_boosted_1h_test_heuristic_ep\$(ep)_hidden64.csv",
                "stats/planning_2nd_20ep/results_of_test_heuristic_boosted_1h_test_heuristic_ep\$(ep)_hidden64.csv"]
        ),
        # VL 1 and 2 on Tree
        "VL 1 Tree" => (
            train=["stats/dqn_first_4th_20ep/results_of_trained_DQN_first_DAG1_ep\$(ep)_hidden64.csv",
                "stats/dqn_first_3rd_20ep/results_of_trained_DQN_first_DAG1_ep\$(ep)_hidden64.csv",
                "stats/dqn_first_2nd_20ep/results_of_trained_DQN_first_DAG1_not_boosted_ep\$(ep)_hidden64.csv"],
            test=["stats/dqn_first_4th_20ep/results_of_test_trained_not_boosted_DQN_first_DAG1_ep\$(ep)_hidden64.csv",
                "stats/dqn_first_3rd_20ep/results_of_test_trained_DQN_first_DAG1_ep\$(ep)_hidden64.csv",
                "stats/dqn_first_2nd_20ep/results_of_test_trained_DQN_first_DAG1_ep\$(ep)_hidden64.csv"]
        ),
        "VL 2 Tree" => (
            train=["stats/dqn_first_4th_20ep/results_of_trained_DQN_second_DAG1_not_boosted_ep\$(ep)_hidden64.csv",
                "stats/dqn_first_3rd_20ep/results_of_trained_DQN_second_DAG1_not_boosted_ep\$(ep)_hidden64.csv",
                "stats/dqn_first_2nd_20ep/results_of_trained_DQN_second_DAG1_ep\$(ep)_hidden64.csv"],
            test=["stats/dqn_first_4th_20ep/results_of_test_trained_DQN_second_DAG1_not_boosted_ep\$(ep)_hidden64.csv",
                "stats/dqn_first_3rd_20ep/results_of_test_trained_DQN_second_DAG1_not_boosted_ep\$(ep)_hidden64.csv",
                "stats/dqn_first_2nd_20ep/results_of_test_trained_DQN_second_DAG1_ep\$(ep)_hidden64.csv"]
        ),
        # VL 1 and 2 on DAG
        "VL 1 DAG" => (
            train=["stats/dqn_first_2nd_20ep/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
                "stats/dqn_first_3rd_20ep/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
                "stats/dqn_first_4th_20ep/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_hidden64.csv"],
            test=["stats/dqn_first_2nd_20ep/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
                "stats/dqn_first_3rd_20ep/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
                "stats/dqn_first_4th_20ep/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_hidden64.csv"]
        ),
        "VL 2 DAG" => (
            train=["stats/dqn_first_2nd_20ep/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
                "stats/dqn_first_3rd_20ep/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
                "stats/dqn_first_4th_20ep/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_hidden64.csv"],
            test=["stats/dqn_first_2nd_20ep/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
                "stats/dqn_first_3rd_20ep/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
                "stats/dqn_first_4th_20ep/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_hidden64.csv"]
        ),
        # VL 1 and 2 on DG
        "VL 1 DG" => (
            train=["stats/dqn_first_2nd_20ep/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
                "stats/dqn_first_3rd_20ep/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
                "stats/dqn_first_4th_20ep/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_hidden64.csv"],
            test=["stats/dqn_first_2nd_20ep/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
                "stats/dqn_first_3rd_20ep/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
                "stats/dqn_first_4th_20ep/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_hidden64.csv"]
        ),
        "VL 2 DG" => (
            train=["stats/dqn_first_2nd_20ep/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
                "stats/dqn_first_3rd_20ep/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
                "stats/dqn_first_4th_20ep/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_hidden64.csv"],
            test=["stats/dqn_first_2nd_20ep/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
                "stats/dqn_first_3rd_20ep/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
                "stats/dqn_first_4th_20ep/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_hidden64.csv"]
        ),
    )

    plot(size=(800, 600), xlabel="Epochs", ylabel="Mean simplification", title="Model Comparison on Test Set")

    for (label, data) in models_results
        means_test, _ = process_model(data.train)
        plot!(1:length(means_test), means_test, label="$label Test Mean", lw=2)
    end
    plot!()
end


function plot_fixed_results_gamma()
    models_results = Dict(
        "Planning" => (
            train=["stats/planning_4th_20ep/results_of_test_heuristic_boosted_1h_heuristic_ep\$(ep)_hidden64.csv",
                "stats/planning_3rd_20ep/results_of_test_heuristic_boosted_1h_heuristic_ep\$(ep)_hidden64.csv",
                "stats/planning_2nd_20ep/results_of_test_heuristic_boosted_1h_heuristic_ep\$(ep)_hidden64.csv"],
            test=["stats/planning_4th_20ep/results_of_test_heuristic_boosted_1h_test_heuristic_ep\$(ep)_hidden64.csv",
                "stats/planning_3rd_20ep/results_of_test_heuristic_boosted_1h_test_heuristic_ep\$(ep)_hidden64.csv",
                "stats/planning_2nd_20ep/results_of_test_heuristic_boosted_1h_test_heuristic_ep\$(ep)_hidden64.csv"],
            val=["stats/planning_4th_20ep/results_of_test_heuristic_boosted_1h_val_heuristic_ep\$(ep)_hidden64.csv",
            "stats/planning_3rd_20ep/results_of_test_heuristic_boosted_1h_val_heuristic_ep\$(ep)_hidden64.csv",
            "stats/planning_2nd_20ep/results_of_test_heuristic_boosted_1h_val_heuristic_ep\$(ep)_hidden64.csv"]
        ),
        # results_of_all_search_trained__l_star_boosted_2dense_97_innep10_epsilon_hidden64
        # results_of_all_search_trained_test__l_star_boosted_2dense_28_innep10_epsilon_hidden64
        "Planning100ep_3dense" => (
            train=["stats/planning_100exp_100ep/results_of_all_search_trained__l_star_boosted_3dense_\$(ep)_innep10_no_epsilon_hidden64.csv",],
            test=["stats/planning_100exp_100ep/results_of_all_search_trained_test__l_star_boosted_3dense_\$(ep)_innep10_no_epsilon_hidden64.csv",],
            val=["stats/planning_100exp_100ep/results_of_all_search_trained_val__l_star_boosted_3dense_\$(ep)_innep10_no_epsilon_hidden64.csv",]
        ),
        "Planning100ep_2dense" => (
            train=["stats/planning_100exp_100ep/results_of_all_search_trained__l_star_boosted_2dense_\$(ep)_innep10_no_epsilon_hidden64.csv",],
            test=["stats/planning_100exp_100ep/results_of_all_search_trained_test__l_star_boosted_2dense_\$(ep)_innep10_no_epsilon_hidden64.csv",],
            val=["stats/planning_100exp_100ep/results_of_all_search_trained_val__l_star_boosted_2dense_\$(ep)_innep10_no_epsilon_hidden64.csv",]
        ),
        "Planning100ep_3dense_epsilon" => (
            train=["stats/planning_100exp_100ep/results_of_all_search_trained__l_star_boosted_3dense_\$(ep)_innep10_epsilon_hidden64.csv",],
            test=["stats/planning_100exp_100ep/results_of_all_search_trained_test__l_star_boosted_3dense_\$(ep)_innep10_epsilon_hidden64.csv",],
            val=["stats/planning_100exp_100ep/results_of_all_search_trained_val__l_star_boosted_3dense_\$(ep)_innep10_epsilon_hidden64.csv",]
        ),
        "Planning100ep_2dense_epsilon" => (
            train=["stats/planning_100exp_100ep/results_of_all_search_trained__l_star_boosted_2dense_\$(ep)_innep10_epsilon_hidden64.csv",],
            test=["stats/planning_100exp_100ep/results_of_all_search_trained_test__l_star_boosted_2dense_\$(ep)_innep10_epsilon_hidden64.csv",],
            val=["stats/planning_100exp_100ep/results_of_all_search_trained_val__l_star_boosted_2dense_\$(ep)_innep10_epsilon_hidden64.csv",]
        ),

        "Planning1000ep_3dense" => (
            train=["stats/planning_100exp_1000ep/results_of_all_search_trained__l_star_boosted_3dense_\$(ep)_innep10_no_epsilon_hidden64.csv",],
            test=["stats/planning_100exp_1000ep/results_of_all_search_trained_test__l_star_boosted_3dense_\$(ep)_innep10_no_epsilon_hidden64.csv",],
            val=["stats/planning_100exp_1000ep/results_of_all_search_trained_val__l_star_boosted_3dense_\$(ep)_innep10_no_epsilon_hidden64.csv",]
        ),
        "Planning1000ep_2dense" => (
            train=["stats/planning_100exp_1000ep/results_of_all_search_trained__l_star_boosted_2dense_\$(ep)_innep10_no_epsilon_hidden64.csv",],
            test=["stats/planning_100exp_1000ep/results_of_all_search_trained_test__l_star_boosted_2dense_\$(ep)_innep10_no_epsilon_hidden64.csv",],
            val=["stats/planning_100exp_1000ep/results_of_all_search_trained_val__l_star_boosted_2dense_\$(ep)_innep10_no_epsilon_hidden64.csv",]
        ),
        "Planning1000ep_3dense_epsilon" => (
            train=["stats/planning_100exp_1000ep/results_of_all_search_trained__l_star_boosted_3dense_\$(ep)_innep10_epsilon_hidden64.csv",],
            test=["stats/planning_100exp_1000ep/results_of_all_search_trained_test__l_star_boosted_3dense_\$(ep)_innep10_epsilon_hidden64.csv",],
            val=["stats/planning_100exp_1000ep/results_of_all_search_trained_val__l_star_boosted_3dense_\$(ep)_innep10_epsilon_hidden64.csv",]
        ),
        "Planning1000ep_2dense_epsilon" => (
            train=["stats/planning_100exp_1000ep/results_of_all_search_trained__l_star_boosted_2dense_\$(ep)_innep10_epsilon_hidden64.csv",],
            test=["stats/planning_100exp_1000ep/results_of_all_search_trained_test__l_star_boosted_2dense_\$(ep)_innep10_epsilon_hidden64.csv",],
            val=["stats/planning_100exp_1000ep/results_of_all_search_trained_val__l_star_boosted_2dense_\$(ep)_innep10_epsilon_hidden64.csv",]
        ),
        
        # "Planning100ep_3dense_1innep" => (
        #     train=["stats/planning_3rd_100ep/results_of_all_search_trained__l_star_boosted_3dense_\$(ep)_innep1_hidden64.csv",],
        #     test=["stats/planning_3rd_100ep/results_of_all_search_trained_test__l_star_boosted_3dense_\$(ep)_innep1_hidden64.csv",],
        #     val=["stats/planning_3rd_100ep/results_of_all_search_trained_val__l_star_boosted_3dense_\$(ep)_innep1_hidden64.csv",]
        # ),
        # "Planning100ep_2dense_1innep" => (
        #     train=["stats/planning_3rd_100ep/results_of_all_search_trained__l_star_boosted_2dense_\$(ep)_innep1_hidden64.csv",],
        #     test=["stats/planning_3rd_100ep/results_of_all_search_trained_test__l_star_boosted_2dense_\$(ep)_innep1_hidden64.csv",],
        #     val=["stats/planning_3rd_100ep/results_of_all_search_trained_val__l_star_boosted_2dense_\$(ep)_innep1_hidden64.csv",]
        # ),
        # "Planning100ep_3dense_epsilon_1innep" => (
        #     train=["stats/planning_3rd_100ep/results_of_all_search_trained__l_star_boosted_3dense_\$(ep)_innep1_epsilon_hidden64.csv",],
        #     test=["stats/planning_3rd_100ep/results_of_all_search_trained_test__l_star_boosted_3dense_\$(ep)_innep1_epsilon_hidden64.csv",],
        #     val=["stats/planning_3rd_100ep/results_of_all_search_trained_val__l_star_boosted_3dense_\$(ep)_innep1_epsilon_hidden64.csv",]
        # ),
        # "Planning100ep_2dense_epsilon_1innep" => (
        #     train=["stats/planning_3rd_100ep/results_of_all_search_trained__l_star_boosted_2dense_\$(ep)_innep1_epsilon_hidden64.csv",],
        #     test=["stats/planning_3rd_100ep/results_of_all_search_trained_test__l_star_boosted_2dense_\$(ep)_innep1_epsilon_hidden64.csv",],
        #     val=["stats/planning_3rd_100ep/results_of_all_search_trained_val__l_star_boosted_2dense_\$(ep)_innep1_epsilon_hidden64.csv",]
        # ),
        # results_of_all_search_trained_test__l_star_boosted_3dense_100_innep10_epsilon_hidden64
        # "Planning" => (
        #     train=["stats/planning_2nd_20ep/results_of_all_search_test_heuristic_boosted_1h_heuristic_ep\$(ep)_epsilon_hidden64.csv",
        #     "stats/planning_3rd_20ep/results_of_all_search_test_heuristic_boosted_1h_heuristic_ep\$(ep)_epsilon_hidden64.csv",
        #     "stats/planning_4th_20ep/results_of_all_search_test_heuristic_boosted_1h_heuristic_ep\$(ep)_epsilon_hidden64.csv"],
        #     test=["stats/planning_2nd_20ep/results_of_all_search_test_heuristic_boosted_1h_heuristic_test_ep\$(ep)_epsilon_hidden64.csv",
        #     "stats/planning_3rd_20ep/results_of_all_search_test_heuristic_boosted_1h_heuristic_test_ep\$(ep)_epsilon_hidden64.csv",
        #     "stats/planning_4th_20ep/results_of_all_search_test_heuristic_boosted_1h_heuristic_test_ep\$(ep)_epsilon_hidden64.csv"]
        # ),
        # VL 1 and 2 on Tree
        "VL Tree" => (
            train_gamma1=["stats/dqn_first_Tree_gamma1/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_Tree_gamma1_2nd/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_Tree_gamma1_3rd/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            test_gamma1=["stats/dqn_first_Tree_gamma1/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_Tree_gamma1_2nd/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_Tree_gamma1_3rd/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            val_gamma1=["stats/dqn_first_Tree_gamma1/results_of_trained_val_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_Tree_gamma1_2nd/results_of_trained_val_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_Tree_gamma1_3rd/results_of_trained_val_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            train_gamma09=["stats/dqn_first_Tree_gamma09/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv", 
            "stats/dqn_first_Tree_gamma09_2nd/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_first_Tree_gamma09_3rd/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"],
            test_gamma09=["stats/dqn_first_Tree_gamma09/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_first_Tree_gamma09_2nd/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_first_Tree_gamma09_3rd/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"],
            val_gamma09=["stats/dqn_first_Tree_gamma09/results_of_trained_val_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_first_Tree_gamma09_2nd/results_of_trained_val_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_first_Tree_gamma09_3rd/results_of_trained_val_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"],
            train_gamma08=["stats/dqn_first_Tree_gamma08/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_Tree_gamma08_2nd/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_Tree_gamma08_3rd/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"],
            test_gamma08=["stats/dqn_first_Tree_gamma08/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_Tree_gamma08_2nd/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_Tree_gamma08_3rd/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"],
            val_gamma08=["stats/dqn_first_Tree_gamma08/results_of_trained_val_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_Tree_gamma08_2nd/results_of_trained_val_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_Tree_gamma08_3rd/results_of_trained_val_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"],
            train_gamma99=["stats/dqn_first_Tree_gamma99/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
            "stats/dqn_first_Tree_gamma99_2nd/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
            "stats/dqn_first_Tree_gamma99_3rd/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"],
            test_gamma99=["stats/dqn_first_Tree_gamma99/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
            "stats/dqn_first_Tree_gamma99_2nd/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
            "stats/dqn_first_Tree_gamma99_3rd/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"],
            val_gamma99=["stats/dqn_first_Tree_gamma99/results_of_trained_val_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
            "stats/dqn_first_Tree_gamma99_2nd/results_of_trained_val_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
            "stats/dqn_first_Tree_gamma99_3rd/results_of_trained_val_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"],
            train_gamma95=["stats/dqn_first_Tree_gamma95/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
            "stats/dqn_first_Tree_gamma95_2nd/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
            "stats/dqn_first_Tree_gamma95_3rd/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"],
            test_gamma95=["stats/dqn_first_Tree_gamma95/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
            "stats/dqn_first_Tree_gamma95_2nd/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
            "stats/dqn_first_Tree_gamma95_3rd/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"],
            val_gamma95=["stats/dqn_first_Tree_gamma95/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
            "stats/dqn_first_Tree_gamma95_2nd/results_of_trained_val_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
            "stats/dqn_first_Tree_gamma95_3rd/results_of_trained_val_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"],
        ),
        "RTDP Tree" => (
            train_gamma1 = [
                "stats/dqn_second_Tree_gamma1/results_of_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_Tree_gamma1_2nd/results_of_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_Tree_gamma1_3rd/results_of_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"
            ],
            test_gamma1 = [
                "stats/dqn_second_Tree_gamma1/results_of_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_Tree_gamma1_2nd/results_of_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_Tree_gamma1_3rd/results_of_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"
            ],
            val_gamma1 = [
                "stats/dqn_second_Tree_gamma1/results_of_trained_val_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_Tree_gamma1_2nd/results_of_trained_val_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_Tree_gamma1_3rd/results_of_trained_val_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"
            ],

            train_gamma09 = [
                "stats/dqn_second_Tree_gamma09/results_of_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_Tree_gamma09_2nd/results_of_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_Tree_gamma09_3rd/results_of_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"
            ],
            test_gamma09 = [
                "stats/dqn_second_Tree_gamma09/results_of_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_Tree_gamma09_2nd/results_of_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_Tree_gamma09_3rd/results_of_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"
            ],
            val_gamma09 = [
                "stats/dqn_second_Tree_gamma09/results_of_trained_val_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_Tree_gamma09_2nd/results_of_trained_val_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_Tree_gamma09_3rd/results_of_trained_val_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"
            ],

            train_gamma08 = [
                "stats/dqn_second_Tree_gamma08/results_of_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_Tree_gamma08_2nd/results_of_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_Tree_gamma08_3rd/results_of_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"
            ],
            test_gamma08 = [
                "stats/dqn_second_Tree_gamma08/results_of_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_Tree_gamma08_2nd/results_of_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_Tree_gamma08_3rd/results_of_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"
            ],
            val_gamma08 = [
                "stats/dqn_second_Tree_gamma08/results_of_trained_val_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_Tree_gamma08_2nd/results_of_trained_val_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_Tree_gamma08_3rd/results_of_trained_val_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"
            ],

            train_gamma99 = [
                "stats/dqn_second_Tree_gamma99/results_of_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_Tree_gamma99_2nd/results_of_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_Tree_gamma99_3rd/results_of_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"
            ],
            test_gamma99 = [
                "stats/dqn_second_Tree_gamma99/results_of_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_Tree_gamma99_2nd/results_of_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_Tree_gamma99_3rd/results_of_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"
            ],
            val_gamma99 = [
                "stats/dqn_second_Tree_gamma99/results_of_trained_val_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_Tree_gamma99_2nd/results_of_trained_val_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_Tree_gamma99_3rd/results_of_trained_val_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"
            ],

            train_gamma95 = [
                "stats/dqn_second_Tree_gamma95/results_of_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_Tree_gamma95_2nd/results_of_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_Tree_gamma95_3rd/results_of_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"
            ],
            test_gamma95 = [
                "stats/dqn_second_Tree_gamma95/results_of_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_Tree_gamma95_2nd/results_of_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_Tree_gamma95_3rd/results_of_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"
            ],
            val_gamma95 = [
                "stats/dqn_second_Tree_gamma95/results_of_trained_val_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_Tree_gamma95_2nd/results_of_trained_val_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_Tree_gamma95_3rd/results_of_trained_val_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"
            ],
        ),

        
        # VL 1 and 2 on DAG
        "VL DAG" => (
            train_gamma1 = [
                "stats/dqn_first_DAG_gamma1/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_DAG_gamma1_2nd/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_DAG_gamma1_3rd/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"
            ],
            test_gamma1 = [
                "stats/dqn_first_DAG_gamma1/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_DAG_gamma1_2nd/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_DAG_gamma1_3rd/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"
            ],
            val_gamma1 = [
                "stats/dqn_first_DAG_gamma1/results_of_trained_val_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_DAG_gamma1_2nd/results_of_trained_val_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_DAG_gamma1_3rd/results_of_trained_val_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"
            ],

            train_gamma09 = [
                "stats/dqn_first_DAG_gamma09/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_first_DAG_gamma09_2nd/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_first_DAG_gamma09_3rd/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"
            ],
            test_gamma09 = [
                "stats/dqn_first_DAG_gamma09/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_first_DAG_gamma09_2nd/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_first_DAG_gamma09_3rd/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"
            ],
            val_gamma09 = [
                "stats/dqn_first_DAG_gamma09/results_of_trained_val_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_first_DAG_gamma09_2nd/results_of_trained_val_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_first_DAG_gamma09_3rd/results_of_trained_val_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"
            ],

            train_gamma08 = [
                "stats/dqn_first_DAG_gamma08/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_first_DAG_gamma08_2nd/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_first_DAG_gamma08_3rd/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"
            ],
            test_gamma08 = [
                "stats/dqn_first_DAG_gamma08/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_first_DAG_gamma08_2nd/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_first_DAG_gamma08_3rd/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"
            ],
            val_gamma08 = [
                "stats/dqn_first_DAG_gamma08/results_of_trained_val_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_first_DAG_gamma08_2nd/results_of_trained_val_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_first_DAG_gamma08_3rd/results_of_trained_val_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"
            ],

            train_gamma99 = [
                "stats/dqn_first_DAG_gamma99/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_first_DAG_gamma99_2nd/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_first_DAG_gamma99_3rd/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"
            ],
            test_gamma99 = [
                "stats/dqn_first_DAG_gamma99/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_first_DAG_gamma99_2nd/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_first_DAG_gamma99_3rd/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"
            ],
            val_gamma99 = [
                "stats/dqn_first_DAG_gamma99/results_of_trained_val_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_first_DAG_gamma99_2nd/results_of_trained_val_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_first_DAG_gamma99_3rd/results_of_trained_val_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"
            ],

            train_gamma95 = [
                "stats/dqn_first_DAG_gamma95/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_first_DAG_gamma95_2nd/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_first_DAG_gamma95_3rd/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"
            ],
            test_gamma95 = [
                "stats/dqn_first_DAG_gamma95/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_first_DAG_gamma95_2nd/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_first_DAG_gamma95_3rd/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"
            ],
            val_gamma95 = [
                "stats/dqn_first_DAG_gamma95/results_of_trained_val_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_first_DAG_gamma95_2nd/results_of_trained_val_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_first_DAG_gamma95_3rd/results_of_trained_val_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"
            ]
        ),

        "RTDP DAG" => (
            train_gamma1 = [
                "stats/dqn_second_DAG_gamma1/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_DAG_gamma1_2nd/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_DAG_gamma1_3rd/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"
            ],
            test_gamma1 = [
                "stats/dqn_second_DAG_gamma1/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_DAG_gamma1_2nd/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_DAG_gamma1_3rd/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"
            ],
            val_gamma1 = [
                "stats/dqn_second_DAG_gamma1/results_of_trained_val_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_DAG_gamma1_2nd/results_of_trained_val_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_DAG_gamma1_3rd/results_of_trained_val_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"
            ],

            train_gamma09 = [
                "stats/dqn_second_DAG_gamma09/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_DAG_gamma09_2nd/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_DAG_gamma09_3rd/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"
            ],
            test_gamma09 = [
                "stats/dqn_second_DAG_gamma09/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_DAG_gamma09_2nd/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_DAG_gamma09_3rd/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"
            ],
            val_gamma09 = [
                "stats/dqn_second_DAG_gamma09/results_of_trained_val_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_DAG_gamma09_2nd/results_of_trained_val_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_DAG_gamma09_3rd/results_of_trained_val_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"
            ],

            train_gamma08 = [
                "stats/dqn_second_DAG_gamma08/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_DAG_gamma08_2nd/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_DAG_gamma08_3rd/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"
            ],
            test_gamma08 = [
                "stats/dqn_second_DAG_gamma08/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_DAG_gamma08_2nd/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_DAG_gamma08_3rd/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"
            ],
            val_gamma08 = [
                "stats/dqn_second_DAG_gamma08/results_of_trained_val_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_DAG_gamma08_2nd/results_of_trained_val_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_DAG_gamma08_3rd/results_of_trained_val_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"
            ],

            train_gamma99 = [
                "stats/dqn_second_DAG_gamma99/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_DAG_gamma99_2nd/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_DAG_gamma99_3rd/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"
            ],
            test_gamma99 = [
                "stats/dqn_second_DAG_gamma99/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_DAG_gamma99_2nd/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_DAG_gamma99_3rd/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"
            ],
            val_gamma99 = [
                "stats/dqn_second_DAG_gamma99/results_of_trained_val_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_DAG_gamma99_2nd/results_of_trained_val_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_DAG_gamma99_3rd/results_of_trained_val_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"
            ],

            train_gamma95 = [
                "stats/dqn_second_DAG_gamma95/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_DAG_gamma95_2nd/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_DAG_gamma95_3rd/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"
            ],
            test_gamma95 = [
                "stats/dqn_second_DAG_gamma95/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_DAG_gamma95_2nd/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_DAG_gamma95_3rd/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"
            ],
            val_gamma95 = [
                "stats/dqn_second_DAG_gamma95/results_of_trained_val_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_DAG_gamma95_2nd/results_of_trained_val_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_DAG_gamma95_3rd/results_of_trained_val_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"
            ]
        ),

        
        # VL 1 and 2 on DG
        "VL DG" => (
            train_gamma1 = [
                "stats/dqn_first_DG_gamma1/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_DG_gamma1_2nd/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_DG_gamma1_3rd/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"
            ],
            test_gamma1 = [
                "stats/dqn_first_DG_gamma1/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_DG_gamma1_2nd/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_DG_gamma1_3rd/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"
            ],
            val_gamma1 = [
                "stats/dqn_first_DG_gamma1/results_of_trained_val_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_DG_gamma1_2nd/results_of_trained_val_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_DG_gamma1_3rd/results_of_trained_val_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"
            ],

            train_gamma09 = [
                "stats/dqn_first_DG_gamma09/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_first_DG_gamma09_2nd/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_first_DG_gamma09_3rd/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"
            ],
            test_gamma09 = [
                "stats/dqn_first_DG_gamma09/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_first_DG_gamma09_2nd/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_first_DG_gamma09_3rd/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"
            ],
            val_gamma09 = [
                "stats/dqn_first_DG_gamma09/results_of_trained_val_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_first_DG_gamma09_2nd/results_of_trained_val_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_first_DG_gamma09_3rd/results_of_trained_val_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"
            ],

            train_gamma08 = [
                "stats/dqn_first_DG_gamma08/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_first_DG_gamma08_2nd/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_first_DG_gamma08_3rd/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"
            ],
            test_gamma08 = [
                "stats/dqn_first_DG_gamma08/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_first_DG_gamma08_2nd/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_first_DG_gamma08_3rd/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"
            ],
            val_gamma08 = [
                "stats/dqn_first_DG_gamma08/results_of_trained_val_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_first_DG_gamma08_2nd/results_of_trained_val_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_first_DG_gamma08_3rd/results_of_trained_val_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"
            ],

            train_gamma99 = [
                "stats/dqn_first_DG_gamma99/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_first_DG_gamma99_2nd/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_first_DG_gamma99_3rd/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"
            ],
            test_gamma99 = [
                "stats/dqn_first_DG_gamma99/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_first_DG_gamma99_2nd/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_first_DG_gamma99_3rd/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"
            ],
            val_gamma99 = [
                "stats/dqn_first_DG_gamma99/results_of_trained_val_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_first_DG_gamma99_2nd/results_of_trained_val_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_first_DG_gamma99_3rd/results_of_trained_val_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"
            ],

            train_gamma95 = [
                "stats/dqn_first_DG_gamma95/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_first_DG_gamma95_2nd/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_first_DG_gamma95_3rd/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"
            ],
            test_gamma95 = [
                "stats/dqn_first_DG_gamma95/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_first_DG_gamma95_2nd/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_first_DG_gamma95_3rd/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"
            ],
            val_gamma95 = [
                "stats/dqn_first_DG_gamma95/results_of_trained_val_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_first_DG_gamma95_2nd/results_of_trained_val_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_first_DG_gamma95_3rd/results_of_trained_val_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"
            ]
        ),

        "RTDP DG" => (
            train_gamma1 = [
                "stats/dqn_second_DG_gamma1/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_DG_gamma1_2nd/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_DG_gamma1_3rd/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"
            ],
            test_gamma1 = [
                "stats/dqn_second_DG_gamma1/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_DG_gamma1_2nd/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_DG_gamma1_3rd/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"
            ],
            val_gamma1 = [
                "stats/dqn_second_DG_gamma1/results_of_trained_val_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_DG_gamma1_2nd/results_of_trained_val_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_DG_gamma1_3rd/results_of_trained_val_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"
            ],

            train_gamma09 = [
                "stats/dqn_second_DG_gamma09/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_DG_gamma09_2nd/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_DG_gamma09_3rd/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"
            ],
            test_gamma09 = [
                "stats/dqn_second_DG_gamma09/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_DG_gamma09_2nd/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_DG_gamma09_3rd/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"
            ],
            val_gamma09 = [
                "stats/dqn_second_DG_gamma09/results_of_trained_val_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_DG_gamma09_2nd/results_of_trained_val_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_DG_gamma09_3rd/results_of_trained_val_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"
            ],

            train_gamma08 = [
                "stats/dqn_second_DG_gamma08/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_DG_gamma08_2nd/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_DG_gamma08_3rd/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"
            ],
            test_gamma08 = [
                "stats/dqn_second_DG_gamma08/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_DG_gamma08_2nd/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_DG_gamma08_3rd/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"
            ],
            val_gamma08 = [
                "stats/dqn_second_DG_gamma08/results_of_trained_val_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_DG_gamma08_2nd/results_of_trained_val_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_DG_gamma08_3rd/results_of_trained_val_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"
            ],

            train_gamma99 = [
                "stats/dqn_second_DG_gamma99/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_DG_gamma99_2nd/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_DG_gamma99_3rd/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"
            ],
            test_gamma99 = [
                "stats/dqn_second_DG_gamma99/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_DG_gamma99_2nd/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_DG_gamma99_3rd/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"
            ],
            val_gamma99 = [
                "stats/dqn_second_DG_gamma99/results_of_trained_val_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_DG_gamma99_2nd/results_of_trained_val_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_DG_gamma99_3rd/results_of_trained_val_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"
            ],

            train_gamma95 = [
                "stats/dqn_second_DG_gamma95/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_DG_gamma95_2nd/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_DG_gamma95_3rd/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"
            ],
            test_gamma95 = [
                "stats/dqn_second_DG_gamma95/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_DG_gamma95_2nd/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_DG_gamma95_3rd/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"
            ],
            val_gamma95 = [
                "stats/dqn_second_DG_gamma95/results_of_trained_val_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_DG_gamma95_2nd/results_of_trained_val_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_DG_gamma95_3rd/results_of_trained_val_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"
            ]
        ),
        "QTreeLearningSolutionSample"=> (
            train_gamma99 = [
                "stats/dqn_q_Tree_gamma99/results_of_trained_tree_MDP_q_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
            ],
            test_gamma99 = [
                "stats/dqn_q_Tree_gamma99/results_of_trained_test_tree_MDP_q_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
            ],
            val_gamma99 = [
                "stats/dqn_q_Tree_gamma99/results_of_trained_val_tree_MDP_q_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
            ],
        ),
        "QTreeLearningDFSSample"=> (
            train_gamma99 = [
                # "stats/dqn_q_Tree_gamma99/results_of_trained_tree_MDP_q_full_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
                "stats/dqn_q_Tree_gamma99/results_of_trained_is_target_tree_MDP_q_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
                # "stats/dqn_q_full_Tree_gamma99_3rd/results_of_trained_tree_MDP_q_full_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            test_gamma99 = [
                # "stats/dqn_q_Tree_gamma99/results_of_trained_test_tree_MDP_q_full_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
                "stats/dqn_q_Tree_gamma99/results_of_trained_test_is_target_tree_MDP_q_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
                # "stats/dqn_q_full_Tree_gamma99_3rd/results_of_trained_test_tree_MDP_q_full_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            val_gamma99 = [
                # "stats/dqn_q_Tree_gamma99/results_of_trained_val_tree_MDP_q_full_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
                "stats/dqn_q_Tree_gamma99/results_of_trained_val_is_target_tree_MDP_q_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
                # "stats/dqn_q_full_Tree_gamma99_3rd/results_of_trained_val_tree_MDP_q_full_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
        ),
        "PolicyTreeLearningOnce"=>(
            train_gamma99 = [
                # "stats/reinforce_policy_Tree_gamma99/results_of_trained_once_tree_PG_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
                # "stats/reinforce_policy_Tree_gamma99_2nd/results_of_trained_once_tree_PG_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
                # "stats/reinforce_policy_Tree_gamma99_3rd/results_of_trained_once_tree_PG_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
                "stats/reinforce_policy_Tree_gamma99/fixed_results_of_trained_once_tree_PG_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
                "stats/reinforce_policy_Tree_gamma99_2nd/fixed_results_of_trained_once_tree_PG_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
                "stats/reinforce_policy_Tree_gamma99_3rd/fixed_results_of_trained_once_tree_PG_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            test_gamma99 = [
                "stats/reinforce_policy_Tree_gamma99/results_of_trained_test_once_tree_PG_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
                "stats/reinforce_policy_Tree_gamma99_2nd/results_of_trained_test_once_tree_PG_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
                "stats/reinforce_policy_Tree_gamma99_3rd/results_of_trained_test_once_tree_PG_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            val_gamma99 = [
                "stats/reinforce_policy_Tree_gamma99/results_of_trained_val_once_tree_PG_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
                "stats/reinforce_policy_Tree_gamma99_2nd/results_of_trained_val_once_tree_PG_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
                "stats/reinforce_policy_Tree_gamma99_3rd/results_of_trained_val_once_tree_PG_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv"
            ],
        ),
        "PolicyTreeLearning"=>(
            train_gamma99 = [
                # C:\Users\thomas\Project\MetaProject1\stats\reinforce_policy_Tree_gamma99\results_of_trained_val_tree_PG_reinforce_policy_Tree_not_boosted_ep100_batch256_gamma99_hidden64.csv
                
                "stats/reinforce_policy_Tree_gamma99/results_of_trained_tree_PG_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
                "stats/reinforce_policy_Tree_gamma99_2nd/results_of_trained_tree_PG_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
                "stats/reinforce_policy_Tree_gamma99_3rd/results_of_trained_tree_PG_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            test_gamma99 = [
                "stats/reinforce_policy_Tree_gamma99/results_of_trained_test_tree_PG_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv"
                "stats/reinforce_policy_Tree_gamma99_2nd/results_of_trained_test_tree_PG_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv"
                "stats/reinforce_policy_Tree_gamma99_3rd/results_of_trained_test_tree_PG_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv"
            ],
            val_gamma99 = [
                "stats/reinforce_policy_Tree_gamma99/results_of_trained_val_tree_PG_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv"
                "stats/reinforce_policy_Tree_gamma99_2nd/results_of_trained_val_tree_PG_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv"
                "stats/reinforce_policy_Tree_gamma99_3rd/results_of_trained_val_tree_PG_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv"
            ],
        ),
        "PPOTreeLearning"=>(
            train_gamma99 = [
                # "stats/ppo_policy_Tree_gamma99_3rd/fixed_results_of_trained_tree_PPO_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv"
                "stats/ppo_policy_Tree_gamma99/results_of_trained_td_rew_tree_PPO_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            test_gamma99 = [
                "stats/ppo_policy_Tree_gamma99/results_of_trained_test_tree_PPO_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv"
            ],
            val_gamma99 = [
                "stats/ppo_policy_Tree_gamma99/results_of_trained_val_tree_PPO_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv"
            ],
        ),
        "PolicyLinearLearning" => (
            train_gamma99 = [
                # stats\reinforce_policy_Linear_gamma99\results_of_trained_linear_PG_reinforce_policy_Linear_not_boosted_ep3_batch256_gamma99_hidden64.csv
                "stats/reinforce_policy_Linear_gamma99/results_of_trained_linear_PG_reinforce_policy_Linear_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv"
            ],
            test_gamma99 = [
                "stats/reinforce_policy_Linear_gamma99/results_of_trained_test_linear_PG_reinforce_policy_Linear_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv"
            ],
            val_gamma99 = [
                "stats/reinforce_policy_Linear_gamma99/results_of_trained_val_linear_PG_reinforce_policy_Linear_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv"
            ],
        ),
        "PolicyTreeMDP" => (
            train_gamma99 = [
                "stats/reinforce_policy_Tree_100exp_gamma99/results_of_trained_tree_PG_no_solution_fixed_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            test_gamma99 = [
                "stats/reinforce_policy_Tree_100exp_gamma99/results_of_trained_test_tree_PG_no_solution_fixed_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            val_gamma99 = [
                "stats/reinforce_policy_Tree_100exp_gamma99/results_of_trained_val_tree_PG_no_solution_fixed_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
        ),
        "PolicyTreeMDPSolution" => (
            # results_of_trained_test_tree_PG_solution_reinforce_policy_Tree_not_boosted_ep75_batch256_gamma99_hidden64
            train_gamma99 = [
                "stats/reinforce_policy_Tree_100exp_gamma99/results_of_trained_tree_PG_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            test_gamma99 = [
                "stats/reinforce_policy_Tree_100exp_gamma99/results_of_trained_test_tree_PG_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            val_gamma99 = [
                "stats/reinforce_policy_Tree_100exp_gamma99/results_of_trained_val_tree_PG_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
        ),
        "PolicyTreeSolutionBoosted" => (
            train_gamma95 = [
                "stats/reinforce_policy_Tree_100exp_gamma95/results_of_trained_tree_PG_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            train_gamma99 = [
                "stats/reinforce_policy_Tree_100exp_gamma99/results_of_trained_tree_PG_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            train_gamma09 = [
                "stats/reinforce_policy_Tree_100exp_gamma09/results_of_trained_tree_PG_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            test_gamma95 = [
                "stats/reinforce_policy_Tree_100exp_gamma95/results_of_trained_test_tree_PG_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            test_gamma99 = [
                "stats/reinforce_policy_Tree_100exp_gamma99/results_of_trained_test_tree_PG_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            test_gamma09 = [
                "stats/reinforce_policy_Tree_100exp_gamma09/results_of_trained_test_tree_PG_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            val_gamma95 = [
                "stats/reinforce_policy_Tree_100exp_gamma95/results_of_trained_val_tree_PG_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            val_gamma99 = [
                "stats/reinforce_policy_Tree_100exp_gamma99/results_of_trained_val_tree_PG_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            val_gamma09 = [
                "stats/reinforce_policy_Tree_100exp_gamma09/results_of_trained_val_tree_PG_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            train_gamma10 = [
                "stats/reinforce_policy_Tree_100exp_gamma10/results_of_trained_tree_PG_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],
            test_gamma10 = [
                "stats/reinforce_policy_Tree_100exp_gamma10/results_of_trained_test_tree_PG_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],
            val_gamma10 = [
                "stats/reinforce_policy_Tree_100exp_gamma10/results_of_trained_val_tree_PG_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],

        ),
        "PolicyTreeNoSolutionBoosted" => (
            train_gamma95 = [
                "stats/reinforce_policy_Tree_100exp_gamma95/results_of_trained_tree_PG_no_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            train_gamma99 = [
                "stats/reinforce_policy_Tree_100exp_gamma99/results_of_trained_tree_PG_no_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            train_gamma09 = [
                "stats/reinforce_policy_Tree_100exp_gamma09/results_of_trained_tree_PG_no_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            test_gamma95 = [
                "stats/reinforce_policy_Tree_100exp_gamma95/results_of_trained_test_tree_PG_no_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            test_gamma99 = [
                "stats/reinforce_policy_Tree_100exp_gamma99/results_of_trained_test_tree_PG_no_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            test_gamma09 = [
                "stats/reinforce_policy_Tree_100exp_gamma09/results_of_trained_test_tree_PG_no_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            val_gamma95 = [
                "stats/reinforce_policy_Tree_100exp_gamma95/results_of_trained_val_tree_PG_no_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            val_gamma99 = [
                "stats/reinforce_policy_Tree_100exp_gamma99/results_of_trained_val_tree_PG_no_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            val_gamma09 = [
                "stats/reinforce_policy_Tree_100exp_gamma09/results_of_trained_val_tree_PG_no_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            train_gamma10 = [
                "stats/reinforce_policy_Tree_100exp_gamma10/results_of_trained_tree_PG_no_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],
            test_gamma10 = [
                "stats/reinforce_policy_Tree_100exp_gamma10/results_of_trained_test_tree_PG_no_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],
            val_gamma10 = [
                "stats/reinforce_policy_Tree_100exp_gamma10/results_of_trained_val_tree_PG_no_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],

        ),
        "PolicyTreeNoSolutionNotBoosted" => (
            train_gamma95 = [
                "stats/reinforce_policy_Tree_100exp_gamma95/results_of_trained_tree_PG_no_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            train_gamma99 = [
                "stats/reinforce_policy_Tree_100exp_gamma99/results_of_trained_tree_PG_no_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            train_gamma09 = [
                "stats/reinforce_policy_Tree_100exp_gamma09/results_of_trained_tree_PG_no_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            test_gamma95 = [
                "stats/reinforce_policy_Tree_100exp_gamma95/results_of_trained_test_tree_PG_no_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            test_gamma99 = [
                "stats/reinforce_policy_Tree_100exp_gamma99/results_of_trained_test_tree_PG_no_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            test_gamma09 = [
                "stats/reinforce_policy_Tree_100exp_gamma09/results_of_trained_test_tree_PG_no_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            val_gamma95 = [
                "stats/reinforce_policy_Tree_100exp_gamma95/results_of_trained_val_tree_PG_no_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            val_gamma99 = [
                "stats/reinforce_policy_Tree_100exp_gamma99/results_of_trained_val_tree_PG_no_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            val_gamma09 = [
                "stats/reinforce_policy_Tree_100exp_gamma09/results_of_trained_val_tree_PG_no_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            train_gamma10 = [
                "stats/reinforce_policy_Tree_100exp_gamma10/results_of_trained_tree_PG_no_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],
            test_gamma10 = [
                "stats/reinforce_policy_Tree_100exp_gamma10/results_of_trained_test_tree_PG_no_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],
            val_gamma10 = [
                "stats/reinforce_policy_Tree_100exp_gamma10/results_of_trained_val_tree_PG_no_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],

        ),
        "PolicyTreeSolutionNotBoosted" => (
            train_gamma95 = [
                "stats/reinforce_policy_Tree_100exp_gamma95/results_of_trained_tree_PG_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            train_gamma99 = [
                "stats/reinforce_policy_Tree_100exp_gamma99/results_of_trained_tree_PG_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            train_gamma09 = [
                "stats/reinforce_policy_Tree_100exp_gamma09/results_of_trained_tree_PG_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            test_gamma95 = [
                "stats/reinforce_policy_Tree_100exp_gamma95/results_of_trained_test_tree_PG_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            test_gamma99 = [
                "stats/reinforce_policy_Tree_100exp_gamma99/results_of_trained_test_tree_PG_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            test_gamma09 = [
                "stats/reinforce_policy_Tree_100exp_gamma09/results_of_trained_test_tree_PG_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            val_gamma95 = [
                "stats/reinforce_policy_Tree_100exp_gamma95/results_of_trained_val_tree_PG_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            val_gamma99 = [
                "stats/reinforce_policy_Tree_100exp_gamma99/results_of_trained_val_tree_PG_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            val_gamma09 = [
                "stats/reinforce_policy_Tree_100exp_gamma09/results_of_trained_val_tree_PG_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            train_gamma10 = [
                "stats/reinforce_policy_Tree_100exp_gamma10/results_of_trained_tree_PG_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],
            test_gamma10 = [
                "stats/reinforce_policy_Tree_100exp_gamma10/results_of_trained_test_tree_PG_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],
            val_gamma10 = [
                "stats/reinforce_policy_Tree_100exp_gamma10/results_of_trained_val_tree_PG_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],

        ),
        "PolicyTreeSolutionNotBoosted1k" => (
            train_gamma95 = [
                "stats/reinforce_policy_Tree_gamma95/results_of_trained_tree_PG_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            train_gamma99 = [
                "stats/reinforce_policy_Tree_gamma99/results_of_trained_tree_PG_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            train_gamma09 = [
                "stats/reinforce_policy_Tree_gamma09/results_of_trained_tree_PG_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            test_gamma95 = [
                "stats/reinforce_policy_Tree_gamma95/results_of_trained_test_tree_PG_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            test_gamma99 = [
                "stats/reinforce_policy_Tree_gamma99/results_of_trained_test_tree_PG_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            test_gamma09 = [
                "stats/reinforce_policy_Tree_gamma09/results_of_trained_test_tree_PG_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            val_gamma95 = [
                "stats/reinforce_policy_Tree_gamma95/results_of_trained_val_tree_PG_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            val_gamma99 = [
                "stats/reinforce_policy_Tree_gamma99/results_of_trained_val_tree_PG_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            val_gamma09 = [
                "stats/reinforce_policy_Tree_gamma09/results_of_trained_val_tree_PG_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            train_gamma10 = [
                "stats/reinforce_policy_Tree_gamma10/results_of_trained_tree_PG_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],
            test_gamma10 = [
                "stats/reinforce_policy_Tree_gamma10/results_of_trained_test_tree_PG_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],
            val_gamma10 = [
                "stats/reinforce_policy_Tree_gamma10/results_of_trained_val_tree_PG_solution_reinforce_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],

        ),
        "PolicyTreeSolutionBoosted1k" => (
            train_gamma95 = [
                "stats/reinforce_policy_Tree_gamma95/results_of_trained_tree_PG_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            train_gamma99 = [
                "stats/reinforce_policy_Tree_gamma99/results_of_trained_tree_PG_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            train_gamma09 = [
                "stats/reinforce_policy_Tree_gamma09/results_of_trained_tree_PG_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            test_gamma95 = [
                "stats/reinforce_policy_Tree_gamma95/results_of_trained_test_tree_PG_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            test_gamma99 = [
                "stats/reinforce_policy_Tree_gamma99/results_of_trained_test_tree_PG_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            test_gamma09 = [
                "stats/reinforce_policy_Tree_gamma09/results_of_trained_test_tree_PG_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            val_gamma95 = [
                "stats/reinforce_policy_Tree_gamma95/results_of_trained_val_tree_PG_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            val_gamma99 = [
                "stats/reinforce_policy_Tree_gamma99/results_of_trained_val_tree_PG_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            val_gamma09 = [
                "stats/reinforce_policy_Tree_gamma09/results_of_trained_val_tree_PG_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            train_gamma10 = [
                "stats/reinforce_policy_Tree_gamma10/results_of_trained_tree_PG_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],
            test_gamma10 = [
                "stats/reinforce_policy_Tree_gamma10/results_of_trained_test_tree_PG_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],
            val_gamma10 = [
                "stats/reinforce_policy_Tree_gamma10/results_of_trained_val_tree_PG_solution_reinforce_policy_Tree_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],

        ),
        "PPOTreeSolutionBoosted" => (
            train_gamma95 = [
                "stats/ppo_policy_Tree_100exp_gamma95/results_of_trained_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            train_gamma99 = [
                "stats/ppo_policy_Tree_100exp_gamma99/results_of_trained_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            train_gamma09 = [
                "stats/ppo_policy_Tree_100exp_gamma09/results_of_trained_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            test_gamma95 = [
                "stats/ppo_policy_Tree_100exp_gamma95/results_of_trained_test_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            test_gamma99 = [
                "stats/ppo_policy_Tree_100exp_gamma99/results_of_trained_test_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            test_gamma09 = [
                "stats/ppo_policy_Tree_100exp_gamma09/results_of_trained_test_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            val_gamma95 = [
                "stats/ppo_policy_Tree_100exp_gamma95/results_of_trained_val_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            val_gamma99 = [
                "stats/ppo_policy_Tree_100exp_gamma99/results_of_trained_val_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            val_gamma09 = [
                "stats/ppo_policy_Tree_100exp_gamma09/results_of_trained_val_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            train_gamma10 = [
                "stats/ppo_policy_Tree_100exp_gamma10/results_of_trained_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],
            test_gamma10 = [
                "stats/ppo_policy_Tree_100exp_gamma10/results_of_trained_test_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],
            val_gamma10 = [
                "stats/ppo_policy_Tree_100exp_gamma10/results_of_trained_val_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],

        ),
        "PPOTreeNoSolutionBoosted" => (
            train_gamma95 = [
                "stats/ppo_policy_Tree_100exp_gamma95/results_of_trained_tree_PG_no_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            train_gamma99 = [
                "stats/ppo_policy_Tree_100exp_gamma99/results_of_trained_tree_PG_no_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            train_gamma09 = [
                "stats/ppo_policy_Tree_100exp_gamma09/results_of_trained_tree_PG_no_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            test_gamma95 = [
                "stats/ppo_policy_Tree_100exp_gamma95/results_of_trained_test_tree_PG_no_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            test_gamma99 = [
                "stats/ppo_policy_Tree_100exp_gamma99/results_of_trained_test_tree_PG_no_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            test_gamma09 = [
                "stats/ppo_policy_Tree_100exp_gamma09/results_of_trained_test_tree_PG_no_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            val_gamma95 = [
                "stats/ppo_policy_Tree_100exp_gamma95/results_of_trained_val_tree_PG_no_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            val_gamma99 = [
                "stats/ppo_policy_Tree_100exp_gamma99/results_of_trained_val_tree_PG_no_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            val_gamma09 = [
                "stats/ppo_policy_Tree_100exp_gamma09/results_of_trained_val_tree_PG_no_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            train_gamma10 = [
                "stats/ppo_policy_Tree_100exp_gamma10/results_of_trained_tree_PG_no_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],
            test_gamma10 = [
                "stats/ppo_policy_Tree_100exp_gamma10/results_of_trained_test_tree_PG_no_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],
            val_gamma10 = [
                "stats/ppo_policy_Tree_100exp_gamma10/results_of_trained_val_tree_PG_no_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],

        ),
        "PPOTreeNoSolutionNotBoosted" => (
            train_gamma95 = [
                "stats/ppo_policy_Tree_100exp_gamma95/results_of_trained_tree_PG_no_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            train_gamma99 = [
                "stats/ppo_policy_Tree_100exp_gamma99/results_of_trained_tree_PG_no_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            train_gamma09 = [
                "stats/ppo_policy_Tree_100exp_gamma09/results_of_trained_tree_PG_no_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            test_gamma95 = [
                "stats/ppo_policy_Tree_100exp_gamma95/results_of_trained_test_tree_PG_no_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            test_gamma99 = [
                "stats/ppo_policy_Tree_100exp_gamma99/results_of_trained_test_tree_PG_no_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            test_gamma09 = [
                "stats/ppo_policy_Tree_100exp_gamma09/results_of_trained_test_tree_PG_no_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            val_gamma95 = [
                "stats/ppo_policy_Tree_100exp_gamma95/results_of_trained_val_tree_PG_no_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            val_gamma99 = [
                "stats/ppo_policy_Tree_100exp_gamma99/results_of_trained_val_tree_PG_no_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            val_gamma09 = [
                "stats/ppo_policy_Tree_100exp_gamma09/results_of_trained_val_tree_PG_no_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            train_gamma10 = [
                "stats/ppo_policy_Tree_100exp_gamma10/results_of_trained_tree_PG_no_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],
            test_gamma10 = [
                "stats/ppo_policy_Tree_100exp_gamma10/results_of_trained_test_tree_PG_no_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],
            val_gamma10 = [
                "stats/ppo_policy_Tree_100exp_gamma10/results_of_trained_val_tree_PG_no_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],

        ),
        "PPOTreeSolutionNotBoosted" => (
            train_gamma95 = [
                "stats/ppo_policy_Tree_100exp_gamma95/results_of_trained_tree_PG_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            train_gamma99 = [
                "stats/ppo_policy_Tree_100exp_gamma99/results_of_trained_tree_PG_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            train_gamma09 = [
                "stats/ppo_policy_Tree_100exp_gamma09/results_of_trained_tree_PG_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            test_gamma95 = [
                "stats/ppo_policy_Tree_100exp_gamma95/results_of_trained_test_tree_PG_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            test_gamma99 = [
                "stats/ppo_policy_Tree_100exp_gamma99/results_of_trained_test_tree_PG_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            test_gamma09 = [
                "stats/ppo_policy_Tree_100exp_gamma09/results_of_trained_test_tree_PG_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            val_gamma95 = [
                "stats/ppo_policy_Tree_100exp_gamma95/results_of_trained_val_tree_PG_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            val_gamma99 = [
                "stats/ppo_policy_Tree_100exp_gamma99/results_of_trained_val_tree_PG_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            val_gamma09 = [
                "stats/ppo_policy_Tree_100exp_gamma09/results_of_trained_val_tree_PG_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            train_gamma10 = [
                "stats/ppo_policy_Tree_100exp_gamma10/results_of_trained_tree_PG_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],
            test_gamma10 = [
                "stats/ppo_policy_Tree_100exp_gamma10/results_of_trained_test_tree_PG_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],
            val_gamma10 = [
                "stats/ppo_policy_Tree_100exp_gamma10/results_of_trained_val_tree_PG_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],

        ),

        "PPOmaxVlTreeSolutionBoosted" => (
            train_gamma95 = [
                "stats/ppo_policy_Tree_100exp_gamma95/results_of_trained_vl_max_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            train_gamma99 = [
                "stats/ppo_policy_Tree_100exp_gamma99/results_of_trained_vl_max_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            train_gamma09 = [
                "stats/ppo_policy_Tree_100exp_gamma09/results_of_trained_vl_max_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            test_gamma95 = [
                "stats/ppo_policy_Tree_100exp_gamma95/results_of_trained_test_vl_max_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            test_gamma99 = [
                "stats/ppo_policy_Tree_100exp_gamma99/results_of_trained_test_vl_max_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            test_gamma09 = [
                "stats/ppo_policy_Tree_100exp_gamma09/results_of_trained_test_vl_max_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            val_gamma95 = [
                "stats/ppo_policy_Tree_100exp_gamma95/results_of_trained_val_vl_max_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            val_gamma99 = [
                "stats/ppo_policy_Tree_100exp_gamma99/results_of_trained_val_vl_max_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            val_gamma09 = [
                "stats/ppo_policy_Tree_100exp_gamma09/results_of_trained_val_vl_max_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            train_gamma10 = [
                "stats/ppo_policy_Tree_100exp_gamma10/results_of_trained_vl_max_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],
            test_gamma10 = [
                "stats/ppo_policy_Tree_100exp_gamma10/results_of_trained_test_vl_max_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],
            val_gamma10 = [
                "stats/ppo_policy_Tree_100exp_gamma10/results_of_trained_val_vl_max_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],

        ),
        "PPOmaxVlTreeNoSolutionBoosted" => (
            train_gamma95 = [
                "stats/ppo_policy_Tree_100exp_gamma95/results_of_trained_vl_max_tree_PG_no_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            train_gamma99 = [
                "stats/ppo_policy_Tree_100exp_gamma99/results_of_trained_vl_max_tree_PG_no_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            train_gamma09 = [
                "stats/ppo_policy_Tree_100exp_gamma09/results_of_trained_vl_max_tree_PG_no_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            test_gamma95 = [
                "stats/ppo_policy_Tree_100exp_gamma95/results_of_trained_test_vl_max_tree_PG_no_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            test_gamma99 = [
                "stats/ppo_policy_Tree_100exp_gamma99/results_of_trained_test_vl_max_tree_PG_no_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            test_gamma09 = [
                "stats/ppo_policy_Tree_100exp_gamma09/results_of_trained_test_vl_max_tree_PG_no_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            val_gamma95 = [
                "stats/ppo_policy_Tree_100exp_gamma95/results_of_trained_val_vl_max_tree_PG_no_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            val_gamma99 = [
                "stats/ppo_policy_Tree_100exp_gamma99/results_of_trained_val_vl_max_tree_PG_no_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            val_gamma09 = [
                "stats/ppo_policy_Tree_100exp_gamma09/results_of_trained_val_vl_max_tree_PG_no_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            train_gamma10 = [
                "stats/ppo_policy_Tree_100exp_gamma10/results_of_trained_vl_max_tree_PG_no_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],
            test_gamma10 = [
                "stats/ppo_policy_Tree_100exp_gamma10/results_of_trained_test_vl_max_tree_PG_no_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],
            val_gamma10 = [
                "stats/ppo_policy_Tree_100exp_gamma10/results_of_trained_val_vl_max_tree_PG_no_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],

        ),
        "PPOmaxVlTreeNoSolutionNotBoosted" => (
            train_gamma95 = [
                "stats/ppo_policy_Tree_100exp_gamma95/results_of_trained_vl_max_tree_PG_no_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            train_gamma99 = [
                "stats/ppo_policy_Tree_100exp_gamma99/results_of_trained_vl_max_tree_PG_no_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            train_gamma09 = [
                "stats/ppo_policy_Tree_100exp_gamma09/results_of_trained_vl_max_tree_PG_no_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            test_gamma95 = [
                "stats/ppo_policy_Tree_100exp_gamma95/results_of_trained_test_vl_max_tree_PG_no_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            test_gamma99 = [
                "stats/ppo_policy_Tree_100exp_gamma99/results_of_trained_test_vl_max_tree_PG_no_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            test_gamma09 = [
                "stats/ppo_policy_Tree_100exp_gamma09/results_of_trained_test_vl_max_tree_PG_no_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            val_gamma95 = [
                "stats/ppo_policy_Tree_100exp_gamma95/results_of_trained_val_vl_max_tree_PG_no_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            val_gamma99 = [
                "stats/ppo_policy_Tree_100exp_gamma99/results_of_trained_val_vl_max_tree_PG_no_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            val_gamma09 = [
                "stats/ppo_policy_Tree_100exp_gamma09/results_of_trained_val_vl_max_tree_PG_no_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            train_gamma10 = [
                "stats/ppo_policy_Tree_100exp_gamma10/results_of_trained_vl_max_tree_PG_no_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],
            test_gamma10 = [
                "stats/ppo_policy_Tree_100exp_gamma10/results_of_trained_test_vl_max_tree_PG_no_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],
            val_gamma10 = [
                "stats/ppo_policy_Tree_100exp_gamma10/results_of_trained_val_vl_max_tree_PG_no_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],

        ),
        "PPOmaxVlTreeSolutionNotBoosted" => (
            train_gamma95 = [
                "stats/ppo_policy_Tree_100exp_gamma95/results_of_trained_vl_max_tree_PG_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            train_gamma99 = [
                "stats/ppo_policy_Tree_100exp_gamma99/results_of_trained_vl_max_tree_PG_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            train_gamma09 = [
                "stats/ppo_policy_Tree_100exp_gamma09/results_of_trained_vl_max_tree_PG_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            test_gamma95 = [
                "stats/ppo_policy_Tree_100exp_gamma95/results_of_trained_test_vl_max_tree_PG_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            test_gamma99 = [
                "stats/ppo_policy_Tree_100exp_gamma99/results_of_trained_test_vl_max_tree_PG_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            test_gamma09 = [
                "stats/ppo_policy_Tree_100exp_gamma09/results_of_trained_test_vl_max_tree_PG_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            val_gamma95 = [
                "stats/ppo_policy_Tree_100exp_gamma95/results_of_trained_val_vl_max_tree_PG_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            val_gamma99 = [
                "stats/ppo_policy_Tree_100exp_gamma99/results_of_trained_val_vl_max_tree_PG_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            val_gamma09 = [
                "stats/ppo_policy_Tree_100exp_gamma09/results_of_trained_val_vl_max_tree_PG_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            train_gamma10 = [
                "stats/ppo_policy_Tree_100exp_gamma10/results_of_trained_vl_max_tree_PG_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],
            test_gamma10 = [
                "stats/ppo_policy_Tree_100exp_gamma10/results_of_trained_test_vl_max_tree_PG_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],
            val_gamma10 = [
                "stats/ppo_policy_Tree_100exp_gamma10/results_of_trained_val_vl_max_tree_PG_solution_ppo_policy_Tree_not_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],

        ),
        "PPOLinearNoSolutionNotBoosted" => (
            train_gamma95 = [
                "stats/ppo_policy_Linear_100exp_gamma95/results_of_trained_linear_PG_no_solution_ppo_policy_Linear_not_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            train_gamma99 = [
                "stats/ppo_policy_Linear_100exp_gamma99/results_of_trained_linear_PG_no_solution_ppo_policy_Linear_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            train_gamma09 = [
                "stats/ppo_policy_Linear_100exp_gamma09/results_of_trained_linear_PG_no_solution_ppo_policy_Linear_not_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            test_gamma95 = [
                "stats/ppo_policy_Linear_100exp_gamma95/results_of_trained_test_linear_PG_no_solution_ppo_policy_Linear_not_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            test_gamma99 = [
                "stats/ppo_policy_Linear_100exp_gamma99/results_of_trained_test_linear_PG_no_solution_ppo_policy_Linear_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            test_gamma09 = [
                "stats/ppo_policy_Linear_100exp_gamma09/results_of_trained_test_linear_PG_no_solution_ppo_policy_Linear_not_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            val_gamma95 = [
                "stats/ppo_policy_Linear_100exp_gamma95/results_of_trained_val_linear_PG_no_solution_ppo_policy_Linear_not_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            val_gamma99 = [
                "stats/ppo_policy_Linear_100exp_gamma99/results_of_trained_val_linear_PG_no_solution_ppo_policy_Linear_not_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            val_gamma09 = [
                "stats/ppo_policy_Linear_100exp_gamma09/results_of_trained_val_linear_PG_no_solution_ppo_policy_Linear_not_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            train_gamma10 = [
                "stats/ppo_policy_Linear_100exp_gamma10/results_of_trained_linear_PG_no_solution_ppo_policy_Linear_not_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],
            test_gamma10 = [
                "stats/ppo_policy_Linear_100exp_gamma10/results_of_trained_test_linear_PG_no_solution_ppo_policy_Linear_not_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],
            val_gamma10 = [
                "stats/ppo_policy_Linear_100exp_gamma10/results_of_trained_val_linear_PG_no_solution_ppo_policy_Linear_not_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],

        ),

        "PPOTreeCorrectSolutionBoosted" => (
            train_gamma95 = [
                "stats/ppo_policy_Tree_100exp_gamma95/results_of_trained_correct_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            train_gamma99 = [
                "stats/ppo_policy_Tree_100exp_gamma99/results_of_trained_correct_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            train_gamma09 = [
                "stats/ppo_policy_Tree_100exp_gamma09/results_of_trained_correct_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            test_gamma95 = [
                "stats/ppo_policy_Tree_100exp_gamma95/results_of_trained_test_correct_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            test_gamma99 = [
                "stats/ppo_policy_Tree_100exp_gamma99/results_of_trained_test_correct_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            test_gamma09 = [
                "stats/ppo_policy_Tree_100exp_gamma09/results_of_trained_test_correct_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            val_gamma95 = [
                "stats/ppo_policy_Tree_100exp_gamma95/results_of_trained_val_correct_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma95_hidden64.csv",
            ],
            val_gamma99 = [
                "stats/ppo_policy_Tree_100exp_gamma99/results_of_trained_val_correct_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma99_hidden64.csv",
            ],
            val_gamma09 = [
                "stats/ppo_policy_Tree_100exp_gamma09/results_of_trained_val_correct_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma09_hidden64.csv",
            ],
            train_gamma10 = [
                "stats/ppo_policy_Tree_100exp_gamma10/results_of_trained_correct_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],
            test_gamma10 = [
                "stats/ppo_policy_Tree_100exp_gamma10/results_of_trained_test_correct_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],
            val_gamma10 = [
                "stats/ppo_policy_Tree_100exp_gamma10/results_of_trained_val_correct_tree_PG_solution_ppo_policy_Tree_boosted_ep\$(ep)_batch256_gamma10_hidden64.csv",
            ],

        ),
    )

    # plot(size=(800, 600), xlabel="Epochs", ylabel="Mean simplification", title="Model Comparison on Training Set")
    # res = Dict()
    # for (label, data) in models_results
    #     if label == "Planning"
    #         means_test, _ = process_model(data.train)
    #         plot!(1:length(means_test), means_test, label="$label Mean", lw=2)
    #     else
    #         means_test, _ = process_model(data.train_gamma1)
    #         # res[label] = means_test[end]
    #         plot!(1:length(means_test), means_test, label="$label γ=8", lw=2)
    #     end
    # end
    # plot!()
    # plots_array = Array{Any}(undef, 3, 2)
    legends_order = [
        "RTDP Tree" => (;style=:dash, color=:green),
        # "RTDP DAG" => (;style=:dash, color=:purple),
        "RTDP DG" => (;style=:dash, color=:orange),
        "Planning" => (;style=:solid, color=:blue),

        "VL Tree" => (;style=:dot, color=:green),
        # "VL DAG" => (;style=:dot, color=:purple),
        "VL DG" => (;style=:dot, color=:orange),
    ]
    legends_order = [
        "Planning100ep_3dense" => (;style=:solid, color=:blue),
        "PPOmaxVlTreeNoSolutionNotBoosted" => (;style=:solid, color=:green),
        "PPOmaxVlTreeSolutionNotBoosted" => (;style=:solid, color=:orange),
        "PPOmaxVlTreeNoSolutionBoosted" => (;style=:solid, color=:purple),
        "PPOmaxVlTreeSolutionBoosted" => (;style=:solid, color=:red),

    ]
    plots_array = []
    # gammas = [0.8, 0.9, 0.95, 0.99, 1]
    gammas = [0.9, 0.95, 0.99, 1]
    for (i, γ) in enumerate(gammas)
        # gamma = γ != 1 ? "0$(Int(γ * 10))" : Int(γ)
        gamma = length("$γ") > 2 ? last(replace("$γ", "." => ""), 2) : replace("$γ", "." => "")
        gamma = γ == 1 ? "10" : gamma
        # p_train = plot(title="Training γ=$γ", legend=false)
        p_train = plot(title="Training γ=$γ", legend=false, left_margin=10Plots.mm)

        # for (label, data) in models_results
        for (label,(style, color)) in legends_order
            data = models_results[label]
            if !(label in ["Planning", "Planning100ep_3dense"])
                means_train, _ = process_model(getfield(data, Symbol("train_gamma$gamma")), get_res=get_res2, final_epoch=100)
            else
                means_train, _ = process_model(getfield(data, Symbol("train")), get_res=get_res2, final_epoch=100)
                label = "L*"
            end
            # style, color = legends_order[label]
            plot!(p_train, 1:length(means_train), means_train, label=label, lw=2, color=color, linestyle=style, xlabel = "Epochs",
        ylabel = "Mean Simplification",)
        end
        # plots_array[i, 1] = p_train

        # For testing column (col=2)
        p_test = plot(title="Testing γ=$γ", legend=false)
        # for (label, data) in models_results
        for (label,(style, color)) in legends_order
            data = models_results[label]
            if !(label in ["Planning", "Planning100ep_3dense"])
                means_test, _ = process_model(getfield(data, Symbol("test_gamma$gamma")), get_res=get_res2, final_epoch=100)
            else
                means_test, _ = process_model(getfield(data, Symbol("test")), get_res=get_res2, final_epoch=100)
                label = "L*"
            end
            # style, color = legends_order[label]
            plot!(p_test, 1:length(means_test), means_test, label=label, lw=2,color=color, linestyle=style, xlabel = "Epochs",
        ylabel = "Mean Simplification",)
        end
        push!(plots_array, p_train, p_test)
    end
    legend_plot = plot(legend=:top, grid=false, framestyle=:none, legendcolumn=3,  size = (2000, 150))
    # for (label, _) in models_results
    lk = ["", "PPO TreeMDP NS NB", "PPO TreeMDP S NB", "PPO TreeMDP NS B", "PPO TreeMDP S B"]
    for (ind,(label,(style, color))) in enumerate(legends_order)
        # style, color = legends_order[label]
        if label in ["Planning", "Planning100ep_3dense"]
            label = "L*"
        else
            label = lk[ind]
        end
        plot!(legend_plot, [missing], label=label, lw=2,color=color, linestyle=style)
    end
    # Combine the grid of plots with the legend below
    main_plot = plot(plots_array..., layout=(4,2), legend=false)
    # main_plot = plot(plots_array[5:6]..., layout=(1,2), legend=false)
    final_plot = plot(main_plot, legend_plot, layout = @layout([a; b{0.05h}]), size=(2000, 2000))
    # final_plot = plot(
    #     legend_plot,
    #     main_plot,
    #     layout = @layout([a{0.1h}; b{0.8h}]),  # top row is legend, 10% height
    #     size = (2000, 1000)
    # )
    # Display
    display(final_plot)
    # Show the plot
    # display(final_plot)
end



function plot_fixed_conv_results_gamma()
    models_conv_results = Dict(
        "Planning" => (
            train=["stats/planning_4th_20ep/convergence_stats_heuristic_ep\$(ep).csv",
                "stats/planning_3rd_20ep/convergence_stats_heuristic_ep\$(ep).csv",
                "stats/planning_2nd_20ep/convergence_stats_heuristic_ep\$(ep).csv"],
            test=["stats/planning_4th_20ep/convergence_stats_test_heuristic_ep\$(ep).csv",
                "stats/planning_3rd_20ep/convergence_stats_test_heuristic_ep\$(ep).csv",
                "stats/planning_2nd_20ep/convergence_stats_test_heuristic_ep\$(ep).csv"]
        ),
        # VL 1 and 2 on Tree
        "VL Tree" => (
            train_gamma1=["stats/dqn_first_Tree_gamma1/convergence_stats_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_Tree_gamma1_2nd/convergence_stats_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_Tree_gamma1_3rd/convergence_stats_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            test_gamma1=["stats/dqn_first_Tree_gamma1/convergence_stats_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_Tree_gamma1_2nd/convergence_stats_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_Tree_gamma1_3rd/convergence_stats_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            train_gamma09=["stats/dqn_first_Tree_gamma09/convergence_stats_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv", 
            "stats/dqn_first_Tree_gamma09_2nd/convergence_stats_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_first_Tree_gamma09_3rd/convergence_stats_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"],
            test_gamma09=["stats/dqn_first_Tree_gamma09/convergence_stats_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_first_Tree_gamma09_2nd/convergence_stats_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_first_Tree_gamma09_3rd/convergence_stats_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"],
            train_gamma08=["stats/dqn_first_Tree_gamma08/convergence_stats_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_Tree_gamma08_2nd/convergence_stats_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_Tree_gamma08_3rd/convergence_stats_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"],
            test_gamma08=["stats/dqn_first_Tree_gamma08/convergence_stats_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_Tree_gamma08_2nd/convergence_stats_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_Tree_gamma08_3rd/convergence_stats_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"],
        ),
        "RTDP Tree" => (
            train_gamma1=["stats/dqn_second_Tree_gamma1/convergence_stats_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_second_Tree_gamma1_2nd/convergence_stats_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_second_Tree_gamma1_3rd/convergence_stats_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            test_gamma1=["stats/dqn_second_Tree_gamma1/convergence_stats_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_second_Tree_gamma1_2nd/convergence_stats_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_second_Tree_gamma1_3rd/convergence_stats_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            train_gamma09=["stats/dqn_second_Tree_gamma09/convergence_stats_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_second_Tree_gamma09_2nd/convergence_stats_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_second_Tree_gamma09_3rd/convergence_stats_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"],
            test_gamma09=["stats/dqn_second_Tree_gamma09/convergence_stats_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_second_Tree_gamma09_2nd/convergence_stats_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_second_Tree_gamma09_3rd/convergence_stats_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"],
            train_gamma08=["stats/dqn_second_Tree_gamma08/convergence_stats_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_second_Tree_gamma08_2nd/convergence_stats_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_second_Tree_gamma08_3rd/convergence_stats_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"],
            test_gamma08=["stats/dqn_second_Tree_gamma08/convergence_stats_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_second_Tree_gamma08_2nd/convergence_stats_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_second_Tree_gamma08_3rd/convergence_stats_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"],
        ),
        
        # VL 1 and 2 on DAG
        "VL DAG" => (
            train_gamma1=["stats/dqn_first_DAG_gamma1/convergence_stats_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_DAG_gamma1_2nd/convergence_stats_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_DAG_gamma1_3rd/convergence_stats_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            test_gamma1=["stats/dqn_first_DAG_gamma1/convergence_stats_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_DAG_gamma1_2nd/convergence_stats_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_DAG_gamma1_3rd/convergence_stats_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            train_gamma09=["stats/dqn_first_DAG_gamma09/convergence_stats_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_first_DAG_gamma09_2nd/convergence_stats_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_first_DAG_gamma09_3rd/convergence_stats_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"],
            test_gamma09=["stats/dqn_first_DAG_gamma09/convergence_stats_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_first_DAG_gamma09_2nd/convergence_stats_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_first_DAG_gamma09_3rd/convergence_stats_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"],
            train_gamma08=["stats/dqn_first_DAG_gamma08/convergence_stats_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_DAG_gamma08_2nd/convergence_stats_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_DAG_gamma08_3rd/convergence_stats_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"],
            test_gamma08=["stats/dqn_first_DAG_gamma08/convergence_stats_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_DAG_gamma08_2nd/convergence_stats_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_DAG_gamma08_3rd/convergence_stats_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"],
        ),
        "RTDP DAG" => (
            train_gamma1=["stats/dqn_second_DAG_gamma1/convergence_stats_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_second_DAG_gamma1_2nd/convergence_stats_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_second_DAG_gamma1_3rd/convergence_stats_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            test_gamma1=["stats/dqn_second_DAG_gamma1/convergence_stats_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_second_DAG_gamma1_2nd/convergence_stats_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_second_DAG_gamma1_3rd/convergence_stats_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            train_gamma09=["stats/dqn_second_DAG_gamma09/convergence_stats_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_second_DAG_gamma09_2nd/convergence_stats_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_second_DAG_gamma09_3rd/convergence_stats_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"],
            test_gamma09=["stats/dqn_second_DAG_gamma09/convergence_stats_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_second_DAG_gamma09_2nd/convergence_stats_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_second_DAG_gamma09_3rd/convergence_stats_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"],
            train_gamma08=["stats/dqn_second_DAG_gamma08/convergence_stats_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_second_DAG_gamma08_2nd/convergence_stats_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_second_DAG_gamma08_3rd/convergence_stats_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"],
            test_gamma08=["stats/dqn_second_DAG_gamma08/convergence_stats_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_second_DAG_gamma08_2nd/convergence_stats_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_second_DAG_gamma08_3rd/convergence_stats_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"],
        ),
        
        # VL 1 and 2 on DG
        "VL DG" => (
            train_gamma1=["stats/dqn_first_DG_gamma1/convergence_stats_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_DG_gamma1_2nd/convergence_stats_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_DG_gamma1_3rd/convergence_stats_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            test_gamma1=["stats/dqn_first_DG_gamma1/convergence_stats_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_DG_gamma1_2nd/convergence_stats_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_DG_gamma1_3rd/convergence_stats_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            train_gamma09=["stats/dqn_first_DG_gamma09/convergence_stats_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_first_DG_gamma09_2nd/convergence_stats_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_first_DG_gamma09_3rd/convergence_stats_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"],
            test_gamma09=["stats/dqn_first_DG_gamma09/convergence_stats_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_first_DG_gamma09_2nd/convergence_stats_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_first_DG_gamma09_3rd/convergence_stats_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"],
            train_gamma08=["stats/dqn_first_DG_gamma08/convergence_stats_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_DG_gamma08_2nd/convergence_stats_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_DG_gamma08_3rd/convergence_stats_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"],
            test_gamma08=["stats/dqn_first_DG_gamma08/convergence_stats_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_DG_gamma08_2nd/convergence_stats_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_DG_gamma08_3rd/convergence_stats_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"],
        ),
        "RTDP DG" => (
            train_gamma1=["stats/dqn_second_DG_gamma1/convergence_stats_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_second_DG_gamma1_2nd/convergence_stats_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_second_DG_gamma1_3rd/convergence_stats_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            test_gamma1=["stats/dqn_second_DG_gamma1/convergence_stats_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_second_DG_gamma1_2nd/convergence_stats_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_second_DG_gamma1_3rd/convergence_stats_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            train_gamma09=["stats/dqn_second_DG_gamma09/convergence_stats_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_second_DG_gamma09_2nd/convergence_stats_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_second_DG_gamma09_3rd/convergence_stats_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"],
            test_gamma09=["stats/dqn_second_DG_gamma09/convergence_stats_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_second_DG_gamma09_2nd/convergence_stats_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_second_DG_gamma09_3rd/convergence_stats_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"],
            train_gamma08=["stats/dqn_second_DG_gamma08/convergence_stats_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_second_DG_gamma08_2nd/convergence_stats_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_second_DG_gamma08_3rd/convergence_stats_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"],
            test_gamma08=["stats/dqn_second_DG_gamma08/convergence_stats_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_second_DG_gamma08_2nd/convergence_stats_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_second_DG_gamma08_3rd/convergence_stats_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"],
        ),
        
    )

    plot(size=(800, 600), xlabel="Epochs", ylabel="Mean stds of children values", title="Model Comparison on Training Set")
    res = Dict()
    for (label, data) in models_conv_results
        if label == "Planning" || label == "VL DAG" || label == "RTDP DAG"
            # means_test, _ = process_model(data.test, get_res=get_conv_res)
            # plot!(1:length(means_test), means_test, label="$label Mean", lw=2)
        else
            means_test, _ = process_model(data.test_gamma08, get_res=get_conv_res)
            # res[label] = means_test[end]
            plot!(1:length(means_test), means_test, label="$label γ=0.8", lw=2)
        end
    end
    plot!()
end


function plot_fixed_filtered_conv_results_gamma()
    models_filtered_conv_results = Dict(
        "Planning" => (
            train=["stats/planning_4th_20ep/convergence_stats_filtered_heuristic_ep\$(ep).csv",
                "stats/planning_3rd_20ep/convergence_stats_filtered_heuristic_ep\$(ep).csv",
                "stats/planning_2nd_20ep/convergence_stats_filtered_heuristic_ep\$(ep).csv"],
            test=["stats/planning_4th_20ep/convergence_stats_filtered_test_heuristic_ep\$(ep).csv",
                "stats/planning_3rd_20ep/convergence_stats_filtered_test_heuristic_ep\$(ep).csv",
                "stats/planning_2nd_20ep/convergence_stats_filtered_test_heuristic_ep\$(ep).csv"]
        ),
        # VL 1 and 2 on Tree
        "VL Tree" => (
            train_gamma1=["stats/dqn_first_Tree_gamma1/convergence_stats_filtered_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_Tree_gamma1_2nd/convergence_stats_filtered_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_Tree_gamma1_3rd/convergence_stats_filtered_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            test_gamma1=["stats/dqn_first_Tree_gamma1/convergence_stats_filtered_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_Tree_gamma1_2nd/convergence_stats_filtered_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_Tree_gamma1_3rd/convergence_stats_filtered_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            train_gamma09=["stats/dqn_first_Tree_gamma09/convergence_stats_filtered_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv", 
            "stats/dqn_first_Tree_gamma09_2nd/convergence_stats_filtered_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_first_Tree_gamma09_3rd/convergence_stats_filtered_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"],
            test_gamma09=["stats/dqn_first_Tree_gamma09/convergence_stats_filtered_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_first_Tree_gamma09_2nd/convergence_stats_filtered_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_first_Tree_gamma09_3rd/convergence_stats_filtered_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"],
            train_gamma08=["stats/dqn_first_Tree_gamma08/convergence_stats_filtered_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_Tree_gamma08_2nd/convergence_stats_filtered_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_Tree_gamma08_3rd/convergence_stats_filtered_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"],
            test_gamma08=["stats/dqn_first_Tree_gamma08/convergence_stats_filtered_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_Tree_gamma08_2nd/convergence_stats_filtered_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_Tree_gamma08_3rd/convergence_stats_filtered_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"],
        ),
        "RTDP Tree" => (
            train_gamma1=["stats/dqn_second_Tree_gamma1/convergence_stats_filtered_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_second_Tree_gamma1_2nd/convergence_stats_filtered_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_second_Tree_gamma1_3rd/convergence_stats_filtered_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            test_gamma1=["stats/dqn_second_Tree_gamma1/convergence_stats_filtered_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_second_Tree_gamma1_2nd/convergence_stats_filtered_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_second_Tree_gamma1_3rd/convergence_stats_filtered_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            train_gamma09=["stats/dqn_second_Tree_gamma09/convergence_stats_filtered_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_second_Tree_gamma09_2nd/convergence_stats_filtered_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_second_Tree_gamma09_3rd/convergence_stats_filtered_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"],
            test_gamma09=["stats/dqn_second_Tree_gamma09/convergence_stats_filtered_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_second_Tree_gamma09_2nd/convergence_stats_filtered_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_second_Tree_gamma09_3rd/convergence_stats_filtered_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"],
            train_gamma08=["stats/dqn_second_Tree_gamma08/convergence_stats_filtered_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_second_Tree_gamma08_2nd/convergence_stats_filtered_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_second_Tree_gamma08_3rd/convergence_stats_filtered_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"],
            test_gamma08=["stats/dqn_second_Tree_gamma08/convergence_stats_filtered_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_second_Tree_gamma08_2nd/convergence_stats_filtered_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_second_Tree_gamma08_3rd/convergence_stats_filtered_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"],
        ),
        
        # VL 1 and 2 on DAG
        "VL DAG" => (
            train_gamma1=["stats/dqn_first_DAG_gamma1/convergence_stats_filtered_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_DAG_gamma1_2nd/convergence_stats_filtered_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_DAG_gamma1_3rd/convergence_stats_filtered_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            test_gamma1=["stats/dqn_first_DAG_gamma1/convergence_stats_filtered_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_DAG_gamma1_2nd/convergence_stats_filtered_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_DAG_gamma1_3rd/convergence_stats_filtered_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            train_gamma09=["stats/dqn_first_DAG_gamma09/convergence_stats_filtered_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_first_DAG_gamma09_2nd/convergence_stats_filtered_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_first_DAG_gamma09_3rd/convergence_stats_filtered_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"],
            test_gamma09=["stats/dqn_first_DAG_gamma09/convergence_stats_filtered_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_first_DAG_gamma09_2nd/convergence_stats_filtered_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_first_DAG_gamma09_3rd/convergence_stats_filtered_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"],
            train_gamma08=["stats/dqn_first_DAG_gamma08/convergence_stats_filtered_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_DAG_gamma08_2nd/convergence_stats_filtered_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_DAG_gamma08_3rd/convergence_stats_filtered_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"],
            test_gamma08=["stats/dqn_first_DAG_gamma08/convergence_stats_filtered_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_DAG_gamma08_2nd/convergence_stats_filtered_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_DAG_gamma08_3rd/convergence_stats_filtered_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"],
        ),
        "RTDP DAG" => (
            train_gamma1=["stats/dqn_second_DAG_gamma1/convergence_stats_filtered_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_second_DAG_gamma1_2nd/convergence_stats_filtered_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_second_DAG_gamma1_3rd/convergence_stats_filtered_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            test_gamma1=["stats/dqn_second_DAG_gamma1/convergence_stats_filtered_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_second_DAG_gamma1_2nd/convergence_stats_filtered_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_second_DAG_gamma1_3rd/convergence_stats_filtered_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            train_gamma09=["stats/dqn_second_DAG_gamma09/convergence_stats_filtered_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_second_DAG_gamma09_2nd/convergence_stats_filtered_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_second_DAG_gamma09_3rd/convergence_stats_filtered_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"],
            test_gamma09=["stats/dqn_second_DAG_gamma09/convergence_stats_filtered_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_second_DAG_gamma09_2nd/convergence_stats_filtered_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_second_DAG_gamma09_3rd/convergence_stats_filtered_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"],
            train_gamma08=["stats/dqn_second_DAG_gamma08/convergence_stats_filtered_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_second_DAG_gamma08_2nd/convergence_stats_filtered_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_second_DAG_gamma08_3rd/convergence_stats_filtered_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"],
            test_gamma08=["stats/dqn_second_DAG_gamma08/convergence_stats_filtered_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_second_DAG_gamma08_2nd/convergence_stats_filtered_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_second_DAG_gamma08_3rd/convergence_stats_filtered_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"],
        ),
        
        # VL 1 and 2 on DG
        "VL DG" => (
            train_gamma1=["stats/dqn_first_DG_gamma1/convergence_stats_filtered_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_DG_gamma1_2nd/convergence_stats_filtered_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_DG_gamma1_3rd/convergence_stats_filtered_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            test_gamma1=["stats/dqn_first_DG_gamma1/convergence_stats_filtered_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_DG_gamma1_2nd/convergence_stats_filtered_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_DG_gamma1_3rd/convergence_stats_filtered_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            train_gamma09=["stats/dqn_first_DG_gamma09/convergence_stats_filtered_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_first_DG_gamma09_2nd/convergence_stats_filtered_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_first_DG_gamma09_3rd/convergence_stats_filtered_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"],
            test_gamma09=["stats/dqn_first_DG_gamma09/convergence_stats_filtered_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_first_DG_gamma09_2nd/convergence_stats_filtered_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_first_DG_gamma09_3rd/convergence_stats_filtered_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"],
            train_gamma08=["stats/dqn_first_DG_gamma08/convergence_stats_filtered_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_DG_gamma08_2nd/convergence_stats_filtered_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_DG_gamma08_3rd/convergence_stats_filtered_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"],
            test_gamma08=["stats/dqn_first_DG_gamma08/convergence_stats_filtered_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_DG_gamma08_2nd/convergence_stats_filtered_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_DG_gamma08_3rd/convergence_stats_filtered_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"],
        ),
        "RTDP DG" => (
            train_gamma1=["stats/dqn_second_DG_gamma1/convergence_stats_filtered_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_second_DG_gamma1_2nd/convergence_stats_filtered_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_second_DG_gamma1_3rd/convergence_stats_filtered_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            test_gamma1=["stats/dqn_second_DG_gamma1/convergence_stats_filtered_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_second_DG_gamma1_2nd/convergence_stats_filtered_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_second_DG_gamma1_3rd/convergence_stats_filtered_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            train_gamma09=["stats/dqn_second_DG_gamma09/convergence_stats_filtered_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_second_DG_gamma09_2nd/convergence_stats_filtered_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_second_DG_gamma09_3rd/convergence_stats_filtered_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"],
            test_gamma09=["stats/dqn_second_DG_gamma09/convergence_stats_filtered_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_second_DG_gamma09_2nd/convergence_stats_filtered_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_second_DG_gamma09_3rd/convergence_stats_filtered_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"],
            train_gamma08=["stats/dqn_second_DG_gamma08/convergence_stats_filtered_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_second_DG_gamma08_2nd/convergence_stats_filtered_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_second_DG_gamma08_3rd/convergence_stats_filtered_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"],
            test_gamma08=["stats/dqn_second_DG_gamma08/convergence_stats_filtered_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_second_DG_gamma08_2nd/convergence_stats_filtered_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_second_DG_gamma08_3rd/convergence_stats_filtered_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"],
        ),
        
    )

    plot(size=(800, 600), xlabel="Epochs", ylabel="Mean stds of children values", title="Model Comparison on Training Set")
    res = Dict()
    for (label, data) in models_conv_results
        if label == "Planning" || label == "VL DAG" || label == "RTDP DAG"
            # means_test, _ = process_model(data.test, get_res=get_conv_res)
            # plot!(1:length(means_test), means_test, label="$label Mean", lw=2)
        else
            means_test, _ = process_model(data.test_gamma08, get_res=get_conv_res)
            # res[label] = means_test[end]
            plot!(1:length(means_test), means_test, label="$label γ=0.8", lw=2)
        end
    end
    plot!()
end



function plot_linear_results()
    dqn_first_paths = ["stats/dqn_first_4th_20ep/linear_results_of_trained_DQN_first_DAG1_not_boosted_ep\$(ep)_hidden64.csv",
    "stats/dqn_first_3rd_20ep/linear_results_of_trained_DQN_first_DAG1_not_boosted_ep\$(ep)_hidden64.csv",
    "stats/dqn_first_2nd_20ep/linear_results_of_trained_DQN_first_DAG1_not_boosted_ep\$(ep)_hidden64.csv"]
    dqn_first_res = hcat(get_res(dqn_first_paths)...)
    dqn_first_res[dqn_first_res .< 0] .= 0.0
    means_dqn_first_res = vec(mean(dqn_first_res, dims=2))
    stds_dqn_first_res = vec(std(dqn_first_res, dims=2))
    plot!(1:length(stds_dqn_first_res), means_dqn_first_res, ribbon=stds_dqn_first_res, label="DQN 1 Tree Mean ± Std", lw=2, xlabel="Epochs", ylabel="Mean simplification")

    dqn_second_paths = ["stats/dqn_first_4th_20ep/linear_results_of_trained_DQN_second_DAG1_not_boosted_ep\$(ep)_hidden64.csv",
    "stats/dqn_first_3rd_20ep/linear_results_of_trained_DQN_second_DAG1_not_boosted_ep\$(ep)_hidden64.csv",
    "stats/dqn_first_2nd_20ep/linear_results_of_trained_DQN_second_DAG1_not_boosted_ep\$(ep)_hidden64.csv"]
    dqn_second_res = hcat(get_res(dqn_second_paths)...)
    dqn_second_res[dqn_second_res .< 0] .= 0.0
    means_dqn_second_res = vec(mean(dqn_second_res, dims=2))
    stds_dqn_second_res = vec(std(dqn_second_res, dims=2))
    plot!(1:length(stds_dqn_second_res), means_dqn_second_res, ribbon=stds_dqn_second_res, label="DQN 2 Tree Mean ± Std", lw=2, xlabel="Epochs", ylabel="Mean simplification")

    # results_of_trained_DQN_first_Tree_not_boosted_ep1_batch256_hidden64
    test_path = ["stats/dqn_first_2nd_20ep/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch256_hidden64.csv"]
    test_res = get_res(test_path)
    plot!(vec(test_res), label="DQN 1 Tree Batch 256 ± Std", lw=2, xlabel="Epochs", ylabel="Mean simplification")
    dqn_first_paths = ["stats/dqn_first_4th_20ep/results_of_trained_DQN_first_DAG1_ep\$(ep)_hidden64.csv",
    "stats/dqn_first_3rd_20ep/results_of_trained_DQN_first_DAG1_ep\$(ep)_hidden64.csv",
    "stats/dqn_first_2nd_20ep/results_of_trained_DQN_first_DAG1_not_boosted_ep\$(ep)_hidden64.csv"]
    dqn_first_res = hcat(get_res(dqn_first_paths)...)
    means_dqn_first_res = vec(mean(dqn_first_res, dims=2))
    plot!(means_dqn_first_res, label="DQN 1 Tree Mean Batch 128 ± Std", lw=2, xlabel="Epochs", ylabel="Mean simplification")

    planning_paths = ["stats/planning_4th_20ep/linear_results_of_trained_heuristic_ep\$(ep)_hidden64.csv",
    "stats/planning_3rd_20ep/linear_results_of_trained_heuristic_ep\$(ep)_hidden64.csv",
    "stats/planning_2nd_20ep/linear_results_of_trained_heuristic_ep\$(ep)_hidden64.csv"]
    planning_res = hcat(get_res(planning_paths)...)
    means_planning_res = vec(mean(planning_res, dims=2))
    stds_planning_res = vec(std(planning_res, dims=2))
    plot(1:length(stds_planning_res), means_planning_res, ribbon=stds_planning_res, label="Planning Mean ± Std", lw=2, xlabel="Epochs", ylabel="Mean simplification", legend=:outerbottom, legendcolumn=2, title="Linear Test on Training Dataset")
    
    # 100 Epochs
    planning_paths = ["stats/dqn_first_2nd_20ep/results_of_trained_DQN_first_DG_100ep_not_boosted_ep\$(ep)_batch128_hidden64.csv"]
    planning_res = hcat(get_res(planning_paths, epochs=100)...)
    means_planning_res = vec(mean(planning_res, dims=2))
    # stds_planning_res = vec(std(planning_res, dims=2))
    plot(1:length(stds_planning_res), means_planning_res, ribbon=stds_planning_res, label="DQN first DG", lw=2, xlabel="Epochs", ylabel="Mean simplification", legend=:outerbottom, legendcolumn=2)

    planning_paths = ["stats/dqn_first_2nd_20ep/results_of_trained_test_DQN_first_DG_100ep_not_boosted_ep\$(ep)_batch128_hidden64.csv"]
    planning_res = hcat(get_res(planning_paths, epochs=100)...)
    means_planning_res = vec(mean(planning_res, dims=2))
    # stds_planning_res = vec(std(planning_res, dims=2))
    plot!(1:length(stds_planning_res), means_planning_res, ribbon=stds_planning_res, label="DQN first DG Test", lw=2, xlabel="Epochs", ylabel="Mean simplification", legend=:outerbottom, legendcolumn=2)

    planning_paths = ["stats/dqn_first_2nd_20ep/results_of_trained_DQN_first_Tree_100ep_not_boosted_ep\$(ep)_batch128_hidden64.csv"]
    planning_res = hcat(get_res(planning_paths, epochs=100)...)
    means_planning_res = vec(mean(planning_res, dims=2))
    # stds_planning_res = vec(std(planning_res, dims=2))
    plot!(1:length(stds_planning_res), means_planning_res, ribbon=stds_planning_res, label="DQN first Tree", lw=2, xlabel="Epochs", ylabel="Mean simplification", legend=:outerbottom, legendcolumn=2)

    planning_paths = ["stats/dqn_first_2nd_20ep/results_of_trained_test_DQN_first_Tree_100ep_not_boosted_ep\$(ep)_batch128_hidden64.csv"]
    planning_res = hcat(get_res(planning_paths, epochs=100)...)
    means_planning_res = vec(mean(planning_res, dims=2))
    # stds_planning_res = vec(std(planning_res, dims=2))
    plot!(1:length(stds_planning_res), means_planning_res, ribbon=stds_planning_res, label="DQN first Tree Test", lw=2, xlabel="Epochs", ylabel="Mean simplification", legend=:outerbottom, legendcolumn=2)

    planning_paths = ["stats/dqn_first_2nd_20ep/results_of_trained_DQN_second_DG_boosted_ep\$(ep)_batch128_hidden64.csv"]
    planning_res = hcat(get_res(planning_paths, epochs=20)...)
    means_planning_res = vec(mean(planning_res, dims=2))
    plot(1:length(means_planning_res), means_planning_res, label="DQN second DG", lw=2, xlabel="Epochs", ylabel="Mean simplification", legend=:outerbottom, legendcolumn=2, title="Boosted DQN")

    planning_paths = ["stats/dqn_first_2nd_20ep/results_of_trained_test_DQN_second_DG_boosted_ep\$(ep)_batch128_hidden64.csv"]
    planning_res = hcat(get_res(planning_paths, epochs=20)...)
    means_planning_res = vec(mean(planning_res, dims=2))
    plot!(1:length(means_planning_res), means_planning_res, label="DQN second DG Test", lw=2, xlabel="Epochs", ylabel="Mean simplification", legend=:outerbottom, legendcolumn=2)

    planning_paths = ["stats/dqn_first_2nd_20ep/results_of_trained_DQN_first_Tree_boosted_ep\$(ep)_batch128_hidden64.csv"]
    planning_res = hcat(get_res(planning_paths, epochs=20)...)
    means_planning_res = vec(mean(planning_res, dims=2))
    plot!(1:length(means_planning_res), means_planning_res, label="DQN first tree", lw=2, xlabel="Epochs", ylabel="Mean simplification", legend=:outerbottom, legendcolumn=2)

    planning_paths = ["stats/dqn_first_2nd_20ep/results_of_trained_test_DQN_first_Tree_boosted_ep\$(ep)_batch128_hidden64.csv"]
    planning_res = hcat(get_res(planning_paths, epochs=20)...)
    means_planning_res = vec(mean(planning_res, dims=2))
    plot!(1:length(means_planning_res), means_planning_res, label="DQN first Tree Test", lw=2, xlabel="Epochs", ylabel="Mean simplification", legend=:outerbottom, legendcolumn=2)

end


function plot_latex_taable(models_results; res_fun::Function=get_res)
    for k in ["VL Tree", "VL DAG", "VL DG", "RTDP Tree", "RTDP DAG", "RTDP DG"]
        println("$(k)")
        a7,b7 = process_model(models_results[k].val_gamma08,get_res=res_fun)
        max_ind7 = argmax(a7)
        print("& $(round(a7[max_ind7], digits=2)) ± $(round(b7[max_ind7], digits=2)) ")
        a8,b8 = process_model(models_results[k].val_gamma09,get_res=res_fun)
        max_ind8 = argmax(a8)
        print("& $(round(a8[max_ind8], digits=2)) ± $(round(b8[max_ind8], digits=2)) ")
        a9,b9 = process_model(models_results[k].val_gamma95,get_res=res_fun)
        max_ind9 = argmax(a9)
        print("& $(round(a9[max_ind9], digits=2)) ± $(round(b9[max_ind9], digits=2)) ")
        a10,b10 = process_model(models_results[k].val_gamma99,get_res=res_fun)
        max_ind10 = argmax(a10)
        print("& $(round(a10[max_ind10], digits=2)) ± $(round(b10[max_ind10], digits=2)) ")
        a11,b11 = process_model(models_results[k].val_gamma1,get_res=res_fun)
        max_ind11 = argmax(a11)
        print("& $(round(a11[max_ind11], digits=2)) ± $(round(b11[max_ind11], digits=2)) ")


        # a1,b1 = process_model(models_results[k].train_gamma08,get_res=res_fun)
        # print("& $(round(a1[max_ind7], digits=2)) ± $(round(b1[max_ind7], digits=2)) ")
        # a2,b2 = process_model(models_results[k].train_gamma09,get_res=res_fun)
        # print("& $(round(a2[max_ind8], digits=2)) ± $(round(b2[max_ind8], digits=2)) ")
        # a2,b2 = process_model(models_results[k].train_gamma95,get_res=res_fun)
        # print("& $(round(a2[max_ind9], digits=2)) ± $(round(b2[max_ind9], digits=2)) ")
        # a2,b2 = process_model(models_results[k].train_gamma99,get_res=res_fun)
        # print("& $(round(a2[max_ind10], digits=2)) ± $(round(b2[max_ind10], digits=2)) ")
        # a3,b3 = process_model(models_results[k].train_gamma1,get_res=res_fun)
        # print("& $(round(a3[max_ind11], digits=2)) ± $(round(b3[max_ind11], digits=2)) ")

        # a4,b4 = process_model(models_results[k].test_gamma08,get_res=res_fun)
        # print("& $(round(a4[max_ind7], digits=2)) ± $(round(b4[max_ind7], digits=2)) ")
        # a5,b5 = process_model(models_results[k].test_gamma09,get_res=res_fun)
        # print("& $(round(a5[max_ind8], digits=2)) ± $(round(b5[max_ind8], digits=2)) ")
        # a5,b5 = process_model(models_results[k].test_gamma95,get_res=res_fun)
        # print("& $(round(a5[max_ind9], digits=2)) ± $(round(b5[max_ind9], digits=2)) ")
        # a5,b5 = process_model(models_results[k].test_gamma99,get_res=res_fun)
        # print("& $(round(a5[max_ind10], digits=2)) ± $(round(b5[max_ind10], digits=2)) ")
        # a6,b6 = process_model(models_results[k].test_gamma1,get_res=res_fun)
        # print("& $(round(a6[max_ind11], digits=2)) ± $(round(b6[max_ind11], digits=2)) ")

        
        println()
    end
end


function plot_bar(models_results, methods_names=["VL Tree", "VL DAG", "VL DG", "RTDP Tree", "RTDP DAG", "RTDP DG", "Planning"];gamma=0.8, ep=100)
    big_df = DataFrame()
    for (ind, m) in enumerate(methods_names)
        if m == "PPOTreeSolutionBoosted"
            gamma = 0.99
        end
        if m == "Planning"
            csv_name = getfield(models_results[m], Symbol("train"))[1]
            df = CSV.read(replace(csv_name, "ep\$(ep)" => "ep20"), DataFrame)
            m = "L*"
        else
            gamma_name = length("$gamma") > 2 ? last(replace("$gamma", "." => ""), 2) : replace("$gamma", "." => "")
            gamma_name = gamma == 1 ? "1" : gamma_name
            @show gamma_name, m
            csv_name = getfield(models_results[m], Symbol("train_gamma$gamma_name"))[1]
            df = CSV.read(replace(csv_name, "ep\$(ep)" => "ep20"), DataFrame)
        end
        if ind == 1
            big_df[!, :s0] = df[!,1]
        end 
        # big_df[!, m] = (df[!,1] - df[!,2] .+ 1) ./ df[!, 1]
        big_df[!, m] = (df[!,1] - df[!,2])
        # big_df[!, m] = map(x->only(Meta.parse(x).args), df[!,4])
        # big_df[!, m] = df[!,2] ./ df[!,1]
    end
    big_df_test = DataFrame()
    for (ind,m) in enumerate(methods_names)
        if m == "PPOTreeSolutionBoosted"
            gamma = 0.99
        end
        if m == "Planning"
            csv_name = getfield(models_results[m], Symbol("test"))[1]
            @show csv_name
            df = CSV.read(replace(csv_name, "ep\$(ep)" => "ep20"), DataFrame)
            m = "L*"
            # @show df
        else
            gamma_name = length("$gamma") > 2 ? last(replace("$gamma", "." => ""), 2) : replace("$gamma", "." => "")
            gamma_name = gamma == 1 ? "1" : gamma_name
            csv_name = getfield(models_results[m], Symbol("test_gamma$gamma_name"))[1]
            df = CSV.read(replace(csv_name, "ep\$(ep)" => "ep20"), DataFrame)
        end
        if ind == 1
            big_df_test[!, :s0] = df[!,1]
        end
        # big_df_test[!, m] = (df[!,1] - df[!,2] .+ 1) ./ df[!, 1]
        big_df_test[!, m] = (df[!,1] - df[!,2])
        # big_df_test[!, m] = map(x->only(Meta.parse(x).args), df[!,4])
        # big_df_test[!, m] = df[!,2] ./ df[!,1]
    end
    groups = groupby(big_df, :s0)
    combined_df = combine(groups,names(big_df)[2:end] .=> mean)
    # weighted_overall = sum(combined_df.n .* combined_df.mean_fraction) / sum(combined_df.n)

    plot_data = stack(combined_df, Not(:s0))
    rename!(plot_data, :variable => :method, :value => :performance)
    ctg = repeat(names(big_df)[2:end], inner = first(size(combined_df)))
    b1 = groupedbar(plot_data.:s0, plot_data.performance, bar_width=1.6,group=ctg, xlabel="Expressions size", ylabel="Average Expression Simplification", legend=:topleft, title="Train γ=$gamma", xticks = (plot_data.:s0, string.(Int.(plot_data.:s0))))
    
    combined_df = combine(groupby(big_df_test, :s0),names(big_df_test)[2:end] .=> mean)
    plot_data = stack(combined_df, Not(:s0))
    rename!(plot_data, :variable => :method, :value => :performance)
    ctg = repeat(names(big_df)[2:end], inner = first(size(combined_df)))
    b2 = groupedbar(plot_data.:s0, plot_data.performance, bar_width=1.6,group=ctg, xlabel="Expressions size", ylabel="Average Expression Simplification", legend=:topleft, title="Test γ=$gamma", xticks = (plot_data.:s0, string.(Int.(plot_data.:s0))))
    return b1, b2
end

function plot_all_bars(models_results, all_gammas=[0.8, 0.9, 0.95, 0.99, Int(1)])
    all_bars = []
    for g in all_gammas
        # plot_bars = plot_bar(models_results, ["Planning", "PolicyTreeLearningOnce", "PPOTreeLearning"], gamma=g)
        plot_bars = plot_bar(models_results, gamma=g)
        push!(all_bars, plot_bars...)
    end
    # plot_bars = plot_bar(models_results, ["Planning", "PolicyTreeSolutionBoosted1k", "PolicyTreeSolutionNotBoosted", "PPOTreeSolutionBoosted"], gamma=0.95)
    # plot_bars = plot_bar(models_results, gamma=g)
    # push!(all_bars, plot_bars...)
    layout = @layout [Plots.grid(length(all_gammas), 2, widths = [0.4, 0.6])]
    # plot(all_bars..., layout = layout, size=(1200, 600), )
    plot(all_bars..., layout = layout, size=(1800, 2000),  background_color=:transparent)
end

function plot_distribution_on_training_test()
    df_test = CSV.read("stats/planning_4th_20ep/results_of_test_heuristic_boosted_1h_test_heuristic_ep20_hidden64.csv", DataFrame) 
    df_train = CSV.read("stats/planning_4th_20ep/results_of_test_heuristic_boosted_1h_heuristic_ep20_hidden64.csv", DataFrame) 
    freq_test = countmap(df_test[!, 1])
    freq_train = countmap(df_train[!, 1])

    b_test = bar(collect(keys(freq_test)), collect(values(freq_test)), xlabel="Expressions size", ylabel="Number of Expressions", title="Expression Size Distribution", label="Testing 1k", xtick=(collect(keys(freq_test)), string.(collect(keys(freq_test)))))
    b_train = bar(collect(keys(freq_train)), collect(values(freq_train)), xlabel="Expressions size", ylabel="Number of Expressions", title="Expression Size Distribution", label="Training 1k", xtick=(collect(keys(freq_train)), string.(collect(keys(freq_train)))))
    layout = @layout [Plots.grid(1, 2, widths = [0.4, 0.6])]
    plot(b_train, b_test, layout = layout, size=(1800, 600))
end

function plot_distribution_on_all_set()
    full_map = countmap(exp_size.(train_data))
    full_map_test = countmap(exp_size.(train_data))

    b_train= bar(collect(keys(full_map)), collect(values(full_map)), xlabel="Expressions size", ylabel="Number of Expressions", title="Expression Size Distribution", label="Training all", xtick=(collect(keys(full_map)), string.(collect(keys(full_map)))), size=(1800, 400))
    b_test= bar(collect(keys(full_map_test)), collect(values(full_map_test)), xlabel="Expressions size", ylabel="Number of Expressions", title="Expression Size Distribution", label="Testing all", xtick=(collect(keys(full_map_test)), string.(collect(keys(full_map_test)))), size=(1800, 400))
    layout = @layout [Plots.grid(2, 1)]
    plot(b_train, b_test, layout = layout, size=(1800, 1200))
end


function plot_results_as_heat_map(;all_gammas=[0.8,0.9,0.95,0.99,Int(1)])
    models_results = Dict(
        "Planning" => (
            train=["stats/planning_4th_20ep/results_of_test_heuristic_boosted_1h_heuristic_ep\$(ep)_hidden64.csv",
                "stats/planning_3rd_20ep/results_of_test_heuristic_boosted_1h_heuristic_ep\$(ep)_hidden64.csv",
                "stats/planning_2nd_20ep/results_of_test_heuristic_boosted_1h_heuristic_ep\$(ep)_hidden64.csv"],
            test=["stats/planning_4th_20ep/results_of_test_heuristic_boosted_1h_test_heuristic_ep\$(ep)_hidden64.csv",
                "stats/planning_3rd_20ep/results_of_test_heuristic_boosted_1h_test_heuristic_ep\$(ep)_hidden64.csv",
                "stats/planning_2nd_20ep/results_of_test_heuristic_boosted_1h_test_heuristic_ep\$(ep)_hidden64.csv"],
            val=["stats/planning_4th_20ep/results_of_test_heuristic_boosted_1h_val_heuristic_ep\$(ep)_hidden64.csv",
            "stats/planning_3rd_20ep/results_of_test_heuristic_boosted_1h_val_heuristic_ep\$(ep)_hidden64.csv",
            "stats/planning_2nd_20ep/results_of_test_heuristic_boosted_1h_val_heuristic_ep\$(ep)_hidden64.csv"]
        ),
        # "Planning" => (
        #     train=["stats/planning_2nd_20ep/results_of_all_search_test_heuristic_boosted_1h_heuristic_ep\$(ep)_epsilon_hidden64.csv",
        #     "stats/planning_3rd_20ep/results_of_all_search_test_heuristic_boosted_1h_heuristic_ep\$(ep)_epsilon_hidden64.csv",
        #     "stats/planning_4th_20ep/results_of_all_search_test_heuristic_boosted_1h_heuristic_ep\$(ep)_epsilon_hidden64.csv"],
        #     test=["stats/planning_2nd_20ep/results_of_all_search_test_heuristic_boosted_1h_heuristic_test_ep\$(ep)_epsilon_hidden64.csv",
        #     "stats/planning_3rd_20ep/results_of_all_search_test_heuristic_boosted_1h_heuristic_test_ep\$(ep)_epsilon_hidden64.csv",
        #     "stats/planning_4th_20ep/results_of_all_search_test_heuristic_boosted_1h_heuristic_test_ep\$(ep)_epsilon_hidden64.csv"]
        # ),
        # VL 1 and 2 on Tree
        "VL Tree" => (
            train_gamma1=["stats/dqn_first_Tree_gamma1/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_Tree_gamma1_2nd/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_Tree_gamma1_3rd/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            test_gamma1=["stats/dqn_first_Tree_gamma1/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_Tree_gamma1_2nd/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_Tree_gamma1_3rd/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            val_gamma1=["stats/dqn_first_Tree_gamma1/results_of_trained_val_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_Tree_gamma1_2nd/results_of_trained_val_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
            "stats/dqn_first_Tree_gamma1_3rd/results_of_trained_val_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"],
            train_gamma09=["stats/dqn_first_Tree_gamma09/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv", 
            "stats/dqn_first_Tree_gamma09_2nd/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_first_Tree_gamma09_3rd/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"],
            test_gamma09=["stats/dqn_first_Tree_gamma09/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_first_Tree_gamma09_2nd/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_first_Tree_gamma09_3rd/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"],
            val_gamma09=["stats/dqn_first_Tree_gamma09/results_of_trained_val_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_first_Tree_gamma09_2nd/results_of_trained_val_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
            "stats/dqn_first_Tree_gamma09_3rd/results_of_trained_val_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"],
            train_gamma08=["stats/dqn_first_Tree_gamma08/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_Tree_gamma08_2nd/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_Tree_gamma08_3rd/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"],
            test_gamma08=["stats/dqn_first_Tree_gamma08/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_Tree_gamma08_2nd/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_Tree_gamma08_3rd/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"],
            val_gamma08=["stats/dqn_first_Tree_gamma08/results_of_trained_val_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_Tree_gamma08_2nd/results_of_trained_val_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
            "stats/dqn_first_Tree_gamma08_3rd/results_of_trained_val_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"],
            train_gamma99=["stats/dqn_first_Tree_gamma99/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
            "stats/dqn_first_Tree_gamma99_2nd/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
            "stats/dqn_first_Tree_gamma99_3rd/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"],
            test_gamma99=["stats/dqn_first_Tree_gamma99/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
            "stats/dqn_first_Tree_gamma99_2nd/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
            "stats/dqn_first_Tree_gamma99_3rd/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"],
            val_gamma99=["stats/dqn_first_Tree_gamma99/results_of_trained_val_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
            "stats/dqn_first_Tree_gamma99_2nd/results_of_trained_val_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
            "stats/dqn_first_Tree_gamma99_3rd/results_of_trained_val_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"],
            train_gamma95=["stats/dqn_first_Tree_gamma95/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
            "stats/dqn_first_Tree_gamma95_2nd/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
            "stats/dqn_first_Tree_gamma95_3rd/results_of_trained_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"],
            test_gamma95=["stats/dqn_first_Tree_gamma95/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
            "stats/dqn_first_Tree_gamma95_2nd/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
            "stats/dqn_first_Tree_gamma95_3rd/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"],
            val_gamma95=["stats/dqn_first_Tree_gamma95/results_of_trained_test_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
            "stats/dqn_first_Tree_gamma95_2nd/results_of_trained_val_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
            "stats/dqn_first_Tree_gamma95_3rd/results_of_trained_val_DQN_first_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"],
        ),
        "RTDP Tree" => (
            train_gamma1 = [
                "stats/dqn_second_Tree_gamma1/results_of_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_Tree_gamma1_2nd/results_of_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_Tree_gamma1_3rd/results_of_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"
            ],
            test_gamma1 = [
                "stats/dqn_second_Tree_gamma1/results_of_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_Tree_gamma1_2nd/results_of_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_Tree_gamma1_3rd/results_of_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"
            ],
            val_gamma1 = [
                "stats/dqn_second_Tree_gamma1/results_of_trained_val_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_Tree_gamma1_2nd/results_of_trained_val_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_Tree_gamma1_3rd/results_of_trained_val_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"
            ],

            train_gamma09 = [
                "stats/dqn_second_Tree_gamma09/results_of_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_Tree_gamma09_2nd/results_of_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_Tree_gamma09_3rd/results_of_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"
            ],
            test_gamma09 = [
                "stats/dqn_second_Tree_gamma09/results_of_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_Tree_gamma09_2nd/results_of_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_Tree_gamma09_3rd/results_of_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"
            ],
            val_gamma09 = [
                "stats/dqn_second_Tree_gamma09/results_of_trained_val_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_Tree_gamma09_2nd/results_of_trained_val_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_Tree_gamma09_3rd/results_of_trained_val_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"
            ],

            train_gamma08 = [
                "stats/dqn_second_Tree_gamma08/results_of_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_Tree_gamma08_2nd/results_of_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_Tree_gamma08_3rd/results_of_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"
            ],
            test_gamma08 = [
                "stats/dqn_second_Tree_gamma08/results_of_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_Tree_gamma08_2nd/results_of_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_Tree_gamma08_3rd/results_of_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"
            ],
            val_gamma08 = [
                "stats/dqn_second_Tree_gamma08/results_of_trained_val_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_Tree_gamma08_2nd/results_of_trained_val_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_Tree_gamma08_3rd/results_of_trained_val_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"
            ],

            train_gamma99 = [
                "stats/dqn_second_Tree_gamma99/results_of_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_Tree_gamma99_2nd/results_of_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_Tree_gamma99_3rd/results_of_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"
            ],
            test_gamma99 = [
                "stats/dqn_second_Tree_gamma99/results_of_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_Tree_gamma99_2nd/results_of_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_Tree_gamma99_3rd/results_of_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"
            ],
            val_gamma99 = [
                "stats/dqn_second_Tree_gamma99/results_of_trained_val_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_Tree_gamma99_2nd/results_of_trained_val_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_Tree_gamma99_3rd/results_of_trained_val_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"
            ],

            train_gamma95 = [
                "stats/dqn_second_Tree_gamma95/results_of_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_Tree_gamma95_2nd/results_of_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_Tree_gamma95_3rd/results_of_trained_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"
            ],
            test_gamma95 = [
                "stats/dqn_second_Tree_gamma95/results_of_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_Tree_gamma95_2nd/results_of_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_Tree_gamma95_3rd/results_of_trained_test_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"
            ],
            val_gamma95 = [
                "stats/dqn_second_Tree_gamma95/results_of_trained_val_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_Tree_gamma95_2nd/results_of_trained_val_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_Tree_gamma95_3rd/results_of_trained_val_DQN_second_Tree_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"
            ],
        ),

        
        # VL 1 and 2 on DAG
        "VL DAG" => (
            train_gamma1 = [
                "stats/dqn_first_DAG_gamma1/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_DAG_gamma1_2nd/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_DAG_gamma1_3rd/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"
            ],
            test_gamma1 = [
                "stats/dqn_first_DAG_gamma1/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_DAG_gamma1_2nd/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_DAG_gamma1_3rd/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"
            ],
            val_gamma1 = [
                "stats/dqn_first_DAG_gamma1/results_of_trained_val_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_DAG_gamma1_2nd/results_of_trained_val_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_DAG_gamma1_3rd/results_of_trained_val_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"
            ],

            train_gamma09 = [
                "stats/dqn_first_DAG_gamma09/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_first_DAG_gamma09_2nd/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_first_DAG_gamma09_3rd/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"
            ],
            test_gamma09 = [
                "stats/dqn_first_DAG_gamma09/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_first_DAG_gamma09_2nd/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_first_DAG_gamma09_3rd/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"
            ],
            val_gamma09 = [
                "stats/dqn_first_DAG_gamma09/results_of_trained_val_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_first_DAG_gamma09_2nd/results_of_trained_val_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_first_DAG_gamma09_3rd/results_of_trained_val_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"
            ],

            train_gamma08 = [
                "stats/dqn_first_DAG_gamma08/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_first_DAG_gamma08_2nd/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_first_DAG_gamma08_3rd/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"
            ],
            test_gamma08 = [
                "stats/dqn_first_DAG_gamma08/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_first_DAG_gamma08_2nd/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_first_DAG_gamma08_3rd/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"
            ],
            val_gamma08 = [
                "stats/dqn_first_DAG_gamma08/results_of_trained_val_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_first_DAG_gamma08_2nd/results_of_trained_val_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_first_DAG_gamma08_3rd/results_of_trained_val_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"
            ],

            train_gamma99 = [
                "stats/dqn_first_DAG_gamma99/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_first_DAG_gamma99_2nd/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_first_DAG_gamma99_3rd/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"
            ],
            test_gamma99 = [
                "stats/dqn_first_DAG_gamma99/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_first_DAG_gamma99_2nd/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_first_DAG_gamma99_3rd/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"
            ],
            val_gamma99 = [
                "stats/dqn_first_DAG_gamma99/results_of_trained_val_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_first_DAG_gamma99_2nd/results_of_trained_val_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_first_DAG_gamma99_3rd/results_of_trained_val_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"
            ],

            train_gamma95 = [
                "stats/dqn_first_DAG_gamma95/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_first_DAG_gamma95_2nd/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_first_DAG_gamma95_3rd/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"
            ],
            test_gamma95 = [
                "stats/dqn_first_DAG_gamma95/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_first_DAG_gamma95_2nd/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_first_DAG_gamma95_3rd/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"
            ],
            val_gamma95 = [
                "stats/dqn_first_DAG_gamma95/results_of_trained_val_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_first_DAG_gamma95_2nd/results_of_trained_val_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_first_DAG_gamma95_3rd/results_of_trained_val_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"
            ]
        ),

        "RTDP DAG" => (
            train_gamma1 = [
                "stats/dqn_second_DAG_gamma1/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_DAG_gamma1_2nd/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_DAG_gamma1_3rd/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"
            ],
            test_gamma1 = [
                "stats/dqn_second_DAG_gamma1/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_DAG_gamma1_2nd/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_DAG_gamma1_3rd/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"
            ],
            val_gamma1 = [
                "stats/dqn_second_DAG_gamma1/results_of_trained_val_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_DAG_gamma1_2nd/results_of_trained_val_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_DAG_gamma1_3rd/results_of_trained_val_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"
            ],

            train_gamma09 = [
                "stats/dqn_second_DAG_gamma09/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_DAG_gamma09_2nd/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_DAG_gamma09_3rd/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"
            ],
            test_gamma09 = [
                "stats/dqn_second_DAG_gamma09/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_DAG_gamma09_2nd/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_DAG_gamma09_3rd/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"
            ],
            val_gamma09 = [
                "stats/dqn_second_DAG_gamma09/results_of_trained_val_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_DAG_gamma09_2nd/results_of_trained_val_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_DAG_gamma09_3rd/results_of_trained_val_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"
            ],

            train_gamma08 = [
                "stats/dqn_second_DAG_gamma08/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_DAG_gamma08_2nd/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_DAG_gamma08_3rd/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"
            ],
            test_gamma08 = [
                "stats/dqn_second_DAG_gamma08/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_DAG_gamma08_2nd/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_DAG_gamma08_3rd/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"
            ],
            val_gamma08 = [
                "stats/dqn_second_DAG_gamma08/results_of_trained_val_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_DAG_gamma08_2nd/results_of_trained_val_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_DAG_gamma08_3rd/results_of_trained_val_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"
            ],

            train_gamma99 = [
                "stats/dqn_second_DAG_gamma99/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_DAG_gamma99_2nd/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_DAG_gamma99_3rd/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"
            ],
            test_gamma99 = [
                "stats/dqn_second_DAG_gamma99/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_DAG_gamma99_2nd/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_DAG_gamma99_3rd/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"
            ],
            val_gamma99 = [
                "stats/dqn_second_DAG_gamma99/results_of_trained_val_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_DAG_gamma99_2nd/results_of_trained_val_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_DAG_gamma99_3rd/results_of_trained_val_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"
            ],

            train_gamma95 = [
                "stats/dqn_second_DAG_gamma95/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_DAG_gamma95_2nd/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_DAG_gamma95_3rd/results_of_trained_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"
            ],
            test_gamma95 = [
                "stats/dqn_second_DAG_gamma95/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_DAG_gamma95_2nd/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_DAG_gamma95_3rd/results_of_trained_test_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"
            ],
            val_gamma95 = [
                "stats/dqn_second_DAG_gamma95/results_of_trained_val_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_DAG_gamma95_2nd/results_of_trained_val_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_DAG_gamma95_3rd/results_of_trained_val_DQN_second_DAG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"
            ]
        ),

        
        # VL 1 and 2 on DG
        "VL DG" => (
            train_gamma1 = [
                "stats/dqn_first_DG_gamma1/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_DG_gamma1_2nd/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_DG_gamma1_3rd/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"
            ],
            test_gamma1 = [
                "stats/dqn_first_DG_gamma1/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_DG_gamma1_2nd/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_DG_gamma1_3rd/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"
            ],
            val_gamma1 = [
                "stats/dqn_first_DG_gamma1/results_of_trained_val_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_DG_gamma1_2nd/results_of_trained_val_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_first_DG_gamma1_3rd/results_of_trained_val_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"
            ],

            train_gamma09 = [
                "stats/dqn_first_DG_gamma09/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_first_DG_gamma09_2nd/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_first_DG_gamma09_3rd/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"
            ],
            test_gamma09 = [
                "stats/dqn_first_DG_gamma09/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_first_DG_gamma09_2nd/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_first_DG_gamma09_3rd/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"
            ],
            val_gamma09 = [
                "stats/dqn_first_DG_gamma09/results_of_trained_val_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_first_DG_gamma09_2nd/results_of_trained_val_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_first_DG_gamma09_3rd/results_of_trained_val_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"
            ],

            train_gamma08 = [
                "stats/dqn_first_DG_gamma08/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_first_DG_gamma08_2nd/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_first_DG_gamma08_3rd/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"
            ],
            test_gamma08 = [
                "stats/dqn_first_DG_gamma08/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_first_DG_gamma08_2nd/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_first_DG_gamma08_3rd/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"
            ],
            val_gamma08 = [
                "stats/dqn_first_DG_gamma08/results_of_trained_val_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_first_DG_gamma08_2nd/results_of_trained_val_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_first_DG_gamma08_3rd/results_of_trained_val_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"
            ],

            train_gamma99 = [
                "stats/dqn_first_DG_gamma99/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_first_DG_gamma99_2nd/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_first_DG_gamma99_3rd/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"
            ],
            test_gamma99 = [
                "stats/dqn_first_DG_gamma99/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_first_DG_gamma99_2nd/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_first_DG_gamma99_3rd/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"
            ],
            val_gamma99 = [
                "stats/dqn_first_DG_gamma99/results_of_trained_val_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_first_DG_gamma99_2nd/results_of_trained_val_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_first_DG_gamma99_3rd/results_of_trained_val_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"
            ],

            train_gamma95 = [
                "stats/dqn_first_DG_gamma95/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_first_DG_gamma95_2nd/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_first_DG_gamma95_3rd/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"
            ],
            test_gamma95 = [
                "stats/dqn_first_DG_gamma95/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_first_DG_gamma95_2nd/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_first_DG_gamma95_3rd/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"
            ],
            val_gamma95 = [
                "stats/dqn_first_DG_gamma95/results_of_trained_val_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_first_DG_gamma95_2nd/results_of_trained_val_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_first_DG_gamma95_3rd/results_of_trained_val_DQN_first_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"
            ]
        ),

        "RTDP DG" => (
            train_gamma1 = [
                "stats/dqn_second_DG_gamma1/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_DG_gamma1_2nd/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_DG_gamma1_3rd/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"
            ],
            test_gamma1 = [
                "stats/dqn_second_DG_gamma1/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_DG_gamma1_2nd/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_DG_gamma1_3rd/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"
            ],
            val_gamma1 = [
                "stats/dqn_second_DG_gamma1/results_of_trained_val_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_DG_gamma1_2nd/results_of_trained_val_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv",
                "stats/dqn_second_DG_gamma1_3rd/results_of_trained_val_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma1_hidden64.csv"
            ],

            train_gamma09 = [
                "stats/dqn_second_DG_gamma09/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_DG_gamma09_2nd/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_DG_gamma09_3rd/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"
            ],
            test_gamma09 = [
                "stats/dqn_second_DG_gamma09/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_DG_gamma09_2nd/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_DG_gamma09_3rd/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"
            ],
            val_gamma09 = [
                "stats/dqn_second_DG_gamma09/results_of_trained_val_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_DG_gamma09_2nd/results_of_trained_val_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv",
                "stats/dqn_second_DG_gamma09_3rd/results_of_trained_val_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma09_hidden64.csv"
            ],

            train_gamma08 = [
                "stats/dqn_second_DG_gamma08/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_DG_gamma08_2nd/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_DG_gamma08_3rd/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"
            ],
            test_gamma08 = [
                "stats/dqn_second_DG_gamma08/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_DG_gamma08_2nd/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_DG_gamma08_3rd/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"
            ],
            val_gamma08 = [
                "stats/dqn_second_DG_gamma08/results_of_trained_val_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_DG_gamma08_2nd/results_of_trained_val_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv",
                "stats/dqn_second_DG_gamma08_3rd/results_of_trained_val_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma08_hidden64.csv"
            ],

            train_gamma99 = [
                "stats/dqn_second_DG_gamma99/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_DG_gamma99_2nd/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_DG_gamma99_3rd/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"
            ],
            test_gamma99 = [
                "stats/dqn_second_DG_gamma99/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_DG_gamma99_2nd/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_DG_gamma99_3rd/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"
            ],
            val_gamma99 = [
                "stats/dqn_second_DG_gamma99/results_of_trained_val_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_DG_gamma99_2nd/results_of_trained_val_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv",
                "stats/dqn_second_DG_gamma99_3rd/results_of_trained_val_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma99_hidden64.csv"
            ],

            train_gamma95 = [
                "stats/dqn_second_DG_gamma95/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_DG_gamma95_2nd/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_DG_gamma95_3rd/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"
            ],
            test_gamma95 = [
                "stats/dqn_second_DG_gamma95/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_DG_gamma95_2nd/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_DG_gamma95_3rd/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"
            ],
            val_gamma95 = [
                "stats/dqn_second_DG_gamma95/results_of_trained_val_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_DG_gamma95_2nd/results_of_trained_val_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv",
                "stats/dqn_second_DG_gamma95_3rd/results_of_trained_val_DQN_second_DG_not_boosted_ep\$(ep)_batch128_gamma95_hidden64.csv"
            ]
        ),
    )
    methods_names = sort(collect(keys(models_results)))
    all_heat_plots =  map(all_gammas) do gamma
        big_df = DataFrame()
        for (ind, m) in enumerate(methods_names)
            if m == "Planning"
                csv_names = getfield(models_results[m], Symbol("train"))
                df = CSV.read(replace(csv_names[1], "ep\$(ep)" => "ep20"), DataFrame)
                tmp = map(csv_names) do path_name
                    df = CSV.read(replace(path_name, "ep\$(ep)" => "ep20"), DataFrame)
                    df[!, 2]
                end
            else
                gamma_name = length("$gamma") > 2 ? last(replace("$gamma", "." => ""), 2) : replace("$gamma", "." => "")
                gamma_name = gamma == 1 ? "1" : gamma_name
                # @show gamma_name, m
                csv_names = getfield(models_results[m], Symbol("train_gamma$gamma_name"))
                df = CSV.read(replace(csv_names[1], "ep\$(ep)" => "ep20"), DataFrame)
                tmp = map(csv_names) do path_name
                    df = CSV.read(replace(path_name, "ep\$(ep)" => "ep20"), DataFrame)
                    df[!, 2]
                end
            end
            if ind == 1
                big_df[!, :s0] = df[!,1]
            end
            big_df[!, m] = mean(tmp)
        end
        big_df_test = DataFrame()
        for (ind,m) in enumerate(methods_names)
            if m == "Planning"
                csv_names = getfield(models_results[m], Symbol("test"))
                # @show csv_names
                df = CSV.read(replace(csv_names[1], "ep\$(ep)" => "ep20"), DataFrame)
                tmp = map(csv_names) do path_name
                    df = CSV.read(replace(path_name, "ep\$(ep)" => "ep20"), DataFrame)
                    df[!, 2]
                end
            else
                gamma_name = length("$gamma") > 2 ? last(replace("$gamma", "." => ""), 2) : replace("$gamma", "." => "")
                gamma_name = gamma == 1 ? "1" : gamma_name
                csv_names = getfield(models_results[m], Symbol("test_gamma$gamma_name"))
                df = CSV.read(replace(csv_names[1], "ep\$(ep)" => "ep20"), DataFrame)
                tmp = map(csv_names) do path_name
                    df = CSV.read(replace(path_name, "ep\$(ep)" => "ep20"), DataFrame)
                    df[!, 2]
                end
            end
            if ind == 1
                big_df_test[!, :s0] = df[!,1]
            end
            big_df_test[!, m] =  mean(tmp)
        end
        
        combined_df = combine(groupby(big_df, :s0),names(big_df)[2:end] .=> mean)
        quality = Matrix(combined_df[!, 2:end])'
        exp_names = string.(combined_df[!,1])
        h_train = heatmap(
            exp_names,            # x-axis labels (environments)
            methods_names,         # y-axis labels (methods)
            quality,         # data matrix
            c=:viridis,      # color scheme (e.g., :viridis, :blues, :coolwarm)
            colorbar_title="Solution Size",
            title="γ=$gamma",
            yticks = gamma == 0.8 ? (1:length(methods_names), methods_names) : false,
            ylabel = "",
            xlabel = "",
            xrotation = 45,
            colorbar = gamma == 2 ? true : false,
            framestyle = :box,
        )
        combined_df = combine(groupby(big_df_test, :s0),names(big_df_test)[2:end] .=> mean)
        quality = Matrix(combined_df[!, 2:end])'
        exp_names = string.(combined_df[!,1])
        h_test =  heatmap(
            exp_names,            # x-axis labels (environments)
            methods_names,         # y-axis labels (methods)
            quality,         # data matrix
            c=:viridis,      # color scheme (e.g., :viridis, :blues, :coolwarm)
            xlabel="Environment",
            ylabel="",
            title="",
            yticks = gamma == 0.8 ? (1:length(methods_names), methods_names) : false,
            xrotation = 45,
            colorbar = gamma == 2 ? true : false,
            framestyle = :box,
        )
        return(h_train, h_test)
    end
    all_heat_plots = [[x[1] for x in all_heat_plots], [x[2] for x in all_heat_plots]]
    # all_heat_plots = collect(Iterators.flatten(all_heat_plots))
    # layout = @layout [Plots.grid(5, 2, widths = [0.4, 0.6])]
    layout = @layout [Plots.grid(1, 5)]
    plot(all_heat_plots[1]..., layout = layout, size=(2500, 600))
end

p = plot_results_as_heat_map()
colorbar_plot = heatmap(
    [0 1; 0 1],      # dummy 2x2 data just for colorbar scaling
    c = :viridis,    # match the color scheme used in your real heatmaps
    axis = false,    # hide axes
    ticks = false,   # no ticks
    framestyle = :none,
    legend = false,
    title = "",
    colorbar_title = "Solution Size"
)

plot!(p, colorbar_plot)
plot(
    p...,
    layout = layout,
    size = (2500, 1200),              # Adjust as needed
    bottom_margin = 10mm,
    top_margin = 10mm,
    colorbar = false                  # Avoid enabling global colorbar
)
# new_train = Dict()
# for i in [5,7,9,11,13,15,17,19,21,23,25,27,29]
# if haskey(new_train, i)
# push!(new_train, )
# end
# end

means_test_pl, means_test_std_pl = process_model(getfield(models_results["Planning"], Symbol("test")), final_epoch=20)
means_train_pl, means_train_std_pl = process_model(getfield(models_results["Planning"], Symbol("train")), final_epoch=20)
means_test_reinforce, means_test_std_reinforce = process_model(getfield(models_results["PolicyTreeLearningOnce"], Symbol("test_gamma99")), final_epoch=100, get_res=get_res)
means_train_reinforce, means_train_std_reinforce = process_model(getfield(models_results["PolicyTreeLearningOnce"], Symbol("train_gamma99")), final_epoch=100, get_res=get_res)

means_test_reinforce, means_test_std_reinforce = process_model(getfield(models_results["PolicyTreeLearning"], Symbol("test_gamma99")), final_epoch=100, get_res=get_res)
means_train_reinforce, means_train_std_reinforce = process_model(getfield(models_results["PolicyTreeLearning"], Symbol("train_gamma99")), final_epoch=100, get_res=get_res)
#  results_of_trained_val_tree_PG_reinforce_policy_Tree_not_boosted_ep97_batch256_gamma99_hidden64
means_test_ppo, means_test_std_ppo = process_model(getfield(models_results["PPOTreeLearning"], Symbol("test_gamma99")), final_epoch=20)
means_train_ppo, means_train_std_ppo = process_model(getfield(models_results["PPOTreeLearning"], Symbol("train_gamma99")), final_epoch=20)
means_test_linear, means_test_std_linear = process_model(getfield(models_results["PolicyLinearLearning"], Symbol("test_gamma99")), final_epoch=20)
means_train_linear, means_train_std_linear = process_model(getfield(models_results["PolicyLinearLearning"], Symbol("train_gamma99")), final_epoch=20)


# means_test_pl2, means_test_std_pl2 = process_model(getfield(models_results["Planning100ep_2dense"], Symbol("test")), final_epoch=100, get_res=get_res1)
# means_train_pl2, means_train_std_pl2 = process_model(getfield(models_results["Planning100ep_2dense"], Symbol("train")), final_epoch=100, get_res=get_res1)

# means_test_pl3, means_test_std_pl3 = process_model(getfield(models_results["Planning100ep_3dense"], Symbol("test")), final_epoch=100, get_res=get_res1)
# means_train_pl3, means_train_std_pl3 = process_model(getfield(models_results["Planning100ep_3dense"], Symbol("train")), final_epoch=100, get_res=get_res1)

# means_test_pl2e, means_test_std_pl2e = process_model(getfield(models_results["Planning100ep_2dense_epsilon"], Symbol("test")), final_epoch=100, get_res=get_res1)
# means_train_pl2e, means_train_std_pl2e = process_model(getfield(models_results["Planning100ep_2dense_epsilon"], Symbol("train")), final_epoch=100, get_res=get_res1)

# means_test_pl3e, means_test_std_pl3e = process_model(getfield(models_results["Planning100ep_3dense_epsilon"], Symbol("test")), final_epoch=100, get_res=get_res1)
# means_train_pl3e, means_train_std_pl3e = process_model(getfield(models_results["Planning100ep_3dense_epsilon"], Symbol("train")), final_epoch=100, get_res=get_res1)

means_train_reinforce_tree_mdp, means_train_std_reinforce_tree_mdp = process_model(getfield(models_results["PolicyTreeMDPSolution"], Symbol("train_gamma99")), final_epoch=100)
means_test_reinforce_tree_mdp, means_test_std_reinforce_tree_mdp = process_model(getfield(models_results["PolicyTreeMDPSolution"], Symbol("test_gamma99")), final_epoch=100)
Plots.plot(t, means_train_reinforce_tree_mdp, color=color_B, linestyle=style_train, label="Tree Reinforce (Train)")
Plots.plot!(t, means_test_reinforce_tree_mdp, color=color_B, linestyle=style_test,  label="Tree Reinforce (Test)")
t = 1:100
color_A = :blue
color_B = :red
color_C = :green
color_D = :orange
style_train = :solid
style_test  = :dash
# Plots.plot(t, means_train_pl2, color=color_A, linestyle=style_train, label="L* 2d (Train)")
# Plots.plot(t, means_test_pl2,  color=color_A, linestyle=style_test,  label="L* 2d (Test)")

# Plots.plot!(t, means_train_pl3, color=color_B, linestyle=style_train, label="L* 3d (Train)")
# Plots.plot!(t, means_test_pl3,  color=color_B, linestyle=style_test,  label="L* 3d (Test)")

# Plots.plot!(t, means_train_pl2e, color=color_C, linestyle=style_train, label="L* 2d ϵ (Train)")
# Plots.plot!(t, means_test_pl2e,  color=color_C, linestyle=style_test,  label="L* 2d ϵ (Test)")

Plots.plot(t, means_train_pl3e, color=color_D, linestyle=style_train, label="L* 3d ϵ (Train)")
Plots.plot!(t, means_test_pl3e,  color=color_D, linestyle=style_test,  label="L* 3d ϵ (Test)")


# Plots.plot!(title="Train/Test Performance Comparison",
#          xlabel="Epoch",
#          ylabel="Mean simplification",
#          legend=:outerbottom, legendcolumns=2)
is_std = false
if is_std
Plots.plot(t, means_train_pl, ribbon=means_train_std_pl, color=color_A, linestyle=style_train, label="L* (Train)")
Plots.plot!(t, means_test_pl,  ribbon=means_test_std_pl,  color=color_A, linestyle=style_test,  label="L* (Test)")
else
Plots.plot(t, means_train_pl, color=color_A, linestyle=style_train, label="L* (Train)")
Plots.plot!(t, means_test_pl,  color=color_A, linestyle=style_test,  label="L* (Test)")
end
# Plot Method B
if is_std
Plots.plot!(t, means_train_reinforce, ribbon=means_train_std_reinforce, color=color_B, linestyle=style_train, label="Tree Reinforce (Train)")
Plots.plot!( t, means_test_reinforce,  ribbon=means_test_std_reinforce,  color=color_B, linestyle=style_test,  label="Tree Reinforce (Test)")
else
Plots.plot!(t, means_train_reinforce, color=color_B, linestyle=style_train, label="Tree Reinforce (Train)")
Plots.plot!(t, means_test_reinforce, color=color_B, linestyle=style_test,  label="Tree Reinforce (Test)")
end

if is_std
Plots.plot!(t, means_train_ppo, ribbon=means_train_std_ppo, color=color_C, linestyle=style_train, label="Tree PPO (Train)")
Plots.plot!(t, means_test_ppo,  ribbon=means_test_std_ppo,  color=color_C, linestyle=style_test,  label="Tree PPO (Test)")
else
Plots.plot!(t, means_train_ppo, color=color_C, linestyle=style_train, label="Tree PPO (Train)")
Plots.plot!(t, means_test_ppo, color=color_C, linestyle=style_test,  label="Tree PPO (Test)")
end
if is_std
Plots.plot!(t, means_train_linear, ribbon=means_train_std_ppo, color=color_D, linestyle=style_train, label="Tree PPO (Train)")
Plots.plot!(t, means_test_linear,  ribbon=means_test_std_ppo,  color=color_D, linestyle=style_test,  label="Tree PPO (Test)")
else
Plots.plot!(t, means_train_linear, color=color_D, linestyle=style_train, label="Linear Reinforce (Train)")
Plots.plot!(t, means_test_linear, color=color_D, linestyle=style_test,  label="Linear Reinforce (Test)")
end
Plots.plot!(title="Train/Test Performance Comparison",
         xlabel="Epoch",
         ylabel="Mean simplification",
         legend=:outerbottom, legendcolumns=2)



gammas = [
    (:gamma09, "09"),
    (:gamma95, "95"),
    (:gamma99, "99"),
    (:gamma10, "10"),
]

# Helper: safe call for a single gamma and mode (:train or :test)
function safe_single(k, mode, gsym; res_fun=get_res2)
    # key = Symbol(mode, "_", gsym)
    # vkey = Symbol("val", "_", gsym)
    key = Symbol(mode, )
    vkey = Symbol("val", )
    try
        av, _ = process_model(models_results[k][vkey], get_res=res_fun)
        a, _ = process_model(models_results[k][key], get_res=res_fun)
        # @show argmax(av)
        return round(a[argmax(av)], digits=2)
    catch e
        # @show e
        return "--"
    end
end

# results_names = ["PPOTreeNoSolutionBoosted", "PPOTreeNoSolutionNotBoosted", "PPOTreeSolutionBoosted", "PPOTreeSolutionNotBoosted"]
# results_names = ["PolicyTreeSolutionBoosted1k", "PolicyTreeSolutionNotBoosted1k", "PolicyTreeNoSolutionBoosted", "PolicyTreeNoSolutionNotBoosted", "PolicyTreeSolutionBoosted", "PolicyTreeSolutionNotBoosted"]
# results_names = ["Planning100ep_2dense", "Planning100ep_3dense", "Planning100ep_2dense_epsilon", "Planning100ep_3dense_epsilon", "Planning1000ep_2dense", "Planning1000ep_3dense", "Planning1000ep_2dense_epsilon", "Planning1000ep_3dense_epsilon"]
results_names = ["PPOmaxVlTreeNoSolutionBoosted", "PPOmaxVlTreeNoSolutionNotBoosted", "PPOmaxVlTreeSolutionBoosted", "PPOmaxVlTreeSolutionNotBoosted"]
# results_names = ["PPOLinearNoSolutionNotBoosted", "PPOTreeCorrectSolutionBoosted"]
# Main loop
for k in results_names
    print("$(k) ")

    ### 1. TRAIN VALUES FOR ALL GAMMAS
    for (gsym, _label) in gammas
        val = safe_single(k, :train, gsym)
        print("& $(val) ")
    end

    ### 2. TEST VALUES FOR ALL GAMMAS
    for (gsym, _label) in gammas
        val = safe_single(k, :test, gsym)
        print("& $(val) ")
    end

    println()
end



results_names = ["Planning100ep_2dense", "Planning100ep_3dense", "Planning100ep_2dense_epsilon", "Planning100ep_3dense_epsilon", "Planning100ep_2dense_1innep", "Planning100ep_3dense_1innep", "Planning100ep_2dense_epsilon_1innep", "Planning100ep_3dense_epsilon_1innep"]
for k in results_names
    println("$(k)")
    try
        a8,b8 = process_model(models_results[k].val,get_res=get_res1)
        max_ind8 = argmax(a8)

        a2,b2 = process_model(models_results[k].train,get_res=get_res1)
        print("& $(round(a2[max_ind8], digits=2)) ")

        a5,b5 = process_model(models_results[k].test,get_res=get_res1)
        print("& $(round(a5[max_ind8], digits=2)) ")    
    catch e
        @show e
    end
    println()
end



# models_test = deserialize("models/reinforce_policy_Linear_100exp_gamma09/bdg_trained_parallel_linear_PG_no_solution_reinforce_policy_Linear_not_boosted_ep1_batch256_gamma09_for_graph_stats.bin")

models = deserialize("models/reinforce_policy_Linear_100exp_gamma09/reinforce_policy_Linear_100exp_gamma09/trained_parallel_linear_PG_no_solution_reinforce_policy_Linear_not_boosted_ep16_batch256_gamma09_for_graph_stats.bin")
models = deserialize("models/reinforce_policy_Linear_100exp_gamma09/reinforce_policy_Linear_100exp_gamma09/trained_parallel_linear_PG_no_solution_reinforce_policy_Linear_not_boosted_ep15_batch256_gamma09_for_graph_stats.bin")
for (ind,d) in enumerate(data[900:end])
    @show ind, d
    t = @elapsed tmp1 = sample_trajectory(sampler, data[950], models)
    @show t
end

env = MyTreeEnv(data[950], models)
broken = []
for _ in 1:100
    reset!(env)
    broken = []
    for t in 1:sampler.max_steps
        @show t
        # @show env.s_init, env.s_current
        possible_actions = action_space(env)
        # push!(broken, possible_actions)
        broken = possible_actions
        weights = [only(models(x)) for x in possible_actions]
        node_index = StatsBase.sample(1:length(weights), Weights(softmax(weights)))
        a = possible_actions[node_index]
        s = state(env)
        act!(env, a)
        ns = state(env)
        r = reward(env, a, s)
        is_done = isempty(action_space(env)) || is_terminal(env)
        # push2traj!(traj, (s, a, r, ns, is_done))
        # other_indexes = setdiff(1:length(possible_actions), [node_index])
        # @show possible_actions[other_indexes]
        # @show a
        # push!(inputs_actions, vcat(deepcopy(a), possible_actions[other_indexes]))
        if is_done
            break
        end
    end
end

broken_input = deserialize("inputs_for_bad_gradient.bin")
learner = PolicyLerner(Flux.mse, model, max_iter=10)
loss = 0
for (ind,s) in enumerate(broken_input)
@elapsed loss += compute_gradient1!(s, model, learner)
@show ind
isnan(loss) && break
end
# weights = [only(model(x)) for x in possible_actions]
# weights = vec(MyModule.heuristic(models_test, broken_input.ds))
# for i in broken_input.softmax_ids
#     @show i
#     node_index = StatsBase.sample(1:length(weights), Weights(softmax(weights[i])))
# end

a, _ = process_model(["stats/ppo_pevnak/results_of_trained_trained_parallel_gae_lr_no_solution_tree_PPO_ep\$(ep)_gamma09_innep1_for_graph_stats.csv"], get_res=get_res, final_epoch=25)
a1, _ = process_model(["stats/ppo_pevnak/results_of_trained_val_trained_parallel_tree_PPO_ep\$(ep)_gamma09_innep20_for_graph_stats.csv"], get_res=get_res, final_epoch=48)
a2, _ = process_model(["stats/ppo_pevnak/results_of_trained_test_trained_parallel_tree_PPO_ep\$(ep)_gamma09_innep20_for_graph_stats.csv"], get_res=get_res, final_epoch=48)
plot([a,a1,a2],labels=["training" "val" "test"])


c, _ = process_model(["stats/ppo_pevnak/results_of_trained_trained_parallel_gae_tree_PPO_ep\$(ep)_gamma09_innep1_for_graph_stats.csv"], get_res=get_res, final_epoch=24)
plot([a,b], labels=["training Tomas" "training Oleksii"])


d, _ = process_model(["stats/ppo_pevnak/results_of_trained_trained_parallel_gae_norm_lr_linear_PPO_ep\$(ep)_gamma09_innep1_for_graph_stats.csv"], get_res=get_res3, final_epoch=100)
p1 = plot(d, title="Mean number of Unique exploreed nodes", label=false)

d, _ = process_model(["stats/ppo_pevnak/results_of_trained_trained_parallel_gae_norm_lr_linear_PPO_ep\$(ep)_gamma09_innep1_for_graph_stats.csv"], get_res=get_res4, final_epoch=100)
dst, _ = process_model(["stats/ppo_pevnak/results_of_trained_trained_parallel_gae_norm_lr_linear_PPO_ep\$(ep)_gamma09_innep1_for_graph_stats.csv"], get_res=get_res5, final_epoch=100)
dst[1] = 0.0
p2 = plot(1:length(d),d,ribbon=dst,lw=2, title="Mean rewards", label=false)

plot(p1, p2, layout=(2,1))