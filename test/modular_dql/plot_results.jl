using CSV
using DataFrames
using Statistics
using Plots

function get_res(paths;epochs=20)
    tmp = map(paths) do path_name
        res = map(1:epochs) do e
            path = replace(path_name, "ep\$(ep)" => "ep$(e)")
            df = CSV.read(path, DataFrame)
            mean(df[!,1] .- df[!,2])
        end
    end
    return tmp
end
function plot_all_results()
    # Planning Tree Training, Testing
    planning_paths = ["stats/planning_4th_20ep/results_of_test_heuristic_boosted_1h_heuristic_ep\$(ep)_hidden64.csv",
    "stats/planning_3rd_20ep/results_of_test_heuristic_boosted_1h_heuristic_ep\$(ep)_hidden64.csv",
    "stats/planning_2nd_20ep/results_of_test_heuristic_boosted_1h_heuristic_ep\$(ep)_hidden64.csv"]
    planning_res = hcat(get_res(planning_paths)...)
    means_planning_res = vec(mean(planning_res, dims=2))
    stds_planning_res = vec(std(planning_res, dims=2))
    # plot(1:length(stds_planning_res), means_planning_res, ribbon=stds_planning_res, label="Planning Mean ± Std", lw=2, xlabel="Epochs", ylabel="Mean simplification", 
    #     legend=:outerbottom, legendcolumn=2, size=(1000, 800))
    # plot(1:length(stds_planning_res), means_planning_res, label="Planning Mean", lw=2, xlabel="Epochs", ylabel="Mean simplification", 
    #     legend=:outerbottom, legendcolumn=2, size=(1000, 800), title="Test on Training Dataset")

    planning_paths_test = ["stats/planning_4th_20ep/results_of_test_heuristic_boosted_1h_test_heuristic_ep\$(ep)_hidden64.csv",
    "stats/planning_3rd_20ep/results_of_test_heuristic_boosted_1h_test_heuristic_ep\$(ep)_hidden64.csv",
    "stats/planning_2nd_20ep/results_of_test_heuristic_boosted_1h_test_heuristic_ep\$(ep)_hidden64.csv"]
    planning_test_res = hcat(get_res(planning_paths_test)...)
    means_planning_test_res = vec(mean(planning_test_res, dims=2))
    stds_planning_test_res = vec(std(planning_test_res, dims=2))
    # plot!(1:length(stds_planning_test_res), means_planning_test_res, ribbon=stds_planning_test_res, label="Planning Test Mean ± Std", lw=2, xlabel="Epochs", ylabel="Mean simplification")
    plot(1:length(stds_planning_test_res), means_planning_test_res, label="Planning Test Mean", lw=2, xlabel="Epochs", ylabel="Mean simplification", legend=:outerbottom, legendcolumn=2, size=(1000, 800), title="Test on Testing Dataset")
    
    # DQN first on Tree Training, Testing
    dqn_first_paths = ["stats/dqn_first_4th_20ep/results_of_trained_DQN_first_DAG1_ep\$(ep)_hidden64.csv",
    "stats/dqn_first_3rd_20ep/results_of_trained_DQN_first_DAG1_ep\$(ep)_hidden64.csv",
    "stats/dqn_first_2nd_20ep/results_of_trained_DQN_first_DAG1_not_boosted_ep\$(ep)_hidden64.csv"]
    dqn_first_res = hcat(get_res(dqn_first_paths)...)
    means_dqn_first_res = vec(mean(dqn_first_res, dims=2))
    stds_dqn_first_res = vec(std(dqn_first_res, dims=2))
    # plot!(1:length(stds_dqn_first_res), means_dqn_first_res, ribbon=stds_dqn_first_res, label="DQN 1 Tree Mean ± Std", lw=2, xlabel="Epochs", ylabel="Mean simplification")
    # plot!(1:length(stds_dqn_first_res), means_dqn_first_res, label="DQN 1 Tree Mean", lw=2, xlabel="Epochs", ylabel="Mean simplification")
    
    dqn_first_paths_test = ["stats/dqn_first_4th_20ep/results_of_test_trained_not_boosted_DQN_first_DAG1_ep\$(ep)_hidden64.csv",
    "stats/dqn_first_3rd_20ep/results_of_test_trained_DQN_first_DAG1_ep\$(ep)_hidden64.csv",
    "stats/dqn_first_2nd_20ep/results_of_test_trained_DQN_first_DAG1_ep\$(ep)_hidden64.csv"]
    dqn_first_test_res = hcat(get_res(dqn_first_paths_test)...)
    means_dqn_first_test_res = vec(mean(dqn_first_test_res, dims=2))
    stds_dqn_first_test_res = vec(std(dqn_first_test_res, dims=2))
    # plot!(1:length(stds_dqn_first_test_res), means_dqn_first_test_res, ribbon=stds_dqn_first_test_res, label="DQN 1 Tree Test Mean ± Std", lw=2, xlabel="Epochs", ylabel="Mean simplification")
    plot!(1:length(stds_dqn_first_test_res), means_dqn_first_test_res, label="DQN 1 Tree Test Mean", lw=2, xlabel="Epochs", ylabel="Mean simplification")
    
    # DQN second on Tree Training, Testing
    dqn_second_paths = ["stats/dqn_first_4th_20ep/results_of_trained_DQN_second_DAG1_not_boosted_ep\$(ep)_hidden64.csv",
    "stats/dqn_first_3rd_20ep/results_of_trained_DQN_second_DAG1_not_boosted_ep\$(ep)_hidden64.csv",
    "stats/dqn_first_2nd_20ep/results_of_trained_DQN_second_DAG1_ep\$(ep)_hidden64.csv"]
    dqn_second_res = hcat(get_res(dqn_second_paths)...)
    means_dqn_second_res = vec(mean(dqn_second_res, dims=2))
    stds_dqn_second_res = vec(std(dqn_second_res, dims=2))
    # plot!(1:length(stds_dqn_second_res), means_dqn_second_res, ribbon=stds_dqn_second_res, label="DQN 2 Tree Mean ± Std", lw=2, xlabel="Epochs", ylabel="Mean simplification")
    # plot!(1:length(stds_dqn_second_res), means_dqn_second_res, label="DQN 2 Tree Mean", lw=2, xlabel="Epochs", ylabel="Mean simplification")

    dqn_second_paths_test = ["stats/dqn_first_4th_20ep/results_of_test_trained_DQN_second_DAG1_not_boosted_ep\$(ep)_hidden64.csv",
    "stats/dqn_first_3rd_20ep/results_of_test_trained_DQN_second_DAG1_not_boosted_ep\$(ep)_hidden64.csv",
    "stats/dqn_first_2nd_20ep/results_of_test_trained_DQN_second_DAG1_ep\$(ep)_hidden64.csv"]
    dqn_second_test_res = hcat(get_res(dqn_second_paths_test)...)
    means_dqn_second_test_res = vec(mean(dqn_second_test_res, dims=2))
    stds_dqn_second_test_res = vec(std(dqn_second_test_res, dims=2))
    # plot!(1:length(stds_dqn_second_test_res), means_dqn_second_test_res, ribbon=stds_dqn_second_test_res, label="DQN 2 Tree Test Mean ± Std", lw=2, xlabel="Epochs", ylabel="Mean simplification")
    plot!(1:length(stds_dqn_second_test_res), means_dqn_second_test_res, label="DQN 2 Tree Test Mean", lw=2, xlabel="Epochs", ylabel="Mean simplification")

    # DQN first on DG Training, Testing
    dqn_first_dg_paths = ["stats/dqn_first_2nd_20ep/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
    "stats/dqn_first_3rd_20ep/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
    "stats/dqn_first_4th_20ep/results_of_trained_DQN_first_DG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
    ]
    dqn_first_dg_res = hcat(get_res(dqn_first_dg_paths)...)
    means_dqn_first_dg_res = vec(mean(dqn_first_dg_res, dims=2))
    stds_dqn_first_dg_res = vec(std(dqn_first_dg_res, dims=2))
    # plot!(1:length(stds_dqn_first_dg_res), means_dqn_first_dg_res, ribbon=stds_dqn_first_dg_res, label="DQN 1 DG Mean ± Std", lw=2, xlabel="Epochs", ylabel="Mean simplification")
    # plot!(1:length(stds_dqn_first_dg_res), means_dqn_first_dg_res, label="DQN 1 DG Mean", lw=2, xlabel="Epochs", ylabel="Mean simplification")


    dqn_first_dg_paths_test = ["stats/dqn_first_2nd_20ep/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
    "stats/dqn_first_3rd_20ep/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
    "stats/dqn_first_4th_20ep/results_of_trained_test_DQN_first_DG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
    ]
    dqn_first_dg_test_res = hcat(get_res(dqn_first_dg_paths_test)...)
    means_dqn_first_dg_test_res = vec(mean(dqn_first_dg_test_res, dims=2))
    stds_dqn_first_dg_test_res = vec(std(dqn_first_dg_test_res, dims=2))
    # plot!(1:length(stds_dqn_first_dg_test_res), means_dqn_first_dg_test_res, ribbon=stds_dqn_first_dg_test_res, label="DQN 1 DG Test Mean ± Std", lw=2, xlabel="Epochs", ylabel="Mean simplification")
    plot!(1:length(stds_dqn_first_dg_test_res), means_dqn_first_dg_test_res, label="DQN 1 DG Test Mean", lw=2, xlabel="Epochs", ylabel="Mean simplification")

    # DQN first on DAG Training, Testing
    dqn_first_dag_paths = ["stats/dqn_first_2nd_20ep/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
    "stats/dqn_first_3rd_20ep/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
    "stats/dqn_first_4th_20ep/results_of_trained_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
    ]
    dqn_first_dag_res = hcat(get_res(dqn_first_dag_paths)...)
    means_dqn_first_dag_res = vec(mean(dqn_first_dag_res, dims=2))
    stds_dqn_first_dag_res = vec(std(dqn_first_dag_res, dims=2))
    # plot!(1:length(stds_dqn_first_dag_res), means_dqn_first_dag_res, ribbon=stds_dqn_first_dag_res, label="DQN 1 DAG Mean ± Std", lw=2, xlabel="Epochs", ylabel="Mean simplification")
    # plot!(1:length(stds_dqn_first_dag_res), means_dqn_first_dag_res, label="DQN 1 DAG Mean", lw=2, xlabel="Epochs", ylabel="Mean simplification")

    dqn_first_dag_paths_test = ["stats/dqn_first_2nd_20ep/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
    "stats/dqn_first_3rd_20ep/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
    "stats/dqn_first_4th_20ep/results_of_trained_test_DQN_first_DAG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
    ]
    dqn_first_dag_test_res = hcat(get_res(dqn_first_dag_paths_test)...)
    means_dqn_first_dag_test_res = vec(mean(dqn_first_dag_test_res, dims=2))
    stds_dqn_first_dag_test_res = vec(std(dqn_first_dag_test_res, dims=2))
    # plot!(1:length(stds_dqn_first_dag_test_res), means_dqn_first_dag_test_res, ribbon=stds_dqn_first_dag_test_res, label="DQN 1 DAG Test Mean ± Std", lw=2, xlabel="Epochs", ylabel="Mean simplification")
    plot!(1:length(stds_dqn_first_dag_test_res), means_dqn_first_dag_test_res, label="DQN 1 DAG Test Mean", lw=2, xlabel="Epochs", ylabel="Mean simplification")

    # DQN second on DG Training, Testing
    dqn_second_dg_paths = ["stats/dqn_first_2nd_20ep/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
    "stats/dqn_first_3rd_20ep/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
    "stats/dqn_first_4th_20ep/results_of_trained_DQN_second_DG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
    ]
    dqn_second_dg_res = hcat(get_res(dqn_second_dg_paths)...)
    means_dqn_second_dg_res = vec(mean(dqn_second_dg_res, dims=2))
    stds_dqn_second_dg_res = vec(std(dqn_second_dg_res, dims=2))
    # plot!(1:length(stds_dqn_second_dg_res), means_dqn_second_dg_res, ribbon=stds_dqn_second_dg_res, label="DQN 2 DG Mean ± Std", lw=2, xlabel="Epochs", ylabel="Mean simplification")
    # plot!(1:length(stds_dqn_second_dg_res), means_dqn_second_dg_res, label="DQN 2 DG Mean", lw=2, xlabel="Epochs", ylabel="Mean simplification")

    dqn_second_dg_paths_test = ["stats/dqn_first_2nd_20ep/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
    "stats/dqn_first_3rd_20ep/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
    "stats/dqn_first_4th_20ep/results_of_trained_test_DQN_second_DG_not_boosted_ep\$(ep)_batch128_hidden64.csv",
    ]
    dqn_second_dg_test_res = hcat(get_res(dqn_second_dg_paths_test)...)
    means_dqn_second_dg_test_res = vec(mean(dqn_second_dg_test_res, dims=2))
    stds_dqn_second_dg_test_res = vec(std(dqn_second_dg_test_res, dims=2))
    # plot!(1:length(stds_dqn_second_dg_test_res), means_dqn_second_dg_test_res, ribbon=stds_dqn_second_dg_test_res, label="DQN 2 DG Test Mean ± Std", lw=2, xlabel="Epochs", ylabel="Mean simplification")
    plot!(1:length(stds_dqn_second_dg_test_res), means_dqn_second_dg_test_res, label="DQN 2 DG Test Mean", lw=2, xlabel="Epochs", ylabel="Mean simplification")


    # 2 Values Training, Testing
    values2_paths = ["stats/dqn_first_4th_20ep/results_of_trained_2Values_not_boosted_ep\$(ep)_hidden64.csv",
    "stats/dqn_first_3rd_20ep/results_of_trained_2Values_not_boosted_ep\$(ep)_hidden64.csv",
    "stats/dqn_first_2nd_20ep/results_of_trained_2Values_not_boosted_ep\$(ep)_hidden64.csv"]
    values2_res = hcat(get_res(values2_paths)...)
    means_values2_res = vec(mean(values2_res, dims=2))
    stds_values2_res = vec(std(values2_res, dims=2))
    plot!(1:length(stds_values2_res), means_values2_res, ribbon=stds_values2_res, label="2 Values Mean ± Std", lw=2, xlabel="Epochs", ylabel="Mean simplification")

    values2_paths_test = ["stats/dqn_first_4th_20ep/results_of_test_trained_2Values_not_boosted_ep\$(ep)_hidden64.csv",
    "stats/dqn_first_3rd_20ep/results_of_test_trained_2Values_not_boosted_ep\$(ep)_hidden64.csv",
    "stats/dqn_first_2nd_20ep/results_of_test_trained_2Values_not_boosted_ep\$(ep)_hidden64.csv"]
    values2_test_res = hcat(get_res(values2_paths_test)...)
    means_values2_test_res = vec(mean(values2_test_res, dims=2))
    stds_values2_test_res = vec(std(values2_test_res, dims=2))
    plot!(1:length(stds_values2_test_res), means_values2_test_res, ribbon=stds_values2_test_res, label="2 Values Test Mean ± Std", lw=2, xlabel="Epochs", ylabel="Mean simplification")

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
end