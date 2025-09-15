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

function process_model(paths::Vector{String})
    res = hcat(get_res(paths)...)
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
        means_test, _ = process_model(data.test)
        plot!(1:length(means_test), means_test, label="$label Test Mean", lw=2)
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

res_second_boosted = """
Ep 1:  5.01
Ep 2:  5.756
Ep 3:  6.08
Ep 4:  6.233
Ep 5:  6.358
Ep 6:  6.449
Ep 7:  6.489
Ep 8:  6.553
Ep 9:  6.594
Ep10:  6.638
Ep11:  6.657
Ep12:  6.685
Ep13:  6.718
Ep14:  6.741
Ep15:  6.771
Ep16:  6.8
Ep17:  6.812
Ep18:  6.844
Ep19:  6.857
Ep20:  6.871
"""
res1 = split(res_second_boosted, "\n")
res_second_boosted = Meta.parse.(map(x->x[2], split.(res1, ":  ")[1:end-1]))


res_first_boosted = """
Ep 1:  5.016
Ep 2:  5.753
Ep 3:  6.081
Ep 4:  6.245
Ep 5:  6.368
Ep 6:  6.473
Ep 7:  6.515
Ep 8:  6.542
Ep 9:  6.589
Ep10:  6.626
Ep11:  6.685
Ep12:  6.706
Ep13:  6.728
Ep14:  6.742
Ep15:  6.769
Ep16:  6.781
Ep17:  6.802
Ep18:  6.819
Ep19:  6.831
Ep20:  6.855
"""
res2 = split(res_first_boosted, "\n")
res_first_boosted = Meta.parse.(map(x->x[2], split.(res2, ":  ")[1:end-1]))

plot(1:length(res_first_boosted), [res_first_boosted, res_second_boosted], label=["DQN first Tree" "DQN second DG"], xlabel="Epochs", ylabel="Mean simplification")