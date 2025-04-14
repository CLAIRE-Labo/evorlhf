import json
import os
import numpy as np


def main(logs_dir, prefix, task, model_name, num_models, sweep_specifier, round_num):
    sweep_identifiers = set()
    seeds = set()
    for folder in os.listdir(f"{logs_dir}/{prefix}"):
        if not os.path.isdir(f"{logs_dir}/{prefix}/{folder}"):
            continue
        if sweep_specifier == "":
            if f"{task}_task{task}_{num_models}m{model_name}" in folder:
                seeds.add(folder.split("_")[-1])
                sweep_identifier = folder.split("_")[-2]
                sweep_identifiers.add(sweep_identifier.split(model_name)[-1])
        else:
            if f"{task}_task{task}_{num_models}m{model_name}{sweep_specifier}" in folder:
                seeds.add(folder.split("_")[-1])
                sweep_identifier = folder.split("_")[-2]
                sweep_identifiers.add(sweep_identifier.split(sweep_specifier)[-1])

    for sweep in sweep_identifiers:
        if sweep != "s2" and sweep != "b":
            continue
        sweep_scores_avg_top_50 = []
        sweep_scores_avg_top_10 = []
        sweep_scores_num_unique = []
        sweep_scores_best_overall = []
        for seed in seeds:
            name = f"{task}_task{task}_{num_models}m{model_name}{sweep_specifier}{sweep}_{seed}" if sweep_specifier != "" else f"{task}_task{task}_{num_models}m{model_name}{sweep}_{seed}"
            path = f"{logs_dir}/{prefix}/{name}/programdb_scores.json"
            with open(path, 'rb') as f:
                pdb_scores = json.load(f)
                assert round_num in [round_dict["round_num"] for round_dict in
                                     pdb_scores], f"round_num {int(round_num)} not available for programbank scores in {os.path.dirname(path)}"
            for round_dict in pdb_scores:
                if round_dict["round_num"] == round_num:
                    all_scores = round_dict["scores"]
                    sweep_scores_avg_top_50.append(np.mean(sorted(all_scores, reverse=True)[:50]))
                    sweep_scores_avg_top_10.append(np.mean(sorted(all_scores, reverse=True)[:10]))
                    sweep_scores_num_unique.append(len(set(all_scores)))
                    sweep_scores_best_overall.append(max(all_scores))
        print(
            f"{task}_task{task}_{num_models}m{model_name}{sweep_specifier}{sweep} across {len(sweep_scores_avg_top_50)} seeds top 50: {np.mean(sweep_scores_avg_top_50):.3f} $\pm$ {np.std(sweep_scores_avg_top_50) / np.sqrt(len(sweep_scores_avg_top_50)):.3f}")
        print(
            f"{task}_task{task}_{num_models}m{model_name}{sweep_specifier}{sweep} across {len(sweep_scores_avg_top_10)} seeds top 10: {np.mean(sweep_scores_avg_top_10):.3f} $\pm$ {np.std(sweep_scores_avg_top_10) / np.sqrt(len(sweep_scores_avg_top_10)):.3f}")
        print(
            f"{task}_task{task}_{num_models}m{model_name}{sweep_specifier}{sweep} across {len(sweep_scores_best_overall)} seeds best overall: {np.mean(sweep_scores_best_overall):.3f} $\pm$ {np.std(sweep_scores_best_overall) / np.sqrt(len(sweep_scores_best_overall)):.3f}")
        print(
            f"{task}_task{task}_{num_models}m{model_name}{sweep_specifier}{sweep} across {len(sweep_scores_num_unique)} seeds num unique scores in DB: {np.mean(sweep_scores_num_unique):.3f} $\pm$ {np.std(sweep_scores_num_unique) / np.sqrt(len(sweep_scores_num_unique)):.3f}\n")
        # print(f"{task}_task{task}_{num_models}m{model_name}{sweep_specifier}{sweep} across {len(sweep_scores_avg_top_10)} seeds top 10: {np.mean(sweep_scores_avg_top_10):.3f} +/- {np.std(sweep_scores_avg_top_10) / np.sqrt(len(sweep_scores_avg_top_10)):.3f}")

    print("\n=====================================================================================\n")
    print("Reading metrics from programdb_scores.json done\nContinuing with /metrics...\n")
    print("=====================================================================================\n")
    # if metrics folder is populated, we will compute the mean +/- stderr for the metrics for different datasets as well
    # metric files are named as metrics_{dataset_name}_round_{round}.json
    # they're stored in the same folder as the programdb_scores.json under metrics folder
    metric_keys = ["best_50_scores_avg_overall", "best_10_scores_avg_overall", "best_overall_score"]

    for sweep in sweep_identifiers:
        if sweep != "s2" and sweep != "b":
            continue
        sweep_metrics = {}
        for seed in seeds:
            name = f"{task}_task{task}_{num_models}m{model_name}{sweep_specifier}{sweep}_{seed}" if sweep_specifier != "" else f"{task}_task{task}_{num_models}m{model_name}{sweep}_{seed}"
            path = f"{logs_dir}/{prefix}/{name}/metrics"
            assert os.path.exists(path), f"metrics folder not found in {os.path.dirname(path)}"
            for metric_file in os.listdir(path):
                if metric_file.startswith("metrics_") and metric_file.endswith(f"round_{str(int(round_num))}.json"):
                    dataset_name = metric_file.split("_")[1]
                    if dataset_name not in sweep_metrics:
                        sweep_metrics[dataset_name] = {}
                    with open(f"{path}/{metric_file}", 'rb') as f:
                        metrics = json.load(f)[0]
                        assert all(metric_key in metrics for metric_key in
                                   metric_keys), f"metric keys not found in {metric_file}"
                    for metric_key in metric_keys:
                        if metric_key not in sweep_metrics[dataset_name]:
                            sweep_metrics[dataset_name][metric_key] = []
                        sweep_metrics[dataset_name][metric_key].append(metrics[metric_key])
        for dataset_name in sweep_metrics:
            for metric_key in metric_keys:
                print(
                    f"{task}_task{task}_{num_models}m{model_name}{sweep_specifier}{sweep} across {len(list(sweep_metrics[dataset_name][metric_key]))} seeds and {dataset_name} {metric_key}: {-np.mean(sweep_metrics[dataset_name][metric_key]) / 100.:.3f} $\pm$ {np.std(sweep_metrics[dataset_name][metric_key]) / 100.0 / np.sqrt(len(sweep_metrics[dataset_name][metric_key])):.3f}")
        print("\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_models", type=str, required=True)
    parser.add_argument("--sweep_specifier", type=str, required=False, default="")
    parser.add_argument("--round_num", type=str, required=True)
    args = parser.parse_args()
    logs_dir = args.logs_dir
    task = args.task
    prefix = args.prefix
    model_name = args.model_name
    num_models = args.num_models
    sweep_specifier = args.sweep_specifier
    round_num = float(args.round_num)
    main(logs_dir, prefix, task, model_name, num_models, sweep_specifier, round_num)

# how to use this file? example:
# python stderr.py --logs_dir /claire-rcp-scratch/home/smmansou/packing/run/logs --prefix sweepfinalstart --task tsp --model_name llama32 --num_models 1 --sweep_specifier start --round_num 3000
# python stderr.py --logs_dir /scratch/surina/packing/run/logs --prefix sweepfinalstart --task bin --model_name llama32 --num_models 1 --sweep_specifier start --round_num 3000
# python stderr.py --logs_dir /scratch/surina/packing/run/logs --prefix sweepfinalicml --task bin --model_name llama32 --num_models 1 --sweep_specifier icml --round_num 3100
# python stderr.py --logs_dir /claire-rcp-scratch/home/smmansou/packing/run/logs --prefix sweepfinal --task tsp --model_name llama32 --num_models 1 --round_num 3100

# For testing
# python stderr.py --logs_dir /scratch/surina/packing/run/logs --prefix sweepfinalstart --task bin --model_name llama32 --num_models 1 --sweep_specifier start --round_num 2700

# Final icml sweeps
# python stderr.py --logs_dir /scratch/surina/packing/run/logs --prefix sweepfinalicml --task bin --model_name llama32 --num_models 1 --sweep_specifier icml --round_num 2700
# python stderr.py --logs_dir /scratch/surina/packing/run/logs --prefix sweepfinalicml --task bin --model_name llama32 --num_models 1 --sweep_specifier icml --round_num 1900
# python stderr.py --logs_dir /scratch/surina/packing/run/logs --prefix sweepfinalicml --task bin --model_name llama32 --num_models 1 --sweep_specifier icml --round_num 1100

# python stderr.py --logs_dir /scratch/surina/packing/run/logs --prefix sweepfinalicml --task bin --model_name qwensmall --num_models 1 --sweep_specifier icml --round_num 2700
# python stderr.py --logs_dir /scratch/surina/packing/run/logs --prefix sweepfinalicml --task bin --model_name qwensmall --num_models 1 --sweep_specifier icml --round_num 1900
# python stderr.py --logs_dir /scratch/surina/packing/run/logs --prefix sweepfinalicml --task bin --model_name qwensmall --num_models 1 --sweep_specifier icml --round_num 1100

# python stderr.py --logs_dir /scratch/surina/packing/run/logs --prefix sweepfinalicml --task bin --model_name granite --num_models 1 --sweep_specifier icml --round_num 2700
# python stderr.py --logs_dir /scratch/surina/packing/run/logs --prefix sweepfinalicml --task bin --model_name granite --num_models 1 --sweep_specifier icml --round_num 1900
# python stderr.py --logs_dir /scratch/surina/packing/run/logs --prefix sweepfinalicml --task bin --model_name granite --num_models 1 --sweep_specifier icml --round_num 1100

# python stderr.py --logs_dir /scratch/smmansou/packing/run/logs --prefix sweepfinalicml --task tsp --model_name llama32 --num_models 1 --sweep_specifier icml --round_num 1100

# python stderr.py --logs_dir /claire-rcp-scratch/shared/packing_logs/logs --prefix sweepfinalicml --task bin --model_name llama32 --num_models 1 --sweep_specifier icml --round_num 2700

# Final rebuttal sweeps
# PYTHONPATH=src python src/plotting/stderr.py --logs_dir /iopsstor/scratch/cscs/asurina/packing/run/out/logs --prefix finalarxiv --task flatpack --model_name llama32 --num_models 1 --sweep_specifier arxiv --round_num 1100
# PYTHONPATH=src python src/plotting/stderr.py --logs_dir /iopsstor/scratch/cscs/asurina/packing/run/out/logs --prefix finalarxiv --task flatpack --model_name llama32 --num_models 1 --sweep_specifier arxiv --round_num 1900
# PYTHONPATH=src python src/plotting/stderr.py --logs_dir /iopsstor/scratch/cscs/asurina/packing/run/out/logs --prefix finalarxiv --task flatpack --model_name llama32 --num_models 1 --sweep_specifier arxiv --round_num 2700

# PYTHONPATH=src python src/plotting/stderr.py --logs_dir /iopsstor/scratch/cscs/asurina/packing/run/out/logs --prefix finalarxiv --task flatpack --model_name phi --num_models 1 --sweep_specifier arxiv --round_num 1100
# PYTHONPATH=src python src/plotting/stderr.py --logs_dir /iopsstor/scratch/cscs/asurina/packing/run/out/logs --prefix finalarxiv --task flatpack --model_name phi --num_models 1 --sweep_specifier arxiv --round_num 1900
# PYTHONPATH=src python src/plotting/stderr.py --logs_dir /iopsstor/scratch/cscs/asurina/packing/run/out/logs --prefix finalarxiv --task flatpack --model_name phi --num_models 1 --sweep_specifier arxiv --round_num 2700

# PYTHONPATH=src python src/plotting/stderr.py --logs_dir /iopsstor/scratch/cscs/asurina/packing/run/out/logs --prefix finalarxiv --task flatpack --model_name granite --num_models 1 --sweep_specifier arxiv --round_num 1100
# PYTHONPATH=src python src/plotting/stderr.py --logs_dir /iopsstor/scratch/cscs/asurina/packing/run/out/logs --prefix finalarxiv --task flatpack --model_name granite --num_models 1 --sweep_specifier arxiv --round_num 1900
# PYTHONPATH=src python src/plotting/stderr.py --logs_dir /iopsstor/scratch/cscs/asurina/packing/run/out/logs --prefix finalarxiv --task flatpack --model_name granite --num_models 1 --sweep_specifier arxiv --round_num 2700
