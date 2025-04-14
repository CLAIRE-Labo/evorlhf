import numpy as np
import wandb
import pandas as pd
import matplotlib.pyplot as plt
api = wandb.Api()

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 2,
    'figure.figsize': (8, 6)
})

entity, project = "claire-labo", "pack"
task = "bin"
num_models = 1
sweep_specifier = "icml"
max_round_num = 2700

# Define legend name mapping
legend_name_map = {
    "llama32-b": "Llama32 - FunSearch",
    "llama32-s2": "Llama32 - EvoTune",
    "qwensmall-b": "Qwen - FunSearch",
    "qwensmall-s0": "Qwen - EvoTune",
    "granite-b": "Granite - FunSearch",
    "granite-s2": "Granite - EvoTune",
}

results = {"llama32":{"b":[], "s2":[]}, "qwensmall":{"b":[], "s0":[]}, "granite":{"b":[], "s2":[]}}#, "phi":{"b":[], "s2":[]}}


for model_name in results.keys():
    for sweep_identifier in results[model_name].keys():
        #model_name = "llama32"
        #sweep_identifier = "b" # "b", "s0", "s1", "s2""
        filters = {"config.config_specific": f"{num_models}m{model_name}{sweep_specifier}{sweep_identifier}"
                    , "config.task_name": task, "config.prefix": "sweepfinalicml"} # "config.config_specific": "1llama32icmlb"
        runs = api.runs(entity + "/" + project, filters=filters)

        seeds = []
        for run in runs:
            print(run.name)
            seeds.append(run.config["seed"])
        seeds = list(set(seeds))


        histories = {}.fromkeys(seeds)
        histories = {seed: [] for seed in seeds}
        keys = ['Number of unique scores in program bank', "round_num"]
        for run in runs:
            # history = run.scan_history(keys=keys)
            history = run.history(keys=keys, pandas=True, samples=100, x_axis='round_num')
            s = run.config["seed"]
            histories[s].append(history)

        for seed in seeds:
            histories[seed] = [history for history in histories[seed] if not history.empty]

        for seed in seeds:
            if len(histories[seed]) != 1:    
                for i in range(1, len(histories[seed])):
                    histories[seed][i] = histories[seed][i].iloc[1:] if histories[seed][i].iloc[0]["round_num"] == histories[seed][i-1].iloc[-1]["round_num"] else histories[seed][i]
            histories[seed] = pd.concat(histories[seed])

        metrics = {}.fromkeys(keys)
        metrics = {key: [] for key in keys}

        for seed, history_df in histories.items():
            for key in keys:
                metrics[key].append(history_df[key].tolist())


        # y,x = keys[0], keys[1]
        # plt.figure()
        # for seed in seeds:
        #     plt.plot(metrics[x][seed], metrics[y][seed], label=f"seed {seed}")
        #     plt.title(f"{num_models}m{model_name}{sweep_specifier}{sweep_identifier}")
        #     plt.legend()
        #     plt.xlabel(x)
        #     plt.ylabel(y)

        # #plt.show()
        # plt.savefig(f"plots/{num_models}m{model_name}{sweep_specifier}{sweep_identifier}_{task}.png")

        results[model_name][sweep_identifier] = metrics


# Average metrics over seeds and plot all in one figure
plt.figure(figsize=(10, 6))
averaged_metrics = {}

for model_name in results.keys():
    for sweep_identifier, metrics in results[model_name].items():
        round_nums = metrics["round_num"][0]
        unique_scores = metrics["Number of unique scores in program bank"]

        # Truncate based on max_round_num and exclude round_num == -1
        truncated_rounds = [rn for rn in round_nums if rn <= max_round_num ]
        truncated_unique_scores = [
            [score for rn, score in zip(metrics["round_num"][seed], unique_scores[seed]) if rn <= max_round_num]
            for seed in range(len(metrics["round_num"]))
        ]
        
        # Calculate average and standard error
        averaged = np.mean(truncated_unique_scores, axis=0)
        std_error = np.std(truncated_unique_scores, axis=0) / np.sqrt(len(truncated_unique_scores))
        
        # To fix the issue of the first round being -1
        truncated_rounds = truncated_rounds[1:]
        truncated_rounds.extend([truncated_rounds[-1] + 100])

        averaged_metrics[f"{model_name}-{sweep_identifier}"] = {
            "round_num": truncated_rounds,
            "unique_scores": averaged,
            "std_error": std_error
        }


# Define color map for models
color_map = {
    "llama32": "blue",
    "qwensmall": "green",
    "granite": "red",
    "phi": "purple"
}

# Define marker styles for sweep identifiers
# Define line styles for sweep identifiers
line_style_map = {
    "b": "solid",
    "s0": "dashed",
    "s2": "dashdot"
}

# for label, data in averaged_metrics.items():
#     plt.plot(data["round_num"], data["unique_scores"], label=label)
# plt.figure(figsize=(10, 6))
# for label, data in averaged_metrics.items():
#     model, identifier = label.split('-')
#     plt.plot(
#         data["round_num"],
#         data["unique_scores"],
#         label=legend_name_map.get(label, label),
#         color=color_map[model],
#         linestyle=line_style_map.get(identifier, "solid")
#     )
#     plt.fill_between(
#         data["round_num"],
#         data["unique_scores"] - data["std_error"],
#         data["unique_scores"] + std_error,
#         color=color_map[model],
#         alpha=0.2
#     )

# plt.xlabel("Timestep")
# plt.ylabel("Number of unique scores in program bank")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("plots/aggregated_plot.png", dpi=300)


fig, ax = plt.subplots(figsize=(10, 6))

# Define refined color and marker styles
color_map = {
    "llama32": "#1f77b4",  # Blue
    "qwensmall": "#2ca02c",  # Green
    "granite": "#d62728",  # Red
}

marker_map = {
    "b": "o",  # Circle
    "s0": "s",  # Square
    "s2": "d",  # Diamond
}

line_style_map = {
    "b": "solid",
    "s0": "dashed",
    "s2": "dashdot",
}

for label, data in averaged_metrics.items():
    model, identifier = label.split('-')
    plt.plot(
        data["round_num"],
        data["unique_scores"],
        label=legend_name_map.get(label, label),
        color=color_map[model],
        linestyle=line_style_map.get(identifier, "solid")
    )
    plt.fill_between(
        data["round_num"],
        data["unique_scores"] - data["std_error"],
        data["unique_scores"] + std_error,
        color=color_map[model],
        alpha=0.2
    )
# Labels and title
ax.set_xlabel("Timestep", fontsize=16, fontweight="bold")
ax.set_ylabel("Unique Scores in Program Bank", fontsize=16, fontweight="bold")

# Legend positioning outside the plot for clarity
ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=14)

# Improve ticks and grid
ax.tick_params(axis="both", which="major", labelsize=14)
ax.grid(True, linestyle="--", alpha=0.6)

# Tight layout to avoid overlap
plt.tight_layout()

# Save improved plot
plt.savefig("plots/aggregated_plot_improved.png", dpi=300, bbox_inches="tight")

# Display plot
plt.show()
