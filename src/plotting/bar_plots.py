
# Bin packing, round 2700
train_bin_num_unique_scores = {"Llama": [985.600, 35.577], "Qwen": [ 1305.400, 36.879], "Granite": [1045.200, 37.907], "Phi": [  0.000,  0.000]}
baseline_bin_num_unique_scores = {"Llama": [ 723.500,  18.541], "Qwen": [1391.100, 50.413], "Granite": [817.100, 25.780], "Phi": [  0.000,  0.000]}

# TSP, round 2700
train_tsp_num_unique_scores = {"Llama": [  0.000,  0.000], "Qwen": [  0.000,  0.000], "Granite": [  0.000,  0.000], "Phi": [  0.000,  0.000]}
baseline_tsp_num_unique_scores = {"Llama": [  0.000,  0.000], "Qwen": [  0.000,  0.000], "Granite": [  0.000,  0.000], "Phi": [  0.000,  0.000]}

# Create a bar plot
import matplotlib.pyplot as plt
import numpy

# Bin packing
fig, ax = plt.subplots()
bar_width = 0.35
index = numpy.arange(4)
opacity = 0.8
labels = list(train_bin_num_unique_scores.keys())
train_scores = [score[0] for score in train_bin_num_unique_scores.values()]
baseline_scores = [score[0] for score in baseline_bin_num_unique_scores.values()]

# # Plotting the bars
# rects1 = ax.bar(index, train_scores, bar_width,
#                 alpha=opacity, label='EvoTune')

# rects2 = ax.bar(index + bar_width, baseline_scores, bar_width,
#                 alpha=opacity, label='FunSearch')

# # Setting labels and title
# ax.set_ylabel('Number of Unique Scores in Program Database')
# # ax.set_title('Bin Packing Unique Scores by Algorithm')
# ax.set_xticks(index + bar_width / 2)
# ax.set_xticklabels(labels)
# ax.legend()

# # Layout adjustment and display
# fig.tight_layout()
# plt.show()
# plt.savefig("bin_packing_unique_scores.png")

# Increase figure size
fig, ax = plt.subplots(figsize=(10, 6))

# Use a professional color palette
colors = ['#4C72B0', '#55A868']

# Plotting the bars with new colors
rects1 = ax.bar(index, train_scores, bar_width,
                alpha=opacity, label='EvoTune', color=colors[0])

rects2 = ax.bar(index + bar_width, baseline_scores, bar_width,
                alpha=opacity, label='FunSearch', color=colors[1])

# Setting labels and title with larger font sizes
fontsize_axis = 22
fontsize_ticks = 18
ax.set_ylabel('Number of Unique Scores', fontsize=fontsize_axis)
ax.tick_params(axis='y', labelsize=fontsize_ticks)
ax.set_xlabel('Model Name', fontsize=fontsize_axis)
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(labels, fontsize=fontsize_ticks)
ax.legend(fontsize=fontsize_ticks)

# Add grid lines
#ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)

# Layout adjustment and save with higher DPI
fig.tight_layout()
plt.savefig("bin_packing_unique_scores.png", dpi=300)
plt.savefig("bin_packing_unique_scores.pdf", dpi=300)
plt.show()

