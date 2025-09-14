import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Define a dictionary for colors, line styles, and line widths for each method
style_dict = {
    "Random selection": {"color": "blue", "linestyle": "--", "linewidth": 1.5},
    "Active selection": {"color": "orange", "linestyle": "-.", "linewidth": 1.5},  # Orange with dash-dot line for Active Selection
    "Power of Choice": {"color": "cyan", "linestyle": "--", "linewidth": 1.5},
    "Greedy": {"color": "magenta", "linestyle": (0, (3, 5, 1, 5)), "linewidth": 1.5},
    "Resource aware": {"color": "red", "linestyle": ":", "linewidth": 1.5},
    "Price First": {"color": "green", "linestyle": "-.", "linewidth": 1.5},
    "FedPROM": {"color": "purple", "linestyle": "-", "linewidth": 1.5}  # Thicker solid line for FedPROM
}

# List of method names
methods = ["Random selection", "Active selection", "Power of Choice", "Greedy", "Resource aware", "Price First", "FedPROM"]

# Function to load data from a pickle file
def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# Define the directories for the runs
run_dirs = []

# List of file names
file_names = []

# Initialize a dictionary to hold all accuracy data
all_accuracies = {method: [] for method in methods}

# Loop through each run directory
for method, file_name in zip(methods, file_names):
    accuracies_lists = []  # This will store all runs' accuracies for the current method

    for run_dir in run_dirs:
        file_path = os.path.join(run_dir, file_name)  # Adjust the subdirectory if needed
        try:
            data = load_data(file_path)
            accuracies_lists.append(data["accuracies"])
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            continue

    # Calculate the mean accuracy across runs if data was loaded successfully
    if accuracies_lists:
        # Assuming all runs are of the same length, if not, handle accordingly
        mean_accuracies = np.mean(accuracies_lists, axis=0)
        all_accuracies[method] = [0] + list(mean_accuracies)

plt.figure(figsize=(4, 3))
for method in methods:
    plt.plot(all_accuracies[method], label=method, color=style_dict[method]["color"], 
              linestyle=style_dict[method]["linestyle"], linewidth=style_dict[method]["linewidth"])

plt.xlabel("Communication Rounds", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=14)  # Updated y-axis label
plt.xticks(fontsize=12)  # Increase x-axis tick font size
plt.yticks(fontsize=12)
plt.xlim(0, 50)  # Set x-axis limits to start at 0 and end at 30
plt.ylim(0,)
# plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/plots/image.png")
plt.show()
