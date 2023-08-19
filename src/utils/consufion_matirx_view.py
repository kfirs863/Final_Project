from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import json


# Convert to dictionary
confusion_matrix_data = json.loads(Path('/homes/kfirs/PycharmProjects/FinalProject/plots/confusion-matrix.json').read_text())

# Extract the matrix and labels
matrix = confusion_matrix_data['matrix']
labels = confusion_matrix_data['labels']

# Create a figure and axes with larger size
plt.figure(figsize=(40, 40))

# Use Seaborn's heatmap function to plot the confusion matrix with the "Blues" color map
sns.heatmap(matrix, annot=False, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=True)

plt.xlabel(confusion_matrix_data['columnLabel'], fontsize=16)
plt.ylabel(confusion_matrix_data['rowLabel'], fontsize=16)
plt.title(confusion_matrix_data['title'], fontsize=20)

# Save the plot as an image file with higher DPI
plt.savefig('/homes/kfirs/PycharmProjects/FinalProject/plots/confusion_matrix.png', dpi=600, bbox_inches='tight')

# Optionally, show the plot
plt.show()
