from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np


# Convert to dictionary
confusion_matrix_data = json.loads(Path('/homes/kfirs/PycharmProjects/FinalProject/plots/dev_confusion-matrix.json').read_text())

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
plt.savefig('/homes/kfirs/PycharmProjects/FinalProject/plots/dev_confusion_matrix.png', dpi=600, bbox_inches='tight')


# Convert the matrix to a NumPy array for easier manipulation
matrix_np = np.array(matrix)

# Calculate the total number of predictions
total_predictions = np.sum(matrix_np)

# Calculate the total number of correct predictions (sum of diagonal elements)
total_correct_predictions = np.trace(matrix_np)

print(f"Total Predictions: {total_predictions}")
print(f"Total Correct Predictions: {total_correct_predictions}")


# Assuming matrix is your confusion matrix as a list of lists
matrix_np = np.array(matrix)

# Sum along the columns to get the total predictions for each class
total_predictions_per_class = np.sum(matrix_np, axis=0)

# Find the index of the class with the most predictions
class_with_most_predictions = np.argmax(total_predictions_per_class)

# If you want to know the label of that class
label_with_most_predictions = labels[class_with_most_predictions]

# Print the results
print(f"Class with most predictions: {class_with_most_predictions}")
print(f"Label with most predictions: {label_with_most_predictions}")
print(f"Number of predictions for this class: {total_predictions_per_class[class_with_most_predictions]}")

# Optionally, show the plot
plt.show()
