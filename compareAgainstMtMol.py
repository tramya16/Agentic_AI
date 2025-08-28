import matplotlib.pyplot as plt
import numpy as np
from math import pi
from pathlib import Path

# Data from your table
tasks = [
    'albuterol_similarity', 'amlodipine_mpo', 'celecoxib_rediscovery', 'deco_hop',
    'drd2_binding', 'fexofenadine_mpo', 'gsk3b_activity', 'isomers_c7h8n2o2',
    'isomers_c9h10n2o2pf2cl', 'jnk3_inhibition', 'median1_similarity', 'median2_similarity',
    'mestranol_similarity', 'osimertinib_mpo', 'perindopril_mpo', 'qed_optimization',
    'ranolazine_mpo', 'scaffold_hop', 'sitagliptin_mpo', 'thiothixene_rediscovery',
    'troglitazone_rediscovery', 'valsartan_smarts', 'zaleplon_similarity'
]

your_scores = [
    0.9863, 0.4033, 0.7604, 0.5923, 0.3183, 0.7427, 0.1603, 0.9723,
    0.4315, 0.0883, 0.2999, 0.3178, 0.9974, 0.2898, 0.4486, 0.6949,
    0.0704, 0.7096, 0.0505, 0.7850, 0.7876, 0.0001, 0.3771
]

mt_mol_scores = [
    0.998, 0.647, 0.867, 0.842, 0.756, 0.883, 0.308, 0.986,
    0.914, 0.125, 0.321, 0.322, 0.996, 0.796, 0.542, 0.903,
    0.233, 0.646, 0.067, 0.719, 0.841, 0.000, 0.625
]

# Create figure with larger size for better readability
fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(projection='polar'))

# Number of variables
N = len(tasks)

# Compute angle for each axis
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]  # Complete the circle

# Add values to complete the circle
your_scores += your_scores[:1]
mt_mol_scores += mt_mol_scores[:1]

# Plot
ax.plot(angles, your_scores, 'o-', linewidth=2, label='Our System', color='#1f77b4', markersize=6)
ax.fill(angles, your_scores, alpha=0.25, color='#1f77b4')

ax.plot(angles, mt_mol_scores, 'o-', linewidth=2, label='MT-MOL', color='#ff7f0e', markersize=6)
ax.fill(angles, mt_mol_scores, alpha=0.25, color='#ff7f0e')

# Add labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(tasks, fontsize=10)

# Set y-axis limits and labels
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
ax.grid(True)

# Add title and legend
plt.title('Performance Comparison: Our System vs MT-MOL\nAcross All Tasks',
          size=16, fontweight='bold', pad=30)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=12)

# Rotate labels for better readability
for angle, label in zip(angles, ax.get_xticklabels()):
    if angle < pi / 2 or angle > 3 * pi / 2:
        label.set_horizontalalignment('left')
    else:
        label.set_horizontalalignment('right')

    # Rotate labels
    rotation = np.degrees(angle)
    if angle > pi / 2 and angle < 3 * pi / 2:
        rotation = rotation + 180
    label.set_rotation(rotation - 90)

plt.tight_layout()

# Create output directory
output_dir = Path("results/final_visualizations")
output_dir.mkdir(exist_ok=True, parents=True)

# Save the chart as PNG
output_path = output_dir / 'performance_comparison_radar_chart.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"Chart saved as '{output_path}'")

# Don't show the plot (comment out if you want to see it)
# plt.show()

# Close the figure to free memory
plt.close()

# Also create a summary statistics
print("\nPerformance Summary:")
print(f"Your System - Average: {np.mean(your_scores[:-1]):.4f}")
print(f"MT-MOL - Average: {np.mean(mt_mol_scores[:-1]):.4f}")
print(
    f"Tasks where you outperform MT-MOL: {sum(1 for i in range(len(your_scores) - 1) if your_scores[i] > mt_mol_scores[i])}")
print(
    f"Tasks where MT-MOL outperforms you: {sum(1 for i in range(len(your_scores) - 1) if mt_mol_scores[i] > your_scores[i])}")