import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Load the data
df = pd.read_csv('kc_house_data.csv')

# Calculate the average price and the number of houses by 'waterfront' category
grouped = df.groupby('waterfront')['price'].agg(['mean', 'count']).reset_index()

# Map the waterfront categories to closer x positions
x_map = {0: 0.45, 1: 0.55}
grouped['x_pos'] = grouped['waterfront'].map(x_map)

plt.figure(figsize=(6, 4))

# Define a scaling factor for the point sizes
size_factor = 0.15

# Create the scatter plot, using our new x positions
plt.scatter(grouped['x_pos'], grouped['mean'], 
            s=grouped['count'] * size_factor, 
            color=['blue', 'orange'], alpha=0.7)

# Annotate each point with the number of houses and the average price
for _, row in grouped.iterrows():
    plt.text(row['x_pos'], row['mean'], 
             f"n = {row['count']}\n{row['mean']:,.0f} USD", 
             ha='center', va='bottom', fontsize=10)

plt.xlabel("Waterfront (0 = No, 1 = Yes)", fontsize=12)
plt.ylabel("Average Price (USD)", fontsize=12)
plt.title("Average Price and Number of Houses by Waterfront", fontsize=14)
plt.grid(True)

# Use custom x-ticks at the positions 0.45 and 0.55, labeled as 0 and 1
plt.xticks([0.45, 0.55], ["0", "1"])

# Optionally narrow the x-axis limits to zoom in on these points
plt.xlim(0.4, 0.6)

# Create custom legend markers
non_waterfront_marker = mlines.Line2D([], [], color='blue', marker='o',
                                      linestyle='None', markersize=10,
                                      label='Non Waterfront (0)')
waterfront_marker = mlines.Line2D([], [], color='orange', marker='o',
                                  linestyle='None', markersize=10,
                                  label='Waterfront (1)')
plt.legend(handles=[non_waterfront_marker, waterfront_marker])

plt.tight_layout()
plt.show()