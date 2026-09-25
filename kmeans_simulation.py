import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math


# STEP 1: CREATE 100 RECORDS (DATASET) AND SAVE TO data/ FOLDER

# We use seed 42 so the random numbers are always the exact same
np.random.seed(42)

# Generate 100 tracks:
danceability = np.concatenate([
    np.random.normal(loc=0.30, scale=0.08, size=30),
    np.random.normal(loc=0.75, scale=0.08, size=35),
    np.random.normal(loc=0.65, scale=0.08, size=35)
])

energy = np.concatenate([
    np.random.normal(loc=0.25, scale=0.08, size=30),
    np.random.normal(loc=0.45, scale=0.08, size=35),
    np.random.normal(loc=0.82, scale=0.07, size=35)
])

# Keep values inside realistic range [0.05, 0.95]
danceability = np.clip(danceability, 0.05, 0.95)
energy = np.clip(energy, 0.05, 0.95)

# Create a DataFrame with Track_ID, Danceability, Energy
track_ids = [f"Track_{i+1:03d}" for i in range(100)]
data = pd.DataFrame({
    'Track_ID': track_ids,
    'Danceability': np.round(danceability, 4),
    'Energy': np.round(energy, 4)
})

# Ensure the data folder exists and save CSV
os.makedirs("data", exist_ok=True)
data.to_csv("data/spotify_100_tracks.csv", index=False)
print("-> Dataset created: data/spotify_100_tracks.csv")

# Ensure the static folder exists for the plots
os.makedirs("static", exist_ok=True)


# STEP 2: DEFINE 3 INITIAL CENTROIDS (FARTHER AWAY TO SHOW CONVERGENCE)

# Colocamos centroides iniciales distantes para forzar la reasignación en las iteraciones
c1 = [0.10, 0.10]  # Esquina inferior izquierda
c2 = [0.50, 0.50]  # Centro absoluto
c3 = [0.90, 0.90]  # Esquina superior derecha

centroids = [c1, c2, c3]

print("\n--- INITIAL CENTROIDS ---")
print(f"Centroid 1: {centroids[0]}")
print(f"Centroid 2: {centroids[1]}")
print(f"Centroid 3: {centroids[2]}")

# Plot Iteration 0: Unassigned tracks and initial centroids
plt.figure(figsize=(7, 5))
plt.scatter(data['Danceability'], data['Energy'], color='gray', alpha=0.6, label='Tracks (Unassigned)')
plt.scatter([c[0] for c in centroids], [c[1] for c in centroids], color=['red', 'blue', 'green'], 
            marker='X', s=200, edgecolors='black', label='Initial Centroids')

# Add text labels for centroids
for i, c in enumerate(centroids):
    plt.text(c[0] + 0.02, c[1] + 0.02, f"C{i+1}", fontsize=11, fontweight='bold')

plt.title("Iteration 0: Initial Centroids and 100 Spotify Tracks")
plt.xlabel("Danceability (0 to 1)")
plt.ylabel("Energy (0 to 1)")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("static/iteration_0_initial.png")
plt.close()
print("-> Initial plot saved: static/iteration_0_initial.png")



# STEP 3: PERFORM 3 MANUAL ITERATIONS

colors = ['red', 'blue', 'green']
variance_history = []

for iteration in range(1, 4):
    print(f"\n==================== ITERATION {iteration} ====================")
    
    dist_c1_list = []
    dist_c2_list = []
    dist_c3_list = []
    assigned_clusters = []
    
    # 3.1 Calculate Euclidean distance for each of the 100 tracks
    # Formula: d = sqrt( (x - cx)^2 + (y - cy)^2 )
    for index, row in data.iterrows():
        x = row['Danceability']
        y = row['Energy']
        
        d1 = math.sqrt((x - centroids[0][0])**2 + (y - centroids[0][1])**2)
        d2 = math.sqrt((x - centroids[1][0])**2 + (y - centroids[1][1])**2)
        d3 = math.sqrt((x - centroids[2][0])**2 + (y - centroids[2][1])**2)
        
        dist_c1_list.append(round(d1, 4))
        dist_c2_list.append(round(d2, 4))
        dist_c3_list.append(round(d3, 4))
        
        # 3.2 Assign track to closest centroid
        if d1 <= d2 and d1 <= d3:
            assigned_clusters.append(1)
        elif d2 <= d1 and d2 <= d3:
            assigned_clusters.append(2)
        else:
            assigned_clusters.append(3)
            
    # Save the table for this iteration
    iter_table = data.copy()
    iter_table['Distance_C1'] = dist_c1_list
    iter_table['Distance_C2'] = dist_c2_list
    iter_table['Distance_C3'] = dist_c3_list
    iter_table['Assigned_Cluster'] = [f"Cluster {c}" for c in assigned_clusters]
    iter_table.to_csv(f"data/iteration_{iteration}_table.csv", index=False)
    print(f"-> Table saved: data/iteration_{iteration}_table.csv")
    
    # 3.3 Recalculate centroids (average coordinates of assigned tracks)
    new_centroids = []
    total_variance = 0.0
    
    for k in [1, 2, 3]:
        # Filter tracks belonging to cluster k
        cluster_x = [data.loc[i, 'Danceability'] for i in range(100) if assigned_clusters[i] == k]
        cluster_y = [data.loc[i, 'Energy'] for i in range(100) if assigned_clusters[i] == k]
        count = len(cluster_x)
        
        if count > 0:
            # Average (Mean)
            new_x = sum(cluster_x) / count
            new_y = sum(cluster_y) / count
            new_centroids.append([round(new_x, 4), round(new_y, 4)])
            
            # Within-cluster variance: sum of squared distances to the new centroid
            cluster_variance = sum((x - new_x)**2 + (y - new_y)**2 for x, y in zip(cluster_x, cluster_y))
            total_variance += cluster_variance
            
            print(f"Cluster {k}: {count} tracks | Old Centroid: {centroids[k-1]} -> New Centroid: [{new_x:.4f}, {new_y:.4f}]")
        else:
            new_centroids.append(centroids[k-1])
            
    variance_history.append(round(total_variance, 4))
    
    # 3.4 Create scatter plot with cluster colors and updated centroids
    plt.figure(figsize=(7, 5))
    for k in [1, 2, 3]:
        k_x = [data.loc[i, 'Danceability'] for i in range(100) if assigned_clusters[i] == k]
        k_y = [data.loc[i, 'Energy'] for i in range(100) if assigned_clusters[i] == k]
        plt.scatter(k_x, k_y, color=colors[k-1], alpha=0.6, label=f"Cluster {k} (n={len(k_x)})")
        
    # Plot new centroids
    plt.scatter([c[0] for c in new_centroids], [c[1] for c in new_centroids], 
                color=colors, marker='X', s=200, edgecolors='black', label='Updated Centroids')
    
    for i, c in enumerate(new_centroids):
        plt.text(c[0] + 0.02, c[1] + 0.02, f"C{i+1} ({c[0]:.2f}, {c[1]:.2f})", fontsize=10, fontweight='bold')
        
    plt.title(f"Iteration {iteration}: Cluster Assignments and New Centroids")
    plt.xlabel("Danceability")
    plt.ylabel("Energy")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"static/iteration_{iteration}_plot.png")
    plt.close()
    print(f"-> Plot saved: static/iteration_{iteration}_plot.png")
    
    # Update centroids for the next iteration
    centroids = new_centroids



# STEP 4: WITHIN-CLUSTER VARIANCE COMPARISON

variance_df = pd.DataFrame({
    'Iteration': ['Iteration 1', 'Iteration 2', 'Iteration 3'],
    'Within_Cluster_Variance': variance_history
})
variance_df.to_csv("data/variance_comparison.csv", index=False)

print("\n--- VARIANCE COMPARISON ACROSS ITERATIONS ---")
print(variance_df.to_string(index=False))

# Plot variance reduction trend
plt.figure(figsize=(6, 4))
plt.plot([1, 2, 3], variance_history, marker='o', color='purple', linewidth=2)
for i, val in enumerate(variance_history):
    plt.text(i + 1, val + 0.02, f"{val:.4f}", ha='center', fontweight='bold')
    
plt.title("Within-Cluster Variance (Inertia) Reduction")
plt.xlabel("Iteration Number")
plt.ylabel("Variance (WCSS)")
plt.xticks([1, 2, 3])
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("static/variance_trend.png")
plt.close()
print("-> Variance plot saved: static/variance_trend.png")
print("\n=== SUCCESS: All 100 records, 3 iterations, tables and plots are ready! ===")