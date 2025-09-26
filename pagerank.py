#power iteration - page rank

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

links = {
    0: [1],     # Doc1 → Doc2
    1: [0,2],  # Doc2 → Doc1, Doc3
    2: [1]      # Doc3 → Doc2
}

n_nodes = 3
H = np.zeros((n_nodes, n_nodes), dtype=float)

for src, dests in links.items():
    for dest in dests:
        H[src][dest] = 1

print("Adjacency Matrix")
print(H)

# Step 1: If a row of H has no 1's, then replace each element by 1/N

H_step1 = H.copy()
row_sums = H_step1.sum(axis=1)
dangling = (row_sums == 0)  #boolean
H_step1[dangling] = np.ones(n_nodes) / n_nodes

# Step 2: Divide each 1 in H by the number of 1's in its row (based on original H)
H_step2 = H_step1.copy()
for i in range(n_nodes):
    num_links = np.sum(H[i])
    if num_links > 0:
        for j in range(n_nodes):
            if H[i, j] == 1:
                H_step2[i, j] = 1 / num_links

# Step 3: Multiply the resulting matrix by (1 - α)
alpha = 0.5
S = H_step2 * (1 - alpha)
print("\nAfter Step 3 (S = H * (1 - α)):")
print(S)

# Step 4: Add α/N to every entry of the resulting matrix to obtain G
G = S + (alpha / n_nodes)
print("\nAfter Step 4 (Final TPM G):")
print(G)

# Initial PageRank vector
PR = np.array([1/3, 1/3, 1/3], dtype=float)

max_iterations = 100
epsilon = 0.01  # Convergence threshold

for iteration in range(max_iterations):
    PR_new = np.dot(PR, G)  # PR^(k+1) = PR^(k) * G
    # Normalize to ensure sum is 1
    print(f"\nIteration {iteration}: {PR_new} , Sum = {np.sum(PR_new)}")

    if np.sum(np.abs(PR_new - PR)) < epsilon:
        print(f"Converged at iteration {iteration}")
        break
    PR = PR_new

PR_new /= np.sum(PR_new)

print(f"\nFinal PageRank (normalized): {PR_new}")
print(f"\n Final Sum of Page Rank = {np.sum(PR_new)}")

my_labels = {}
for i in range(n_nodes):
  my_labels[i] = f"Doc{i+1}"

# --- VISUALIZE PAGE RANK ---
def visualize_page_rank(page_rank):
    n_nodes = len(page_rank)
    nodes = []
    for i in range(n_nodes):
        nodes.append("Doc " + str(i+1))
    plt.bar(nodes, page_rank, color='skyblue')  # simple color
    plt.ylabel('PageRank Value')
    plt.title('PageRank Distribution')
    # Add values on top of bars
    for i, v in enumerate(page_rank):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')

    plt.show()

visualize_page_rank(PR_new)
# --- VISUALIZE GRAPH ---
G_nx = nx.from_numpy_array(H, create_using=nx.DiGraph)
plt.figure(figsize=(6, 4))
pos = nx.spring_layout(G_nx)
nx.draw(G_nx, pos, with_labels=True, labels=my_labels,
        node_color='skyblue', node_size=1500, arrowsize=20)

plt.title("Document Graph (from adjacency matrix)")
plt.show()


import networkx as nx

# Build graph from adjacency
links = {
    0: [1],
    1: [0, 2],
    2: [1]
}
G = nx.DiGraph(links)

# Compute PageRank
pr = nx.pagerank(G, alpha=0.85)  # damping factor
print(pr)
