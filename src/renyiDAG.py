import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


def plot_graph(g, red_matrix):
    # Create graph object
    G_red = nx.DiGraph()
    # Generate group labels
    red_group_labels = [f'G{i+1}' for i in range(g)]

    # Add edges to the graph based on adjacency_matrix
    for j in range(g):
        for i in range(g):
            if red_matrix[i, j] == 1 and i != j:  # Exclude self-loops
                G_red.add_edge(red_group_labels[i], red_group_labels[j])

    # Add nodes to the graph to ensure all nodes are displayed
    for label in red_group_labels:
        if label not in G_red.nodes:
            G_red.add_node(label)
    # Plot the graph with custom styling
    fig, ax = plt.subplots(figsize=(7, 7))
    pos = nx.circular_layout(G_red)

    # Define node properties
    node_size = 5000
    node_color = sns.light_palette("green", len(G_red.nodes))  # Use lighter shades from the seaborn color palette
    node_shape = "o"  # Circle shape
    node_alpha = 0.75

    # Draw the graph
    nx.draw_networkx(G_red, pos, arrows=True, node_size=node_size, node_color=node_color, node_shape=node_shape,
                     edge_color='black', width=2, connectionstyle='arc3, rad=0.25',
                     edgecolors="grey", linewidths=2, alpha=node_alpha, font_size=16,
                     font_weight='bold', ax=ax, arrowsize=20)  # Adjust arrowsize

    # Add labels with the format G1, G2, ..., Gn
    labels = {node: f'{node}' for node in G_red.nodes}
    nx.draw_networkx_labels(G_red, pos, labels, font_size=16, font_weight='bold')

    # Customize plot appearance
    ax.set(facecolor="white")
    ax.grid(False)
    ax.set_xlim([1.1 * x for x in ax.get_xlim()])
    ax.set_ylim([1.1 * y for y in ax.get_ylim()])
    plt.axis('off')
    plt.tight_layout()  # Adjust layout to make the graph fit better
    plt.show()


def reduce_causal_matrix(causal_matrix, num_groups, group_size):
    """
    Reduces a causal matrix into a smaller matrix based on the specified number of groups and group size.
ss
    Parameters:
    causal_matrix (np.ndarray): The original causal matrix.
    num_groups (int): The number of groups.
    group_size (int): The size of each group.

    Returns:
    np.ndarray: The reduced causal matrix.
    """
    n = len(causal_matrix)
    reduced_matrix = np.zeros((num_groups, num_groups), dtype=int)
    
    # Check if the number of groups and group size match the matrix size
    if num_groups * group_size != n:
        raise ValueError("The number of groups and group size do not match the matrix size.")
    
    # Iterate through each pair of groups
    for i in range(num_groups):
        for j in range(num_groups):
            if i != j:
                # Indices for the groups
                group_i_indices = range(i * group_size, (i + 1) * group_size)
                group_j_indices = range(j * group_size, (j + 1) * group_size)
                
                # Check if there's any edge from group i to group j
                if np.any(causal_matrix[np.ix_(group_i_indices, group_j_indices)]):
                    reduced_matrix[i, j] = 1
            else:
                # Indices for the group
                group_indices = range(i * group_size, (i + 1) * group_size)
                
                # Check if there's any edge within the group
                reduced_matrix[i, j] = np.any(causal_matrix[np.ix_(group_indices, group_indices)])
    
    return reduced_matrix


# Normalization (MinMax/Standard)
def normalize(data, type='minmax'):
    if type == 'std':
        return (np.array(data) - np.mean(data)) / np.std(data)
    elif type == 'minmax':
        return (np.array(data) - np.min(data)) / (np.max(data) - np.min(data))


def generate_dag_and_time_series(n, p, nonlinear_prob, timesteps, g, s):
    """
    Generates a directed Erdős–Rényi DAG, visualizes it with custom styling, and creates multivariate time series data.
    Also returns the DataFrame of the time series data and the causal matrix.

    Parameters:
    n (int): Number of nodes.
    p (float): Probability of edge creation.
    timesteps (int): Number of time steps to generate.
    
    Returns:
    pd.DataFrame: The generated time series data.
    np.ndarray: The causal matrix.
    """
    # Create an adjacency matrix for a random graph
    adjacency_matrix = np.random.rand(n, n) < p
    # Retain only the lower triangular part to ensure a DAG
    adjacency_matrix = np.tril(adjacency_matrix, -1)

    # Print the adjacency matrix
    # print("Causal Matrix (Adjacency Matrix):")
    # print(adjacency_matrix.astype(int))  # Convert boolean matrix to int for better readability

    # Generate group labels
    group_labels = [f'G{i+1}' for i in range(n)]
    # Create graph object
    G = nx.DiGraph()

    # Add edges to the graph based on adjacency_matrix
    for j in range(n):
        for i in range(n):
            if adjacency_matrix[i, j] == 1 and i != j:  # Exclude self-loops
                G.add_edge(group_labels[i], group_labels[j])

    # Add nodes to the graph to ensure all nodes are displayed
    for label in group_labels:
        if label not in G.nodes:
            G.add_node(label)

    # Determine which links are nonlinear
    nonlinear_links = {}
    for j in range(n):
        for i in range(n):
            if adjacency_matrix[i, j] == 1 and i != j:
                nonlinear_links[(i, j)] = np.random.rand() < nonlinear_prob

    # Function to apply nonlinear transformation
    def nonlinear_transform(value):
        return np.sin(value)  # Example nonlinear function

   # Initialize the time series with random Gaussian noise
    data = np.random.normal(size=(timesteps, n))
    # Add seasonality with cycles repeating every 24 samples
    seasonality = np.sin(np.linspace(0, 2 * np.pi * timesteps / 12, timesteps))[:, None] + np.cos(np.linspace(0, 3 * np.pi * timesteps / 16, timesteps))[:, None]
    # Add the seasonality to both time series
    data = data + seasonality

# ------------ Replace the source signal ----------
    # # Create a sample signal
    # # A sine wave with two frequencies (chirp-like signal)
    # t = np.linspace(0, 1, 1000, endpoint=False)  # Time vector
    # signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)

    # # Add some noise
    # signal += 0.5 * np.random.normal(size=t.shape)
# --------------------------------------------------

    # Update time series data based on DAG relationships, including autoregression on itself
    for t in range(1, timesteps):
        for i in range(n):
            # # Start with the own past value
            data[t, i] = data[t, i] + 0.33 * data[t-1, i]

            # Add contributions from the parents
            parents = list(G.predecessors(f'G{i+1}'))
            if parents:
                parent_indices = [int(p[1:]) - 1 for p in parents]
                for parent_index in parent_indices:
                    if nonlinear_links[(parent_index, i)]:
                        # print(f'Var: {i} is nonlinear parents: {parent_index}')
                        data[t, i] += nonlinear_transform(data[t-1, parent_index])
                    else:
                        data[t, i] += data[t-1, parent_index]

            # Add Gaussian noise
            data[t, i] += np.random.normal(scale=0.01)

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=[f'Z{i}' for i in range(n)])
    df = df.apply(normalize)

    red_matrix = reduce_causal_matrix(adjacency_matrix.astype(int), g, s)
    np.fill_diagonal(red_matrix, 1)
    full_matrix = adjacency_matrix.astype(int)
    np.fill_diagonal(full_matrix, 1) 

    plot_graph(g, red_matrix)

    return df, red_matrix, full_matrix
