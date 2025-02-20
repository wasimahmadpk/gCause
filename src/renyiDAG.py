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
    Reduces a causal matrix into a smaller matrix based on the specified number of groups and group size.ss
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


def generate_stationary_data(timesteps, n):
    """
    Generates stationary Gaussian noise for n variables with dynamically assigned means and variances.
    
    Parameters:
        timesteps (int): Number of time steps.
        n (int): Number of variables.
    
    Returns:
        noise (ndarray): Noise matrix of shape (timesteps, n).
        means (ndarray): Randomly generated means for each variable (shape: (n,))
        variances (ndarray): Randomly generated variances for each variable (shape: (n,))
    """
    # Randomly generate different means and variances for each variable
    means = np.random.uniform(-3, 3, size=n)  # Means in range [-2, 2]
    variances = np.random.uniform(0.1, 3.0, size=n)  # Variances in range [0.2, 2]
    std_devs = np.sqrt(variances)  # Convert variances to standard deviations
    print(f'Means: {means}')
    print(f'Variances: {std_devs}')
    # Generate Gaussian noise with these means and std deviations
    data = np.random.normal(loc=means.reshape(1, n), scale=std_devs.reshape(1, n), size=(timesteps, n))
    return data


def generate_stationary_noise(timesteps, n):
    """
    Generates stationary Gaussian noise for n variables with dynamically assigned means and variances.
    
    Parameters:
        timesteps (int): Number of time steps.
        n (int): Number of variables.
    
    Returns:
        noise (ndarray): Noise matrix of shape (timesteps, n).
        means (ndarray): Randomly generated means for each variable (shape: (n,))
        variances (ndarray): Randomly generated variances for each variable (shape: (n,))
    """
    # Randomly generate different means and variances for each variable
    means = np.random.uniform(0, 1, size=n)  # Means in range [-2, 2]
    variances = np.random.uniform(0.5, 1.5, size=n)  # Variances in range [0.2, 2]
    std_devs = np.sqrt(variances)  # Convert variances to standard deviations
    # Generate Gaussian noise with these means and std deviations
    noise = np.random.normal(loc=means.reshape(1, n), scale=std_devs.reshape(1, n), size=(timesteps, n))
    return noise

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

    # # Function to apply nonlinear transformation
    # def nonlinear_transform(value):
    #     return np.sin(value)  # Example nonlinear function

    def nonlinear_transform(value):
        '''
        A more complicated but stable nonlinear transformation combining sine, cosine, exponential, and polynomial components.
        
        Parameters:
        value (float or np.ndarray): The input value or array to apply the transformation on.

        Returns:
        float or np.ndarray: The transformed value.
        '''
        # Ensure the exponential term does not grow too fast
        exp_term = np.exp(-0.05 * np.abs(value))  # Controlled decay with a smaller factor
        # Polynomial term with a cubic component, but with scaling factor to keep values manageable
        poly_term = 0.1 * np.clip(value**3, -100, 100)  # Clip to limit large values
        # Sine and Cosine terms with added scaling to prevent extreme oscillations
        trig_term = np.sin(value) * 0.5 + np.cos(value)**2 * 0.3
        # Apply a bounded logarithmic transformation for negative values
        log_term = np.zeros_like(value)
        log_term[value < 0] = np.log1p(np.abs(value[value < 0]))  # log(1 + abs(x)) for stability
        # Combine all terms while ensuring the final result is bounded between -10 and 10
        transformed_value = trig_term 
        # Clip the final result to a reasonable range to avoid explosion
        transformed_value = np.clip(transformed_value, -10, 10)
        return transformed_value


   # Initialize the time series with random Gaussian noise
    # data = np.random.normal(size=(timesteps, n))
    data = generate_stationary_data(timesteps, n)
    noise = generate_stationary_noise(timesteps, n)
    coeffs = np.random.uniform(0.5, 1.0, n)
    # Generate unique seasonality for each variable
    seasonality = np.zeros((timesteps, n))

    for i in range(n):
        phase_shift = np.random.uniform(0, 2 * np.pi)  # Unique phase shift for each variable
        freq1 = 12 + np.random.randint(-2, 3)  # Slightly vary the frequency
        freq2 = 18 + np.random.randint(-2, 3)
    
        seasonality[:, i] = (
            np.sin(np.linspace(0, 2 * np.pi * timesteps / freq1, timesteps) + phase_shift) +
            np.cos(np.linspace(0, 4 * np.pi * timesteps / freq2, timesteps) + phase_shift)
            )

    # Add unique seasonality to each time series
    data = data + 0.33*seasonality

    # Update time series data based on DAG relationships, including autoregression on itself
    for t in range(1, timesteps):
        for i in range(n):
            # # Start with the own past value
            data[t, i] = data[t, i] + coeffs[i]*data[t-1, i]

            # Add contributions from the parents
            parents = list(G.predecessors(f'G{i+1}'))
            # print(f'Parents of var {i} are {parents} with nonlinear link: {nonlinear_links}')
            parents_sum = 0
            if parents:
                parent_indices = [int(p[1:]) - 1 for p in parents]
                for parent_index in parent_indices:
                    if nonlinear_links[(parent_index, i)]:
                        # print(f'Var: {i} is nonlinear parents: {parent_index}')
                        parents_sum += coeffs[i]*nonlinear_transform(data[t-3, parent_index])
                        # data[t, i] = data[t, i] + coeffs[i]*nonlinear_transform(data[t-3, parent_index])
                    else:
                        parents_sum += coeffs[i]*data[t-3, parent_index]
                        # data[t, i] = data[t, i] + coeffs[i]*data[t-3, parent_index]

            # Add Gaussian noise again
            # data[t, i] += noise[t, i]
            data[t, i] += parents_sum

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=[f'Z{i}' for i in range(n)])
    df = df.apply(normalize)

    red_matrix = reduce_causal_matrix(adjacency_matrix.astype(int), g, s)
    np.fill_diagonal(red_matrix, 1)
    full_matrix = adjacency_matrix.astype(int)
    np.fill_diagonal(full_matrix, 1) 

    plot_graph(g, red_matrix)
    return df, red_matrix, full_matrix
