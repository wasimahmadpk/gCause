import re
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

class StructuralCausalModel:
    
    def __init__(self):
        self.causal_graph = None
        self.labels = None
        self.path = r'../datasets/synthetic_datasets'
        self.nonlinear_functions = ['trig'] # [trig, quadratic, exp]

    def generate_multivariate_time_series(self, num_nodes, nonlinearity, interaction_density, num_samples=111):
        # Step 1: Define initial causal graph structure (acyclic) and generate links
        
        # Generate initial causal graph with randomly assigned 0s and 1s
        causal_graph = np.zeros((num_nodes, num_nodes), dtype=int)
        links = []
        
        # Calculate the number of possible links (without cycles/loops)
        num_possible_links = num_nodes * (num_nodes - 1)
        
        # Calculate the number of links to retain based on interaction density
        num_links_to_retain = int(num_possible_links * interaction_density)
        
        if num_links_to_retain == 0:
            # For interaction_density=0, print a message and break
            print("There are no causal links at all.")
            return None, None, None
        
        # Randomly assign causal connections based on interaction density
        for i in range(num_nodes):
            # Ensure at least one parent for each node
            num_parents = min(i, max(1, int(np.random.binomial(num_nodes-1, interaction_density))))
            if num_parents == 0:
                continue  # Skip this node if no parents are selected
            parents = np.random.choice(np.arange(i), size=num_parents, replace=False)
            print(f'Parents: {parents}')
            for p in parents:
                lag_var = np.random.randint(1, 3)  # Random lag between 1 and 4 for the variable itself
                lag_parent = np.random.randint(1, 3)  # Random lag between 1 and 4 for the parent variable
                causal_graph[i, p] = lag_parent
                links.append(((p, i), lag_parent, lag_var))  # Store parent index, parent lag, and variable lag
        
        # Set diagonal elements to represent self-connections (autoregressive dependencies)
        np.fill_diagonal(causal_graph, 1)
        
        # Print the causal graph matrix for debugging
        print("Causal graph matrix:")
        print(causal_graph)
        
        # Step 2: Assign functions to variables based on nonlinearity ratio
        num_nonlinear_vars = int(num_nodes * nonlinearity)
        nonlinear_vars = np.random.choice(np.arange(num_nodes), size=num_nonlinear_vars, replace=False)
        
        # Step 3: Generate time series data based on structural equations
        data = np.zeros((num_samples, num_nodes))
        for t in range(1, num_samples):
            for i in range(num_nodes):
                parents = np.nonzero(causal_graph[i])[0]
                if i in nonlinear_vars:
                    # Calculate the contributions from parent variables
                    parent_values_sum = sum(data[t-lp, p] for p, lp, _ in links if i == p)  # Sum over parent variables' lagged values
                    var_lag = next((lv for p, _, lv in links if i == p), 1)  # Get the lag for the variable itself, default to 1 if not found
                    data[t, i] = 0.5 * data[t-var_lag, i] + parent_values_sum + np.random.normal(0, 0.25)
                else:
                    # Calculate the contributions from parent variables
                    parent_values_sum = sum(data[t-lp, p] for p, lp, _ in links if i == p)  # Sum over parent variables' lagged values
                    data[t, i] = parent_values_sum + np.random.normal(0, 0.25)
        
        # Create DataFrame with variable names Z1, Z2, ..., Zn
        column_names = [f'Z{i+1}' for i in range(num_nodes)]
        df = pd.DataFrame(data, columns=column_names)
        
        # Normalize the DataFrame
        normalized_df = self.normalize_dataframe(df)
        filename = 'synthetic_dataset_n2n.csv'
        normalized_df.to_csv(os.path.join(self.path, filename), index_label=False, header=True)
        normalized_df.plot(figsize=(10, 6))
        # Save causal graph and labels to class attributes
        self.causal_graph = causal_graph
        self.labels = column_names
        
        return normalized_df, links, causal_graph


    def apply_nonlinear_function(self, x, function_choice):
        
        if function_choice == 'trig':
            return np.sin(x)
        elif function_choice == 'exp':
            # Limit the range of input values to avoid overflow
            x_clipped = np.clip(x, a_min=None, a_max=700)  # Set maximum value to avoid overflow
            return np.exp(x_clipped)
        elif function_choice == 'quadratic':
            return x ** 2
        else:
            raise ValueError("Invalid nonlinear function choice")

    def draw_causal_graph(self):
        if self.causal_graph is None or self.labels is None:
            print("Causal graph and labels are not available.")
            return
        for i, label in enumerate(self.labels):
            self.labels[i] = re.sub(r'(\d+)', lambda match: f'$_{match.group(1)}$', label)            # fr'{label[:-1]}$_{{{label[-1]}}}$'
        # Initialize empty lists for FROM and TO
        src_node = []
        dest_node = []

        # Iterate over rows
        for i, row in enumerate(self.causal_graph):
            # Iterate over columns
            for j, value in enumerate(row):
                # If value is not 0, add variable name to FROM list and column name to TO list
                if value != 0:
                    src_node.append(self.labels[i])
                    dest_node.append(self.labels[j])

        # Create graph object
        G = nx.DiGraph()

        # Add all nodes to the graph
        for label in self.labels:
            G.add_node(label)
        
        # Add edges from FROM to TO
        for from_node, to_node in zip(src_node, dest_node):
            # Exclude self-connections
            if from_node != to_node:
                G.add_edge(from_node, to_node)

        # Plot the graph
        fig, ax = plt.subplots(figsize=(7, 7))

        pos = nx.circular_layout(G)

        # Draw nodes with fancy shapes and colors
        node_size = 5000
        node_color = sns.light_palette("green", len(G.nodes))  # Use lighter shades from the seaborn color palette
        node_shape = "o"  # Circle shape
        node_style = "solid"  # Solid outline
        node_alpha = 0.75
        nx.draw_networkx(G, pos, arrows=True, node_size=node_size, node_color=node_color[0], node_shape=node_shape,
                         edge_color='black', width=2, connectionstyle='arc3, rad=0.25',
                         edgecolors="grey", linewidths=2, alpha=node_alpha, font_size=16,
                         font_weight='bold', ax=ax, arrowsize=20)  # Adjust arrowsize

        # Add labels with numbers as suffixes in the format Z$_1$, Z$_2$, etc.
        labels = {node: fr'{node[:-1]}$_{{{node[-1]}}}$' for node in G.nodes}
        # nx.draw_networkx_labels(G, pos, labels, font_size=16, font_weight='bold')

        ax.set(facecolor="white")
        ax.grid(False)
        ax.set_xlim([1.1 * x for x in ax.get_xlim()])
        ax.set_ylim([1.1 * y for y in ax.get_ylim()])

        plt.axis('off')
        plt.tight_layout()  # Adjust layout to make the graph fit better
        plt.show()

    def normalize_dataframe(self, df):
        normalized_df = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        return normalized_df

if __name__ == "__main__":
    # Example usage:
    model = StructuralCausalModel()
    num_nodes = 2  # Set to 2 for testing
    nonlinearity = 0.5
    interaction_density = 1.0  # Ensure high interaction density
    num_samples = 100
    normalized_df, links, causal_graph = model.generate_multivariate_time_series(num_nodes, nonlinearity, interaction_density, num_samples)
    print("Causal graph matrix:")
    print(causal_graph)
    model.draw_causal_graph()
