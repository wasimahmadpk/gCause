# import netCDF
import math
import random
import parameters
import numpy as np
import pandas as pd
import networkx as nx
from netCDF4 import Dataset
import preprocessing as prep
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
# np.random.seed(1)

class SCMS:

    def __init__(self, num_nodes, link_density=0.15, time_steps=15000):

        self.num_nodes = num_nodes
        self.link_density = link_density
        self.time_steps = time_steps
        self.ts_length = time_steps - 13500
        self.Tao = range(1, 6)

        # adj_mat = np.random.randint(2, size=(self.num_nodes, self.num_nodes))
        adj_mat = self.generate_adj_matrix()
        print('Matrix:\n', adj_mat)
        adj_mat_upp = np.triu(adj_mat)
        res = np.where(adj_mat_upp==1)
        
        list_links_all = list(zip(res[0], res[1]))
        self.list_links = []

        for links in list_links_all:
            if links[0]!=links[1]:
                self.list_links.append(links)
        self.num_links = len(self.list_links)
                
        self.node_labels = [f'Z{l+1}'for l in range(num_nodes)]
        self.generate_ts_DAG()

    def generate_adj_matrix(self):

            # Generate random indices for non-zero elements
            nonzero_indices = np.random.choice(self.num_nodes**2, size=int((self.num_nodes**2) * self.link_density), replace=False)

            # Create a binary matrix with ones at the specified indices
            data = np.ones(len(nonzero_indices), dtype=int)
            row_indices = nonzero_indices // self.num_nodes
            col_indices = nonzero_indices % self.num_nodes

            # Create a sparse binary matrix using CSR format
            sparse_binary_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(self.num_nodes, self.num_nodes))

            # print(sparse_binary_matrix.toarray())
            adj_matrix = sparse_binary_matrix.toarray()
            return adj_matrix

    # Linear cause-effect relation
    def linear(self, cause, effect):
        lag_effect = random.choice(self.Tao)
        lag_cause = random.choice(self.Tao)

        effect = np.random.rand(1)[0]*effect[lag_effect: lag_effect + self.ts_length] + np.random.rand(1)[0]*cause[lag_cause:lag_cause + self.ts_length]
        return effect, len(effect)
    
    # NOn-linear dependency
    def non_linear(self, cause, effect):
        lag_effect = random.choice(self.Tao)
        lag_cause = random.choice(self.Tao)
        effect = np.random.rand(1)[0]*effect[lag_effect: lag_effect + self.ts_length] + np.random.rand(1)[0]**cause[lag_cause:lag_cause + self.ts_length]
        return effect
    
    # Time series base
    def generate_ts(self):

        multivariate_ts = []
        for i in range(self.num_nodes):
            multivariate_ts.append(np.random.normal(0, np.random.rand(1), self.time_steps))
        return np.array(multivariate_ts)
        
        # Generate sine wave and the gaussian noise 
        pars = parameters.get_sig_params()
        SAMPLE_RATE = pars.get("sample_rate")  # Hertz
        DURATION = pars.get("duration")  # Seconds

        # Generate a 2 hertz sine wave that lasts for 5 seconds
        # t, y = generate_sine_wave(2, SAMPLE_RATE, DURATION)

        _, nice_wave = generate_sine_wave(400, SAMPLE_RATE, DURATION)
        _, noise_wave = generate_sine_wave(4000, SAMPLE_RATE, DURATION)
        noise_wave = noise_wave * 0.50
    
        noise1 = np.random.normal(2, 1.10, len(nice_wave))
        root1 = noise1
        ts = np.random.normal(0, 1, self.time_stxeps)
         
    
    def generate_sine_wave(freq, sample_rate, duration):
        t = np.linspace(0, duration, sample_rate * duration, endpoint=False)
        frequencies = t*freq
        # 2pi because np.sin takes radians
        y = np.sin((2*np.pi)*frequencies)
        return t, y
    
    def generate_ts_DAG(self):

        multivariate_dag_ts = self.generate_ts()
        
        for links in range(self.num_links):
            cnode, enode = self.list_links[links][0], self.list_links[links][1]
            ts , len = self.linear(multivariate_dag_ts[enode], multivariate_dag_ts[cnode])
            multivariate_dag_ts[enode, : len] = ts  

        return multivariate_dag_ts
    
    def df_timeseries(self):
        data_dict = {}
        timeseries = self.generate_ts_DAG()
        
        for nodes in range(self.num_nodes):
            data_dict[self.node_labels[nodes]] = timeseries[nodes][: 1500]
        
        df = pd.DataFrame(data=data_dict, columns=self.node_labels)
        df.to_csv(r'/home/ahmad/Projects/gCause/datasets/synthetic_datasets/synthetic_gts.csv', index_label=False, header=True)
        return df, self.list_links
    
    def plot_ts(self):

        fig = plt.figure()
        df, links = self.df_timeseries()
        for i in range(5):
            ax = fig.add_subplot(int(f'51{i+1}'))
            ax.plot(df[df.columns[i]][150:1500].values)
            ax.set_ylabel(f'{df.columns[i]}')
        plt.show()

    def draw_DAG(self):
          
        # Create an empty graph
        G = nx.DiGraph()

        # Add nodes
        for n in range(self.num_nodes):
            G.add_node(n+1, label=f'Z{n+1}')

        for e in range(len(self.list_links)):
            G.add_edge(self.list_links[e][0]+1, self.list_links[e][1]+1)

       # Draw the directed graph with labels
        # pos = nx.spring_layout(G)  # Positions of nodes for visualization
        pos = nx.circular_layout(G)
        labels = nx.get_node_attributes(G, 'label')  # Get labels from node attributes
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=1000, node_color='lightblue', font_size=12, font_color='black', font_weight='bold', edge_color='gray', width=2.0, arrows=True)

        # Display the directed graph with labels
        plt.show()



if __name__ == '__main__':
    
    nodes = 5
    scms = SCMS(nodes)
    df = scms.df_timeseries()
    scms.plot_ts()
