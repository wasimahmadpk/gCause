# import netCDF
import math
import random
import parameters
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import preprocessing as prep
import matplotlib.pyplot as plt


# np.random.seed(1)

class SyntheticDataset:

    def __init__(self, root1, root2, time_steps, Tref, C, Tao, dynamic_noise):

        self.time_steps = time_steps

        self.root1 = root1
        self.root2 = root2
        self.C = C
        self.Tao = Tao
    
        self.X1 = list(np.zeros(10))
        self.X2 = list(np.zeros(10))
        self.X3 = list(np.zeros(10))
        self.X4 = list(np.zeros(10))
        self.X5 = list(np.zeros(10))
        self.X6 = list(np.zeros(10))

    def generate_data(self):

        for t in range(10, self.time_steps): 
            
            # Subsystem: 1
            self.X1.append(self.root1[t] + dynmaic_noise['n1'][t])
            self.X2.append(C.get('c1') * self.X1[t - Tao.get('t1')] + dynmaic_noise['n2'][t])
            self.X3.append(C.get('c2') * ((self.X1[t - Tao.get('t2')])/2) + dynmaic_noise['n3'][t])
            
            # Subsystem: 2
            self.X4.append(self.root2[t] + dynmaic_noise['n4'][t])
            # self.X4.append(C.get('c1') * self.X1[t - Tao.get('t2')] + dynmaic_noise['n4'][t])
            self.X5.append(C.get('c4') * self.X2[t - Tao.get('t3')] + dynmaic_noise['n5'][t])
            self.X6.append(C.get('c5') * self.X4[t - Tao.get('t3')] + dynmaic_noise['n6'][t])
            
        return self.X1, self.X2, self.X3, self.X4, self.X5, self.X6


if __name__ == '__main__':

    def generate_sine_wave(freq, sample_rate, duration):
        t = np.linspace(0, duration, sample_rate * duration, endpoint=False)
        frequencies = t * freq
        # 2pi because np.sin takes radians
        y = np.sin((2 * np.pi) * frequencies)
        return t, y

    # Generate sine wave
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

    noise2 = np.random.normal(0, 1.5, len(nice_wave))
    root2 = noise2

    # root = np.random.normal(0, 1.0, 2000)
    time_steps, Tref = 2100, 15

    dynmaic_noise = {'n1': np.random.normal(0.0, 0.30, 2*time_steps),
                     'n2': np.random.normal(0.5, 0.40, 2*time_steps),
                     'n3': np.random.normal(1.0, 0.25, 2*time_steps),

                     'n4': np.random.normal(0.75, 0.40, 2*time_steps),
                     'n5': np.random.normal(1.5, 0.30, 2*time_steps),
                     'n6': np.random.normal(1.0, 0.50, 2*time_steps)}
    
    
    C = {'c1': 1.50, 'c2': 1.50, 'c3': 1.75, 'c4': 1.25, 'c5': 1.60, 'c6': 1.25}           # c2:1.75, c5:1.85
    Tao = {'t1': 2, 't2': 3, 't3': 4, 't4': 1, 't5': 6, 't6': 5}
    data_obj = SyntheticDataset(root1, root2, time_steps, Tref, C, Tao, dynmaic_noise)
    X1, X2, X3, X4, X5, X6 = data_obj.generate_data()

    data = {'Z1': X1[50:], 'Z2': X2[50:], 'Z3': X3[50:], 'Z4': X4[50:], 'Z5': X5[50:], 'Z6': X6[50:]}
    df = pd.DataFrame(data, columns=['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6'])
    df.to_csv(r'/home/ahmad/Projects/gCause/datasets/synthetic_datasets/synthetic_gts.csv', index_label=False, header=True)
    print(df.head(10))
    print("Correlation Matrix")
    print(df.corr(method='pearson'))

    fig = plt.figure()
    ax1 = fig.add_subplot(511)
    ax1.plot(X1[150:1500])
    ax1.set_ylabel('X1')

    ax2 = fig.add_subplot(512)
    ax2.plot(X2[150:1500])
    ax2.set_ylabel("X2")

    ax3 = fig.add_subplot(513)
    ax3.plot(X3[150:1500])
    ax3.set_ylabel("X3")

    ax4 = fig.add_subplot(514)
    ax4.plot(X4[150:1500])
    ax4.set_ylabel("X4")
    #
    ax5 = fig.add_subplot(515)
    ax5.plot(X5[150:1500])
    ax5.set_ylabel("X5")

    plt.show()
