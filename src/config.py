import numpy as np

class BaseParams:
    def __init__(self):
        self.epochs = 50
        self.train_len = 500
        self.pred_len = 5
        self.num_layers = 5
        self.num_cells = 50
        self.num_samples = 5
        self.dropout_rate = 0.1
        self.win_size = 1
        self.batch_size = 32
        self.slidingwin_size = 100
        self.step_size = 10
        self.freq = '30min'
        self.plot_path = "/home/ahmad/Projects/gCause/plots/"
        self.model_path = "/home/ahmad/Projects/gCause/models/"
        self.prior_graph = np.array([[1, 1, 1, 0, 1],
                                     [0, 1, 0, 0, 0],
                                     [0, 0, 1, 1, 0],
                                     [0, 0, 0, 1, 1],
                                     [0, 0, 0, 0, 1]])
        self.true_graph = [1, 1, 1, 0, 1,
                           0, 1, 0, 0, 0,
                           0, 0, 1, 1, 0,
                           0, 0, 0, 1, 1,
                           0, 0, 0, 0, 1]


class SyntheticParams(BaseParams):
    def __init__(self):
        super().__init__()
        self.group_num = 4
        self.groups = {'g1': [0, 2], 'g2': [2, 4], 'g3': [4, 6], 'g4': [6, 8]}
        self.groups_size = {'g1': [2], 'g2': [2], 'g3': [2], 'g4': [2]}
        self.num_sliding_win = 18
        self.dim = 8
        self.dim_fs = 8
        self.dim_cc = 4
        self.model_name = 'trained_synthetic'
        self.model_name_cc = 'trained_synthetic_cc'


class FluxParams(BaseParams):
    def __init__(self):
        super().__init__()
        self.group_num = 2
        self.groups = {'g1': [0, 2], 'g2': [2, 4]}
        self.groups_size = {'g1': [2], 'g2': [2]}
        self.pred_len = 14
        self.train_len = 1000
        self.num_layers = 4
        self.num_cells = 40
        self.slidingwin_size = 100
        self.step_size = 10
        self.dim = 4
        self.model_name = 'trained_model_geotest2.sav'


class NinoParams(BaseParams):
    def __init__(self):
        super().__init__()
        self.group_num = 2
        self.groups = {'g1': [0, 7], 'g2': [7, 16]}
        self.groups_size = {'g1': [7], 'g2': [9]}
        self.epochs = 75
        self.pred_len = 8
        self.train_len = 40
        self.num_layers = 4
        self.num_cells = 50
        self.dim = 16


class GeoParams(BaseParams):
    def __init__(self):
        super().__init__()
        self.epochs = 80
        self.pred_len = 15
        self.train_len = 666
        self.num_cells = 55
        self.num_samples = 10
        self.dim = 5
        self.freq = 'H'


class GeoParamsGC(BaseParams):
    def __init__(self):
        super().__init__()
        self.group_num = 3
        self.groups = {'g1': [0, 2], 'g2': [2, 4], 'g3': [4, 6]}
        self.groups_size = {'g1': [2], 'g2': [2], 'g3': [2]}
        self.epochs = 150
        self.pred_len = 15
        self.train_len = 666
        self.num_cells = 50
        self.slidingwin_size = 15
        self.step_size = 10
        self.dim = 6


class HackParams(BaseParams):
    def __init__(self):
        super().__init__()
        self.epochs = 100
        self.pred_len = 24
        self.train_len = 21 * 24
        self.num_layers = 6
        self.dim = 6
        self.prior_graph = np.array([[1, 0, 1, 0, 1, 1],
                                     [1, 0, 1, 0, 0, 1],
                                     [1, 1, 0, 1, 1, 0],
                                     [1, 0, 1, 0, 1, 1],
                                     [1, 0, 1, 0, 1, 1],
                                     [1, 0, 1, 0, 1, 1]])
        self.true_graph = [1, 0, 1, 0, 1, 1,
                           1, 0, 1, 0, 1, 1,
                           1, 1, 0, 1, 1, 0,
                           1, 0, 1, 0, 1, 0,
                           1, 0, 1, 0, 1, 1,
                           1, 0, 1, 0, 1, 1]
        self.freq = 'D'


class NetsimParams(BaseParams):
    def __init__(self):
        super().__init__()
        self.group_num = 3
        self.groups = {'g1': [0, 5], 'g2': [5, 10], 'g3': [10, 15]}
        self.groups_size = {'g1': [5], 'g2': [5], 'g3': [5]}
        self.epochs = 75
        self.pred_len = 5
        self.train_len = 145
        self.dim = 15
        self.model_name = 'trained_netsim'
        self.model_name_cc = 'trained_netsim_cc'


class SimsParams(BaseParams):
    def __init__(self):
        super().__init__()
        self.group_num = 4
        self.groups = {'g1': [0, 2], 'g2': [2, 4], 'g3': [4, 6], 'g4': [6, 8]}
        self.groups_size = {'g1': [2], 'g2': [2], 'g3': [2], 'g4': [2]}
        self.num_sliding_win = 15
        self.step_size = 3
        self.dim = 8
        self.dim_cc = 4
        self.model_name = 'trained_netsim'
        self.model_name_cc = 'trained_netsim_cc'


# Factory method to get params for different models
def get_params(model_name):
    if model_name == "synthetic":
        return SyntheticParams()
    elif model_name == "flux":
        return FluxParams()
    elif model_name == "nino":
        return NinoParams()
    elif model_name == "geo":
        return GeoParams()
    elif model_name == "geo_gc":
        return GeoParamsGC()
    elif model_name == "hack":
        return HackParams()
    elif model_name == "netsim":
        return NetsimParams()
    elif model_name == "sims":
        return SimsParams()
    else:
        raise ValueError(f"Unknown model: {model_name}")

