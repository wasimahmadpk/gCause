import numpy as np
import pandas as pd
import random
import pathlib

def get_sig_params():
    pars = dict()
    pars["sample_rate"] = 44100  # Hertz
    pars["duration"] = 5   # seconds
    return pars


def GetDistributionParams(model, p):
    """
    Returns parameters for generating different data distributions
    """
    params = dict()
    params["model"] = model
    params["p"] = p
    if model == "gaussian":
        params["rho"] = 0.5
    elif model == "gmm":
        params["rho-list"] = [0.3,0.5,0.7]
    elif model == "mstudent":
        params["df"] = 3
        params["rho"] = 0.5
    elif model == "sparse":
        params["sparsity"] = int(0.3*p)
    else:
        raise Exception('Unknown model generating distribution: ' + model)
    
    return params
        

def GetTrainingHyperParams(model):
    """
    Returns the default hyperparameters for training deep knockoffs
    as described in the paper
    """
    params = dict()
    
    params['GAMMA'] = 1.0
    if model == "gaussian":
        params['LAMBDA'] = 1.0
        params['DELTA'] = 1.0
    elif model == "gmm":
        params['LAMBDA'] = 1.0
        params['DELTA'] = 1.0
    elif model == "mstudent":
        params['LAMBDA'] = 0.01
        params['DELTA'] = 0.01
    elif model == "sparse":
        params['LAMBDA'] = 0.1
        params['DELTA'] = 1.0
    else:
        raise Exception('Unknown data distribution: ' + model)
        
    return params


def GetFDRTestParams(model):
    """
    Returns the default hyperparameters for performing controlled
    variable selection experiments as described in the paper
    """
    params = dict()
    # Test parameters for each model
    if model in ["gaussian", "gmm"]:
        params["n"] = 150
        params["elasticnet_alpha"] = 0.1
    elif model in ["mstudent"]:
        params["n"] = 200
        params["elasticnet_alpha"] = 0.0
    elif model in ["sparse"]:
        params["n"] = 200
        params["elasticnet_alpha"] = 0.0
    
    return params


# Default parameters dictionary
def get_default_params():
    default_params = {

        'group_num': 4,    
        'groups': {'g1': [0, 3], 'g2': [3, 5], 'g3': [5, 8], 'g4': [8, 10]},
        'groups_size': {'g1':[3], 'g2':[2], 'g3':[3], 'g4':[2]},
        'epochs': 100,
        'pred_len': 15,
        'train_len': 555,
        'num_layers': 6,
        'num_cells': 60,
        'num_samples': 10,
        'dropout_rate': 0.1,
        'win_size': 1,
        'num_sliding_win': 15,
        'step_size': 5,
        'dim': 10,
        'batch_size': 32,
        'prior_graph': np.array([[1, 1, 1, 0, 1],
                                 [0, 1, 0, 0, 0],
                                 [0, 0, 1, 1, 0],
                                 [0, 0, 0, 1, 1],
                                 [0, 0, 0, 0, 1]]),
        'true_graph': [1, 1, 1, 0, 1,
                       0, 1, 0, 0, 0,
                       0, 0, 1, 1, 0,
                       0, 0, 0, 1, 1,
                       0, 0, 0, 0, 1],
        'freq': '30min',
        'plot_path': "/home/ahmad/Projects/gCause/plots/multigraphs/",
        'model_path': "/home/ahmad/Projects/gCause/models/",
        'model_name': 'trained_synthetic'
    }
    return default_params


def get_syn_params():
    # Parameters for synthetic data
    rnd = random.randint(1, 9999)  # Generate a random number
         # Ensure the directory exists
    plot_path_rnd = f"/home/ahmad/Projects/gCause/plots/multigraphs_syn_{rnd}/"
    pathlib.Path(plot_path_rnd).mkdir(parents=True, exist_ok=True)
    
    params = {
            'group_num': 4,    
            'groups': {'g1': [0, 2], 'g2': [2, 4], 'g3': [4, 6], 'g4': [6, 8]},    
            'groups_size': {'g1':[2], 'g2':[2], 'g3':[2], 'g4':[2]},               

            'group_num_fs': 18,    
            'groups_fs': {'g1': [0, 1], 'g2': [1, 2], 'g3': [2, 3], 'g4': [3, 4], 'g5': [4, 5], 
                        'g6': [5, 6], 'g7': [6, 7], 'g8': [7, 8], 'g9': [8, 9], 'g10': [9, 10], 
                        'g11': [10, 11], 'g12': [11, 12], 'g13': [12, 13], 'g14': [13, 14], 
                        'g15': [14, 15], 'g16': [15, 16], 'g17': [16, 17],'g18': [17, 18]},     
            'groups_size_fs': {f'g{i}': [1] for i in range(1, 13)},                        

            'groups_cc': {'g1': [0, 1], 'g2': [1, 2], 'g3': [2, 3], 'g4': [3, 4]},      
            'groups_size_cc': {'g1':[1], 'g2':[1], 'g3':[1], 'g4':[1]},                    
        
            'epochs': 40,
            'pred_len': 4,
            'train_len': 1000,
            'num_layers': 5,
            'num_cells': 50,
            'num_samples': 3,
            'dropout_rate': 0.1,
            'num_sliding_win': 30,
            'step_size': 3,
            'dim': 8,
            'dim_fs': 8,
            'dim_cc': 4,
            'alpha' : 0.1,
            'batch_size': 32,
            'test': 'ks',
            'intervention': 'knockoffs',
            'prior_graph': [[1, 1, 1, 0, 1],
                            [0, 1, 0, 0, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 1]],
            'true_graph': [1, 1, 1, 0, 1,
                        0, 1, 0, 0, 0,
                        0, 0, 1, 1, 0,
                        0, 0, 0, 1, 1,
                        0, 0, 0, 0, 1],
            'test_names' : [
                "KS Test", "MWU Test", "t-Test", "AD Test", 
                "SW Test", "CM Test", "WSR Test"
            ],
            'freq': '30min',
            'plot_path': plot_path_rnd,
            'model_path': "/home/ahmad/Projects/gCause/models/gc_syn/",
            'model_name': 'trained_synthetic',
            'model_name_cc': 'trained_synthetic_cc'
        }

   
    return params


def get_flux_params():

    # Parameters for flux data
    params = {

        'group_num': 2,    
        'groups': {'g1': [0, 2], 'g2': [2, 4]},
        'groups_size': {'g1':[2], 'g2':[2]},
        'epochs': 30,
        'pred_len': 14,
        'train_len': 1000,
        'num_layers': 4,
        'num_cells': 40,
        'num_samples': 10,
        'dropout_rate': 0.1,
        'win_size': 1,
        'num_sliding_win': 50,
        'step_size': 10,
        'alpha' : 0.10,
        'dim': 4,
        'batch_size': 32,
        'prior_graph': np.array([[1, 1, 1, 0, 1],
                                 [0, 1, 0, 0, 0],
                                 [0, 0, 1, 1, 0],
                                 [0, 0, 0, 1, 1],
                                 [0, 0, 0, 0, 1]]),
        'true_graph': [1, 1, 1, 0, 1,
                       0, 1, 0, 0, 0,
                       0, 0, 1, 1, 0,
                       0, 0, 0, 1, 1,
                       0, 0, 0, 0, 1],
        'freq': '30min',
        'plot_path': "/home/ahmad/Projects/gCause/plots/multigraphs_flux/",
        'model_path': "/home/ahmad/Projects/gCause/models/gc_flux/",
        'model_name': 'trained_model_geotest2.sav'
    }
    return params


def get_nino_params():

    params = {

        'group_num': 2,    
        'groups': {'g1': [0, 7], 'g2': [7, 16]},    # 'g1': [0, 33], 'g2': [33, 69]
        'groups_size': {'g1':[7], 'g2':[9]},
        'epochs': 75,
        'pred_len': 8,
        'train_len': 40,
        'num_layers': 4,
        'num_cells': 50,
        'num_samples': 10,
        'dropout_rate': 0.1,
        'win_size': 1,
        'slidingwin_size': 100,
        'step_size': 10,
        'dim': 16,
        'batch_size': 32,
        'prior_graph': np.array([[1, 1, 1, 0, 1],
                                 [0, 1, 0, 0, 0],
                                 [0, 0, 1, 1, 0],
                                 [0, 0, 0, 1, 1],
                                 [0, 0, 0, 0, 1]]),
        'true_graph': [1, 1, 1, 0, 1,
                       0, 1, 0, 0, 0,
                       0, 0, 1, 1, 0,
                       0, 0, 0, 1, 1,
                       0, 0, 0, 0, 1],
        'freq': '30min',
        'plot_path': "/home/ahmad/Projects/gCause/plots/"
    }
    return params


def get_geo_params():

    params = {
        'epochs': 80,    # 125
        'pred_len': 15,    # 15
        'train_len': 666,  # 1500
        'num_layers': 5,    # 5
        'num_samples': 10,
        'num_cells': 55,     # 50
        'dropout_rate': 0.1,
        'win_size': 1,
        'slidingwin_size': 100,
        'dim': 5,
        'batch_size': 32,
        'prior_graph': np.array([[1, 0, 0, 0, 1],
                                 [0, 1, 0, 0, 1],
                                 [0, 0, 1, 0, 1],
                                 [0, 0, 0, 1, 1],
                                 [0, 0, 0, 0, 1]]),
        'true_graph': [1, 0, 0, 0, 1,
                       0, 1, 0, 0, 1,
                       0, 0, 1, 0, 1,
                       0, 0, 0, 1, 1,
                       0, 0, 0, 0, 1],
        'freq': 'H',
        'plot_path': "/home/ahmad/Projects/gCause/plots/"
    }
    return params


def get_geo_params_gc():

    params = {
        
        'group_num': 3,    
        'groups': {'g1': [0, 4], 'g2': [4, 6], 'g3': [6, 8]}, #  
        'groups_size': {'g1':[4], 'g2':[2], 'g3':[2]}, # 

        'group_num_fs': 8,    
        'groups_fs': {'g1': [0, 1], 'g2': [1, 2], 'g3': [2, 3], 'g4': [3, 4], 'g5': [4, 5], 'g6': [5, 6], 'g7': [6, 7], 'g8': [7, 8]}, #  
        'groups_size_fs': {'g1': [1], 'g2': [1], 'g3': [1], 'g4': [1], 'g5': [1], 'g6': [1], 'g7': [1], 'g8': [1]}, # 
         
        'groups_cc': {'g1': [0, 1], 'g2': [1, 2], 'g3': [2, 3]}, #  
        'groups_size_cc': {'g1':[1], 'g2':[1], 'g3':[1]}, #    

        'epochs': 50,
        'pred_len': 12,
        'train_len': 1000,
        'num_layers': 5,
        'num_cells': 40,
        'num_samples': 2,
        'dropout_rate': 0.10,
        'win_size': 1,
        'num_sliding_win': 21, #21, 25
        'step_size': 12,
        'dim': 8,
        'dim_fs': 8,
        'dim_cc': 4,
        'batch_size': 32,
        'prior_graph': np.array([[1, 1, 1, 0, 1],
                                 [0, 1, 0, 0, 0],
                                 [0, 0, 1, 1, 0],
                                 [0, 0, 0, 1, 1],
                                 [0, 0, 0, 0, 1]]),
        'true_graph': [1, 1, 1, 0, 1,
                       0, 1, 0, 0, 0,
                       0, 0, 1, 1, 0,
                       0, 0, 0, 1, 1,
                       0, 0, 0, 0, 1],
        'freq': 'H',
        'plot_path': "/home/ahmad/Projects/gCause/plots/multigraphs/",
        'model_path': "/home/ahmad/Projects/gCause/models/gc/",
        'model_name': 'trained_strain',
        'model_name_cc': 'trained_strain_cc'
    }
    return params


def get_rivernet_params_gc():

    params = {
        
        'group_num': 3,    
        'groups': {'g1': [0, 1], 'g2': [1, 2], 'g3': [2, 5]}, #  
        'groups_size': {'g1':[1], 'g2':[1], 'g3':[3]}, # 

        'group_num_fs': 5,    
        'groups_fs': {'g1': [0, 1], 'g2': [1, 2], 'g3': [2, 3], 'g4': [3, 4], 'g5': [4, 5]}, #  
        'groups_size_fs': {'g1': [1], 'g2': [1], 'g3': [1], 'g4': [1], 'g5': [1]}, # 
         
        'groups_cc': {'g1': [0, 1], 'g2': [1, 2], 'g3': [2, 3]}, #  
        'groups_size_cc': {'g1':[1], 'g2':[1], 'g3':[1]}, #    

        'epochs': 30,
        'pred_len': 10,
        'train_len': 1000,
        'num_layers': 4,
        'num_cells': 40,
        'num_samples': 1,
        'dropout_rate': 0.01,
        'win_size': 1,
        'num_sliding_win': 21, #21, 25
        'step_size': 1,
        'dim': 8,
        'dim_fs': 8,
        'dim_cc': 4,
        'batch_size': 32,
        'prior_graph': np.array([[1, 1, 1, 0, 1],
                                 [0, 1, 0, 0, 0],
                                 [0, 0, 1, 1, 0],
                                 [0, 0, 0, 1, 1],
                                 [0, 0, 0, 0, 1]]),
        'true_graph': [],
        'freq': '6H',
        'plot_path': "/home/ahmad/Projects/gCause/plots/multigraphs_rivernet/",
        'model_path': "/home/ahmad/Projects/gCause/models/gc_rivernet/",
        'model_name': 'trained_rivernet',
        'model_name_cc': 'trained_rivernet_cc'
    }
    return params


def get_melodi_params():

    params = {
        
        'group_num': 3,    
        'groups': {'g1': [0, 3], 'g2': [3, 4], 'g3': [4, 7]}, #  
        'groups_size': {'g1':[3], 'g2':[1], 'g3':[3]}, # 

        'group_num_fs': 6,    
        'groups_fs': {'g1': [0, 1], 'g2': [1, 2], 'g3': [2, 3], 'g4': [3, 4], 'g5': [4, 5]}, #  
        'groups_size_fs': {'g1': [1], 'g2': [1], 'g3': [1], 'g4': [1], 'g5': [1]}, # 
         
        'groups_cc': {'g1': [0, 1], 'g2': [1, 2], 'g3': [2, 3]}, #  
        'groups_size_cc': {'g1':[1], 'g2':[1], 'g3':[1]}, #    

        'epochs': 25,
        'pred_len': 3,
        'train_len': 666,
        'num_layers': 4,
        'num_cells': 40,
        'num_samples': 1,
        'dropout_rate': 0.01,
        'win_size': 3,
        'num_sliding_win': 30,
        'step_size': 3,
        'dim': 8,
        'dim_fs': 6,
        'dim_cc': 3,
        'batch_size': 32,
        'alpha' : 0.01,
        'prior_graph': np.array([[1, 1, 1, 0, 1],
                                 [0, 1, 0, 0, 0],
                                 [0, 0, 1, 1, 0],
                                 [0, 0, 0, 1, 1],
                                 [0, 0, 0, 0, 1]]),
        'true_graph': [1, 1, 1, 0, 1,
                       0, 1, 0, 0, 0,
                       0, 0, 1, 1, 0,
                       0, 0, 0, 1, 1,
                       0, 0, 0, 0, 1],
        'freq': '6H',
        'plot_path': "/home/ahmad/Projects/gCause/plots/multigraphs_melodi/",
        'model_path': "/home/ahmad/Projects/gCause/models/gc_melodi/",
        'model_name': 'trained_melodi',
        'model_name_cc': 'trained_melodi_cc'
    }
    return params



def get_dyadic_params_gc():

    motor_task = 'M1'
    params = {
        
        'group_num': 2,    
        'groups': {'g1': [0, 2], 'g2': [2, 4]}, #  
        'groups_size': {'g1':[2], 'g2':[2]}, # 

        'group_num_fs': 4,    
        'groups_fs': {'g1': [0, 1], 'g2': [1, 2], 'g3': [2, 3], 'g4': [3, 4]}, #  
        'groups_size_fs': {'g1': [1], 'g2': [1], 'g3': [1], 'g4': [1]}, # 
         
        'groups_cc': {'g1': [0, 1], 'g2': [1, 2]}, #  
        'groups_size_cc': {'g1':[1], 'g2':[1]}, #    

        'epochs': 30,
        'pred_len': 5,
        'train_len': 170,
        'num_layers': 4,
        'num_cells': 40,
        'num_samples': 2,
        'dropout_rate': 0.10,
        'win_size': 1,
        'num_sliding_win': 18, #21, 25
        'step_size': 1,
        'dim': 4,
        'dim_fs': 4,
        'dim_cc': 2,
        'batch_size': 32,
        'prior_graph': np.array([[1, 1, 1, 0, 1],
                                 [0, 1, 0, 0, 0],
                                 [0, 0, 1, 1, 0],
                                 [0, 0, 0, 1, 1],
                                 [0, 0, 0, 0, 1]]),
        'true_graph': [1, 1, 1, 0, 1,
                       0, 1, 0, 0, 0,
                       0, 0, 1, 1, 0,
                       0, 0, 0, 1, 1,
                       0, 0, 0, 0, 1],
        'freq': 'H',
        'motor_task': motor_task,
        'data_path': f'/home/ahmad/Projects/gCause/datasets/fnirs/{motor_task}',
        'plot_path': '/home/ahmad/Projects/gCause/plots/multigraphs/',
        'json_path': '/home/ahmad/Projects/gCause/src/json/',
        'model_path': '/home/ahmad/Projects/gCause/models/gc/',
        'model_name': 'trained_fnirs',
        'model_name_cc': 'trained_fnirs_cc'
    }
    return params



def get_hack_params():

    params = {
        'epochs': 100,
        'pred_len': 24,
        'train_len': 21*24,
        'num_layers': 6,
        'num_samples': 10,
        'num_cells': 50,
        'dropout_rate': 0.1,
        'win_size': 1,
        'dim': 6,
        'batch_size': 32,
        'prior_graph': np.array([[1, 0, 1, 0, 1, 1],
                                 [1, 0, 1, 0, 0, 1],
                                 [1, 1, 0, 1, 1, 0],
                                 [1, 0, 1, 0, 1, 1],
                                 [1, 0, 1, 0, 1, 1],
                                 [1, 0, 1, 0, 1, 1]]),
        'true_graph': [1, 0, 1, 0, 1, 1,
                       1, 0, 1, 0, 1, 1,
                       1, 1, 0, 1, 1, 0,
                       1, 0, 1, 0, 1, 0,
                       1, 0, 1, 0, 1, 1,
                       1, 0, 1, 0, 1, 1],
        'freq': 'D',
        'plot_path': "/home/ahmad/Projects/gCause/plots/"
    }
    return params


def get_netsim_params():

    params = {

        'group_num': 3,    
        'groups': {'g1': [0, 5], 'g2': [5, 10], 'g3': [10, 15]},    # 'g1': [0, 33], 'g2': [33, 69]
        'groups_size': {'g1':[5], 'g2':[5], 'g3':[5]},
        'epochs': 25,
        'pred_len': 3,
        'train_len': 150,
        'num_layers': 5,
        'num_cells': 50,
        'num_samples': 10,
        'dropout_rate': 0.1,
        'win_size': 1,
        'slidingwin_size': 50,
        'step_size': 1,
        'dim': 15,
        'batch_size': 32,
        'prior_graph': np.array([[1, 1, 1, 0, 1],
                                 [0, 1, 0, 0, 0],
                                 [0, 0, 1, 1, 0],
                                 [0, 0, 0, 1, 1],
                                 [0, 0, 0, 0, 1]]),
        'true_graph': [1, 1, 1, 0, 1,
                       0, 1, 0, 0, 0,
                       0, 0, 1, 1, 0,
                       0, 0, 0, 1, 1,
                       0, 0, 0, 0, 1],
        'freq': '30min',
        'plot_path': "/home/ahmad/Projects/gCause/plots/"
    }
    return params

def get_sims_params():

    params = {

        'group_num': 4,    
        'groups': {'g1': [0, 2], 'g2': [2, 4], 'g3': [4, 6], 'g4': [6, 8]}, #  
        'groups_size': {'g1':[2], 'g2':[2], 'g3':[2], 'g4':[2]}, # 

        'group_num_full': 8,    
        'groups_full': {'g1': [0, 1], 'g2': [1, 2], 'g3': [2, 3], 'g4': [3, 4], 'g5': [4, 5], 'g6': [5, 6], 'g7': [6, 7], 'g8': [7, 8]}, #  
        'groups_size_full': {'g1':[1], 'g2':[1], 'g3':[1], 'g4':[1], 'g5':[1], 'g6':[1], 'g7':[1], 'g8':[1]}, # 
         
        'groups_cc': {'g1': [0, 1], 'g2': [1, 2], 'g3': [2, 3], 'g4': [3, 4]}, #  
        'groups_size_cc': {'g1':[1], 'g2':[1], 'g3':[1], 'g4':[1]}, #    
        
        'epochs': 50,
        'pred_len': 5,
        'train_len': 125, #120,
        'num_layers': 5,
        'num_cells': 50,
        'num_samples': 5,
        'dropout_rate': 0.10,
        'num_sliding_win': 5,
        'step_size': 1,
        'dim': 8,
        'dim_cc': 4,
        'alpha' : 0.05,
        'batch_size': 32,
        'prior_graph': np.array([[1, 1, 1, 0, 1],
                                 [0, 1, 0, 0, 0],
                                 [0, 0, 1, 1, 0],
                                 [0, 0, 0, 1, 1],
                                 [0, 0, 0, 0, 1]]),
        'true_graph': [1, 1, 1, 0, 1,
                       0, 1, 0, 0, 0,
                       0, 0, 1, 1, 0,
                       0, 0, 0, 1, 1,
                       0, 0, 0, 0, 1],
        'freq': '30min',
        'plot_path': "/home/ahmad/Projects/gCause/plots/multigraphs_sim/",
        'model_path': "/home/ahmad/Projects/gCause/models/gc_sim/",
        'model_name': 'trained_netsim',
        'model_name_cc': 'trained_netsim_cc'
    }

    return params
