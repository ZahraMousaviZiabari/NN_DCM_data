from sklearn.model_selection import KFold
import torch
import argparse
from torch.utils.data import DataLoader, Subset
from data_utils import ChoiceDataset
from models import TasteNetChoice
from train import train
from evaluation import summarize, printSummary, predictTastes, RMSE, ABSE, RE, RMSE_vector, ABSE_vector, RE_vector
from simulate import error_of_vot, error_of_vowt, dic_z, dic_z_z01, dic_z_zall, inc
import pickle
import os
import copy
import numpy as np
from regress_coef import regress
import itertools
from joblib import Parallel, delayed, parallel_backend
import torch
from data_transform import csv_to_dict


def one_run(data_train, data_dev, args, params):
    """
    A single model training run with one set of hyper-parameters and random initialization seed
    """
    # set up parameters
    if (args.kfold == True):
        torch.manual_seed(params['seed'])
    input_size = 3  # z has 3 dim
    output_size = 1 if params['separate'] else 2
    args.layer_sizes = [input_size, params['hidden_size'], output_size]
    args.activation = params['activation']
    args.transform = params['transform']
    args.weight_decay = params['weight_decay']
    args.separate = params['separate']
    print(args)

    # train

    success = False
    while not success:
        # initialize model
        model = TasteNetChoice(args)
        _, _, model, success = train(model, data_train, data_dev, args, save=False)

    # evaluate
    result, pred_vots_test, pred_vowts_test = one_run_summary(model, ds_train, ds_dev, ds_test)
    mean_vots = pred_vots_test.mean()
    mean_vowts = pred_vowts_test.mean()
    var_vots = pred_vots_test.var()
    var_vowts = pred_vowts_test.var()
    return result, model, pred_vots_test, pred_vowts_test, mean_vots, mean_vowts, var_vots, var_vowts


def one_run_summary(model, ds_train, ds_dev, ds_test):
    # Metrics
    result = {}
    # 1. NLL and ACC for train/dev/test
    summary = summarize(model, ds_train, ds_dev, ds_test)
    result.update(summary)
    x = printSummary(summary, precision=3)
    print(x)

    # 2. Tastes (vots, vowts): predicted vs true (dollar per minute)
    allds = {"train": ds_train, "dev": ds_dev, "test": ds_test}

    pred_vots = [predictTastes(model, allds[name].z)['vots'] for name in ["train", "dev", "test"]]
    true_vots = [allds[name].vots for name in ["train", "dev", "test"]]

    result['vot_rmse'] = RMSE(pred_vots, true_vots).item()
    result['vot_mabse'] = ABSE(pred_vots, true_vots).item()
    result['vot_re'] = RE(pred_vots, true_vots).item()

    pred_vowts = [predictTastes(model, allds[name].z)['vowts'] for name in ["train", "dev", "test"]]
    true_vowts = [allds[name].vowts for name in ["train", "dev", "test"]]

    result['vowt_rmse'] = RMSE(pred_vowts, true_vowts).item()
    result['vowt_mabse'] = ABSE(pred_vowts, true_vowts).item()
    result['vowt_re'] = RE(pred_vowts, true_vowts).item()

    # 3. Tastes (vots, vowts) on simulated z: unit is dollar per hour!
    input_z = copy.deepcopy(dic_z)
    sim_pred_vots, sim_true_vots, rmse, mabse, re = error_of_vot(model, dic_z, input_z, ds_train.params)
    result['sim_vot_rmse'] = rmse
    result['sim_vot_mabse'] = mabse
    result['sim_vot_re'] = re

    sim_pred_vowts, sim_true_vowts, rmse, mabse, re = error_of_vowt(model, dic_z, input_z, ds_train.params)
    result['sim_vowt_rmse'] = rmse
    result['sim_vowt_mabse'] = mabse
    result['sim_vowt_re'] = re

    # 4. Coefficients from regression
    coefs_asc1 = model.getParams()[-1][0].item()
    coefs_time, coefs_wait = regress(model, dic_z, dic_z_zall)
    coefs_pred = [coefs_asc1] + coefs_time + coefs_wait

    coefs_asc1_true = ds_train.asc1
    coefs_time_true = ds_train.coefs_time
    coefs_wait_true = ds_train.coefs_wait
    coefs_true = [coefs_asc1_true] + coefs_time_true + coefs_wait_true

    coefs_pred = torch.Tensor(coefs_pred)
    coefs_true = torch.Tensor(coefs_true)

    rmse = RMSE_vector(coefs_pred, coefs_true)
    mabse = ABSE_vector(coefs_pred, coefs_true)
    re = RE_vector(coefs_pred, coefs_true)

    result['coef_rmse'] = rmse
    result['coef_mabse'] = mabse
    result['coef_re'] = re
    result['coef'] = coefs_pred
    result['coef_true'] = coefs_true
    
    pred_vots_test = predictTastes(model, allds['test'].z)['vots'] 
    pred_vowts_test = predictTastes(model, allds['test'].z)['vowts']

    return result, pred_vots_test, pred_vowts_test

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='TasteNet-MNL (A Toy Example)')
    parser.add_argument("--data_dir", type=str, default="toy_data")
    parser.add_argument("--data_file", type=str, default="data_10k.pkl", help="data dictionary in pickle format")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--scenario", type=str)
    
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    
    # Training params
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='maximum number of epochs to train (default: 100)')
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--nll_tol", type=float, default=0.0001, help="tolerance for nll convergence")
    
    parser.add_argument('--kfold', default=True,
                        help='K fold cross validation')
    
    args = parser.parse_args()
    
    cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    
    # load data
    data = pickle.load(open(os.path.join(args.data_dir, args.data_file), "rb"))
    
    if (args.kfold == False):
        ds_train = ChoiceDataset(data['train'])
        ds_dev = ChoiceDataset(data['dev'])
        ds_test = ChoiceDataset(data['test'])
        
        data_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
        data_dev = DataLoader(ds_dev, batch_size=args.batch_size, shuffle=False, num_workers=0)
        data_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
        # Set hyperparam_candidates
        hyperparam_candidates = dict(
                hidden_size=[60],
                weight_decay=[0.001],
                activation=["relu"],
                transform=["relu"],
                separate=[False],
                no=range(100) # number of runs with different random initialization
            )
        
        hyperparam_sets = [dict(zip(hyperparam_candidates.keys(), values))
                            for values
                            in itertools.product(*hyperparam_candidates.values())]
        print(f"{len(hyperparam_sets)} set of params")
        print(hyperparam_sets)
        
        
        result_all = []
        model_all = []
        
        mean_vots_list = []
        mean_vowts_list = []
        var_vots_list = []
        var_vowts_list = []
        accuracy_list = []
        nll_list = []
        for params in hyperparam_sets:
            result, model = one_run(data_train, data_dev, args, params)
            result_all.append(result)
            model_all.append(model)

        
        evals_name = f"{args.scenario}_evals.pkl"
        models_name = f"{args.scenario}_models.pkl"
        args_name = f"{args.scenario}_args.pkl"
        hyper_name = f"{args.scenario}_hyper.pkl"
        
        pickle.dump(result_all, open(os.path.join(args.output_dir, evals_name), "wb"))
        pickle.dump(model_all, open(os.path.join(args.output_dir, models_name), "wb"))
        pickle.dump(hyperparam_sets, open(os.path.join(args.output_dir, hyper_name), "wb"))
        pickle.dump(args, open(os.path.join(args.output_dir, args_name), "wb"))


    else:
        # Set hyperparam_candidates
        hyperparam_candidates = dict(
                hidden_size=[60],
                weight_decay=[0.001],
                activation=["relu"],
                transform=["relu"],
                separate=[False],
                no=range(1), #one random initialization
                seed=[42]
            )
        
        hyperparam_sets = [dict(zip(hyperparam_candidates.keys(), values))
                    for values
                    in itertools.product(*hyperparam_candidates.values())]
        print(f"{len(hyperparam_sets)} set of params")
        print(hyperparam_sets)
        
        mean_vots_list = []
        mean_vowts_list = []
        var_vots_list = []
        var_vowts_list = []

        fold_acc = []
        fold_nlls_train = []
        fold_nlls_test = []
        fold_vots = []
        fold_vowts = []

        ds_dev = ChoiceDataset(data['dev'])
        data_dev = DataLoader(ds_dev, batch_size=args.batch_size, shuffle=False, num_workers=0)

        for n in range(5):
            data = csv_to_dict(dir = "toy_data/CV/", filename="10k_split"+str(n)+".csv", dev_av = False, save = False)
            ds_train = ChoiceDataset(data['train'])
            ds_test = ChoiceDataset(data['test'])

            data_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
            data_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=0)

            for params in hyperparam_sets:
                result, _, pred_vots, pred_vowts, mean_vots, mean_vowts, var_vots, var_vowts = one_run(data_train, data_dev, args, params)
                nll_train = result['train']['nll']
                nll_test = result['test']['nll']
                accuracy = result['test']['acc']*100
                
                fold_vots.append(np.array(pred_vots, copy=True))
                fold_vowts.append(np.array(pred_vowts, copy=True))

                mean_vots_list.append(mean_vots)
                mean_vowts_list.append(mean_vowts)
                var_vots_list.append(var_vots)
                var_vowts_list.append(var_vowts)

                fold_acc.append(accuracy)
                fold_nlls_train.append(nll_train)
                fold_nlls_test.append(nll_test)

        print("\n=== CV Results ===")
        print(
            f"CV NLL (train): "
            f"mean={np.mean(fold_nlls_train):.5f}, "
            f"std={np.std(fold_nlls_train):.5f}"
        )

        print(
        f"CV NLL (test): "
        f"mean={np.mean(fold_nlls_test):.5f}, "
        f"std={np.std(fold_nlls_test):.5f}"
        )
      

         # ======= Save Results ====== #
        save_dir = "results"  
        os.makedirs(save_dir, exist_ok=True)
        summary_cv5 = {
        "vot": fold_vots,
        "vowt": fold_vowts,
        "accuracy": fold_acc,
        "ll_train": fold_nlls_train,
        "ll_test": fold_nlls_test
        }
        with open(save_dir + "/" +'summary_cv5.pkl', 'wb') as f:
            pickle.dump(summary_cv5, f)
      
        # === Convert to arrays ===
        mean_vots_arr = np.stack(mean_vots_list, axis=0)# shape (runs, r, c)
        mean_vowts_arr = np.stack(mean_vowts_list, axis=0)
        var_vots_arr = np.stack(var_vots_list, axis=0)
        var_vowts_arr = np.stack(var_vowts_list, axis=0)


        print("--Averages over runs:")
        print(f"Mean E[vot]: {np.mean(mean_vots_arr):.6f}")
        print(f"Mean E[vowt]: {np.mean(mean_vowts_arr):.6f}")
        print(f"Mean var[vot]: {np.mean(var_vots_arr):.6f}")
        print(f"Mean var[vowt]: {np.mean(var_vowts_arr):.6f}")
        print(f"Mean accuracy: {np.mean(fold_acc):.6f}")
        