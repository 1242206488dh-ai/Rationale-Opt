import numpy as np
import build_data
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from maskgnn import collate_molgraphs, EarlyStopping, run_a_train_epoch, \
    run_an_eval_epoch, set_random_seed, RGCN, pos_weight
import pickle as pkl
import os
import time

def SMEG_explain_for_sar_optimization(seed, task_name, model_name='None', rgcn_hidden_feats=[64, 64, 64], ffn_hidden_feats=128,
               lr=0.0003, classification=True):
    args = {}
    args['device'] = "cpu"
    args['node_data_field'] = 'node'
    args['edge_data_field'] = 'edge'
    args['substructure_mask'] = 'smask'
    args['classification'] = classification
    # model parameter
    args['num_epochs'] = 500
    args['patience'] = 30
    args['batch_size'] = 128
    args['mode'] = 'higher'
    args['in_feats'] = 40
    args['rgcn_hidden_feats'] = rgcn_hidden_feats
    args['ffn_hidden_feats'] = ffn_hidden_feats
    args['rgcn_drop_out'] = 0
    args['ffn_drop_out'] = 0
    args['lr'] = lr
    args['loop'] = True
    # task name (model name)
    args['task_name'] = task_name  # change
    args['data_name'] = task_name  # change
    args['bin_path'] = '/HOME/scz4306/run/SME/optimization/data/graph_data/' + args['data_name'] + '.bin'
    args['group_path'] = '/HOME/scz4306/run/SME/optimization/data/graph_data/' + args['data_name'] + '_group.csv'
    # args['bin_path'] = '/HOME/scz4306/run/SME/optimization/data/subnear_association_rules_withsmiles/' + args['data_name'] + '.bin'
    # args['group_path'] = '/HOME/scz4306/run/SME/optimization/data/subnear_association_rules_withsmiles/' + args['data_name'] + '_group.csv'
    args['seed'] = seed
    
    print('***************************************************************************************************')
    print('{} seed {}'.format(args['task_name'], args['seed']))
    print('***************************************************************************************************')
    train_set, val_set, test_set, task_number = build_data.load_graph_from_csv_bin_for_splited(
        bin_path=args['bin_path'],
        group_path=args['group_path'],
        classification=args['classification'],
        random_shuffle=False
    )
    print("Molecule graph is loaded!")

    train_loader = DataLoader(dataset=train_set,
                              batch_size=args['batch_size'],
                              collate_fn=collate_molgraphs)
    
    val_loader = DataLoader(dataset=val_set,
                            batch_size=args['batch_size'],
                            collate_fn=collate_molgraphs)
    
    test_loader = DataLoader(dataset=test_set,
                             batch_size=args['batch_size'],
                             collate_fn=collate_molgraphs)
    if args['classification']:
        loss_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    else:
        loss_criterion = torch.nn.MSELoss(reduction='none')
    model = RGCN(ffn_hidden_feats=args['ffn_hidden_feats'],
                 ffn_dropout=args['ffn_drop_out'],
                 rgcn_node_feats=args['in_feats'], rgcn_hidden_feats=args['rgcn_hidden_feats'],
                 rgcn_drop_out=args['rgcn_drop_out'],
                 classification=args['classification'])
    stopper = EarlyStopping(patience=args['patience'], task_name=model_name + '_' + str(seed + 1),
                            mode=args['mode'])
    # stopper = EarlyStopping(patience=args['patience'], task_name='BBBP' + '_' + str(seed + 1),
    #                         mode=args['mode'])
    model.to(args['device'])
    stopper.load_checkpoint(model)
    pred_name = 'mol_{}'.format(seed + 1)
    stop_train_list, _ = run_an_eval_epoch(args, model, train_loader, loss_criterion,
                                           out_path='/HOME/scz4306/run/SME/optimization/prediction/mol/' + args['task_name'] + '_' + pred_name + '_train')
    stop_val_list, _ = run_an_eval_epoch(args, model, val_loader, loss_criterion,
                                          out_path='/HOME/scz4306/run/SME/optimization/prediction/mol/' + args['task_name'] + '_' + pred_name + '_val')

    stop_test_list, _ = run_an_eval_epoch(args, model, test_loader, loss_criterion,
                                          out_path='/HOME/scz4306/run/SME/optimization/prediction/mol/' + args['task_name'] + '_' + pred_name + '_test')
    print('Mask prediction is generated!')



for task in ['TYK2']:#'drd2'
    with open('/HOME/scz4306/run/SME/optimization/result/hyperparameter_{}.pkl'.format(task), 'rb') as f:
    # with open('/HOME/scz4306/run/SME/optimization/result/hyperparameter_BBBP.pkl', 'rb') as f:
        hyperparameter = pkl.load(f)
    for i in range(10):
        SMEG_explain_for_sar_optimization(seed=i, task_name='{}'.format(task), model_name=task,
                                          rgcn_hidden_feats=hyperparameter['rgcn_hidden_feats'],
                                          ffn_hidden_feats=hyperparameter['ffn_hidden_feats'],
                                          lr=hyperparameter['lr'], classification=hyperparameter['classification'])








