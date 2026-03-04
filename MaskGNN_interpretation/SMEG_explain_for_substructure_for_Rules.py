import numpy as np
from build_data import *
import pandas as pd
import torch
from torch.utils.data import DataLoader
from maskgnn import collate_molgraphs, EarlyStopping, run_a_train_epoch, \
    run_an_eval_epoch, set_random_seed, RGCN, pos_weight
import pickle as pkl

# task_name_list = ['ESOL', 'hERG','Mutagenicity', 'BBBP','drd2']#
task_name_list = ['TYK2']#
# fix parameters of model
def SMEG_explain_for_substructure(seed, task_name, rgcn_hidden_feats=[256, 256, 256], ffn_hidden_feats=128,
                                  lr=0.001, classification=True, sub_type='brics_combine'):#'brics_combine'
    args = {}
    args['device'] = "cuda"
    args['node_data_field'] = 'node'
    args['edge_data_field'] = 'edge'
    args['substructure_mask'] = 'smask'
    args['classification'] = classification
    # model parameter
    args['num_epochs'] = 500
    args['patience'] = 30
    args['batch_size'] = 256
    args['mode'] = 'higher'
    args['in_feats'] = 40
    args['rgcn_hidden_feats'] = rgcn_hidden_feats
    args['ffn_hidden_feats'] = ffn_hidden_feats
    args['rgcn_drop_out'] = 0
    args['ffn_drop_out'] = 0
    args['lr'] = lr
    args['loop'] = True
    # task name (model name)
    args['task_name'] = task_name
    args['data_name'] = task_name
    args['bin_path'] = '/HOME/scz4306/run/SME/optimization/data/subnear_association_rules_withsmiles/{}_for_{}.bin'.format(args['data_name'], sub_type)
    args['group_path'] = '/HOME/scz4306/run/SME/optimization/data/subnear_association_rules_withsmiles/{}_group_for_{}.csv'.format(args['data_name'], sub_type)
    args['smask_path'] = '/HOME/scz4306/run/SME/optimization/data/subnear_association_rules_withsmiles/{}_smask_for_{}.npy'.format(args['data_name'], sub_type)
    # args['bin_path'] = '/HOME/scz4306/run/SME/optimization/data/Rules_20_mol_generator/Top_rules_20_mol_generator_for_{}.bin'.format(sub_type)
    # args['group_path'] = '/HOME/scz4306/run/SME/optimization/data/Rules_20_mol_generator/Top_rules_20_mol_generator_group_for_{}.csv'.format( sub_type)
    # args['smask_path'] = '/HOME/scz4306/run/SME/optimization/data/Rules_20_mol_generator/Top_rules_20_mol_generator_smask_for_{}.npy'.format( sub_type)
    args['seed'] = seed

    print('***************************************************************************************************')
    print('{}, seed {}, substructure type {}'.format(args['task_name'], args['seed']+1, sub_type))
    print('***************************************************************************************************')
    
    # Load the entire dataset
    train_set, val_set, test_set, T_20_mol_generator_set,B_20_mol_generator_set,task_number = load_graph_from_csv_bin_for_splited(
        bin_path=args['bin_path'],
        group_path=args['group_path'],
        smask_path=args['smask_path'],
        classification=args['classification'],
        random_shuffle=False
    )
    print("Molecule graph is loaded!")



    # ✅ 调试: 确保 train_set, val_set, test_set 不是空的
    if len(train_set) == 0:
        print("❗Warning: train_set is empty!")
    if len(val_set) == 0:
        print("❗Warning: val_set is empty!")
    if len(test_set) == 0:
        print("❗Warning: test_set is empty!")
    if len(T_20_mol_generator_set) == 0:
        print("❗Warning: T_20_mol_generator_set is empty!")
    if len(B_20_mol_generator_set) == 0:
        print("❗Warning: B_20_mol_generator_set is empty!")


      # Use the test set for evaluation
    train_loader = DataLoader(dataset=train_set,
                              batch_size=args['batch_size'],
                              collate_fn=collate_molgraphs)
            
    
    val_loader = DataLoader(dataset=val_set,
                            batch_size=args['batch_size'],
                            collate_fn=collate_molgraphs)
    
    test_loader = DataLoader(dataset=test_set,
                             batch_size=args['batch_size'],
                             collate_fn=collate_molgraphs)
    
    T_20_mol_generator_loader = DataLoader(dataset=T_20_mol_generator_set,
                            batch_size=args['batch_size'],
                            collate_fn=collate_molgraphs)
    

    B_20_mol_generator_loader = DataLoader(dataset=B_20_mol_generator_set,
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
                 
    stopper = EarlyStopping(patience=args['patience'], task_name=args['task_name'] + '_' + str(seed + 1),
                            mode=args['mode'])
    # stopper = EarlyStopping(patience=args['patience'], task_name= 'BBBP_' + str(seed + 1),
    #                     mode=args['mode'])
    model.to(args['device'])
    stopper.load_checkpoint(model)

    pred_name = '{}_{}_{}'.format(args['task_name'], sub_type, seed + 1)
    stop_train_list, _ = run_an_eval_epoch(args, model, train_loader, loss_criterion,
                                           out_path='/HOME/scz4306/run/SME/optimization/prediction/{}/{}_train'.format(sub_type, pred_name))
    stop_val_list, _ = run_an_eval_epoch(args, model, val_loader, loss_criterion,
                                          out_path='/HOME/scz4306/run/SME/optimization/prediction/{}/{}_val'.format(sub_type, pred_name))
    stop_test_list, _ = run_an_eval_epoch(args, model, test_loader, loss_criterion,
                                          out_path='/HOME/scz4306/run/SME/optimization/prediction/{}/{}_test'.format(sub_type, pred_name))
    # stop_T_20_mol_generator_list, _ = run_an_eval_epoch(args, model, T_20_mol_generator_loader, loss_criterion,
    #                                       out_path='/HOME/scz4306/run/SME/optimization/prediction/{}/{}_T_20_mol_generator'.format(sub_type, pred_name))
    # stop_B_20_mol_generator_list, _ = run_an_eval_epoch(args, model, B_20_mol_generator_loader, loss_criterion,
    #                                       out_path='/HOME/scz4306/run/SME/optimization/prediction/{}/{}_B_20_mol_generator'.format(sub_type, pred_name))
    print('Mask prediction is generated!')


# Iterate through subtypes for evaluation
for task in task_name_list:
    # for sub_type in ['fg', 'brics', 'murcko']:
    for sub_type in ['brics','brics_combine']:#
        with open('/HOME/scz4306/run/SME/optimization/result/hyperparameter_{}.pkl'.format(task), 'rb') as f:
        # with open('/HOME/scz4306/run/SME/optimization/result/hyperparameter_BBBP.pkl'.format(task), 'rb') as f:
            hyperparameter = pkl.load(f)
        for i in range(10):
            SMEG_explain_for_substructure(seed=i, task_name=task, 
                                        rgcn_hidden_feats=hyperparameter['rgcn_hidden_feats'],
                                        ffn_hidden_feats=hyperparameter['ffn_hidden_feats'],
                                        lr=hyperparameter['lr'], classification=hyperparameter['classification'],
                                        sub_type=sub_type)
