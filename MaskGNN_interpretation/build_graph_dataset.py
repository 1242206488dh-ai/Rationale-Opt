from build_data import built_mol_graph_data_and_save
import argparse
import pandas as pd
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


task_list = ['TYK2']#,'BBBP','Mutagenicity','ESOL', 'hERG'
for task in task_list:
    input_csv = '/HOME/scz4306/run/SME/optimization/data/subnear_association_rules_withsmiles/' + task + '_subnear_association_rules_withsmiles.csv' 
    output_g_path = '/HOME/scz4306/run/SME/optimization/data/subnear_association_rules_withsmiles/' + task + '.bin'
    output_g_group_path = '/HOME/scz4306/run/SME/optimization/data/subnear_association_rules_withsmiles/' + task + '_group.csv'

    output_g_for_brics_path = '/HOME/scz4306/run/SME/optimization/data/subnear_association_rules_withsmiles/' + task + '_for_brics.bin'
    output_g_group_for_brics_path = '/HOME/scz4306/run/SME/optimization/data/subnear_association_rules_withsmiles/' + task + '_group_for_brics.csv'
    output_g_smask_for_brics_path = '/HOME/scz4306/run/SME/optimization/data/subnear_association_rules_withsmiles/' + task + '_smask_for_brics.npy'

    output_g_for_brics_combine_path = '/HOME/scz4306/run/SME/optimization/data/subnear_association_rules_withsmiles/' + task + '_for_brics_combine.bin'
    output_g_group_for_brics_combine_path = '/HOME/scz4306/run/SME/optimization/data/subnear_association_rules_withsmiles/' + task + '_group_for_brics_combine.csv'
    output_g_smask_for_brics_combine_path = '/HOME/scz4306/run/SME/optimization/data/subnear_association_rules_withsmiles/' + task + '_smask_for_brics_combine.npy'

    output_g_for_murcko_path = '/public3/home/scg3876/SME/optimization//data/graph_data/' + task + '_for_murcko.bin'
    # output_g_group_for_murcko_path = '/public3/home/scg3876/SME/optimization//data/graph_data/' + task + '_group_for_murcko.csv'
    output_g_smask_for_murcko_path = '/public3/home/scg3876/SME/optimization//data/graph_data/' + task + '_smask_for_murcko.npy'

    output_g_for_murcko_emerge_path = '/public3/home/scg3876/SME/optimization//data/graph_data/' + task + '_for_murcko_emerge.bin'
    # output_g_group_for_murcko_emerge_path = '/public3/home/scg3876/SME/optimization//data/graph_data/' + task + '_group_for_murcko_emerge.csv'
    output_g_smask_for_murcko_emerge_path = '/public3/home/scg3876/SME/optimization//data/graph_data/' + task + '_smask_for_murcko_emerge.npy'

    output_g_for_fg_path = '/public3/home/scg3876/SME/optimization//data/graph_data/' + task + '_for_fg.bin'
    output_g_group_for_fg_path = '/public3/home/scg3876/SME/optimization//data/graph_data/' + task + '_group_for_fg.csv'
    output_g_smask_for_fg_path = '/public3/home/scg3876/SME/optimization//data/graph_data/' + task + '_smask_for_fg.npy'
    

    built_mol_graph_data_and_save(
        task_name=task,
        origin_data_path=input_csv,
        labels_name='label',
        save_g_path=output_g_path,
        save_g_group_path=output_g_group_path,
    
        save_g_for_brics_smarts_path=output_g_for_brics_path,
        save_g_smask_for_brics_smarts_path=output_g_smask_for_brics_path,
        save_g_group_for_brics_smarts_path=output_g_group_for_brics_path,
    
        save_g_for_brics_combine_path=output_g_for_brics_combine_path,
        save_g_smask_for_brics_combine_path=output_g_smask_for_brics_combine_path,
        save_g_group_for_brics_combine_path=output_g_group_for_brics_combine_path,
    
        # save_g_for_murcko_path=output_g_for_murcko_path,
        # save_g_smask_for_murcko_path=output_g_smask_for_murcko_path,
        # save_g_group_for_murcko_path=output_g_group_for_murcko_path,
    
        # save_g_for_murcko_emerge_path=output_g_for_murcko_emerge_path,
        # save_g_smask_for_murcko_emerge_path=output_g_smask_for_murcko_emerge_path,
        # save_g_group_for_murcko_emerge_path=output_g_group_for_murcko_emerge_path,
    
        # save_g_for_fg_path=output_g_for_fg_path,
        # save_g_smask_for_fg_path=output_g_smask_for_fg_path,
        # save_g_group_for_fg_path=output_g_group_for_fg_path
    )

