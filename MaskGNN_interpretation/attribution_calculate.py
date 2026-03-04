import pandas as pd
import numpy as np


import pandas as pd
import os

# for task_name in ['ESOL','Mutagenicity', 'hERG','drd2']:
for task_name in ['TYK2']:
    # for sub_type in ['fg', 'murcko', 'brics', 'brics_emerge', 'murcko_emerge']:
    for sub_type in [ 'brics','brics_combine']:#'brics_combine', 
        attribution_result = pd.DataFrame()
        print('{} {}'.format(task_name, sub_type))
        result_sub = pd.read_csv('/HOME/scz4306/run/SME/optimization/prediction/summary/{}_{}_prediction_summary.csv'.format(task_name, sub_type))
        result_mol = pd.read_csv('/HOME/scz4306/run/SME/optimization/prediction/summary/{}_{}_prediction_summary.csv'.format(task_name, 'mol'))

        # result_sub = pd.read_csv('/HOME/scz4306/run/SME/optimization/prediction/summary/{}_{}_B_20_mol_generator_prediction_summary.csv'.format(task_name, sub_type))
        # result_mol = pd.read_csv('/HOME/scz4306/run/SME/optimization/prediction/summary/{}_{}_B_20_mol_generator_prediction_summary.csv'.format(task_name, 'mol'))
        mol_pred_mean_list_for_sub = [result_mol[result_mol['smiles'] == smi]['pred_mean'].tolist()[0] for smi in
                                 result_sub['smiles'].tolist()]
        mol_pred_std_list_for_sub = [result_mol[result_mol['smiles'] == smi]['pred_std'].tolist()[0] for smi in
                                 result_sub['smiles'].tolist()]
        attribution_result['smiles'] = result_sub['smiles']
        attribution_result['label'] = result_sub['label']
        attribution_result['sub_name'] = result_sub['sub_name']
        attribution_result['group'] = result_sub['group']
        
        attribution_result['antecedents'] = result_sub['antecedents']
        attribution_result['consequents'] = result_sub['consequents']
        attribution_result['support'] = result_sub['support']
        attribution_result['confidence'] = result_sub['confidence']
        attribution_result['lift'] = result_sub['lift']
        attribution_result['leverage'] = result_sub['leverage']
        attribution_result['conviction'] = result_sub['conviction']
        attribution_result['zhangs_metric'] = result_sub['zhangs_metric']
        attribution_result['brics_indices'] = result_sub['brics_indices']
        attribution_result['brics_comb_indices'] = result_sub['brics_comb_indices']
        attribution_result['ante_indices'] = result_sub['ante_indices']
        attribution_result['ante_comb_indices'] = result_sub['ante_comb_indices']
        attribution_result['conse_indices'] = result_sub['conse_indices']
        attribution_result['conse_comb_indices'] = result_sub['conse_comb_indices']
        attribution_result['ante_conse_indices'] = result_sub['ante_conse_indices']

        attribution_result['sub_pred_mean'] = result_sub['pred_mean']
        attribution_result['sub_pred_std'] = result_sub['pred_std']
        attribution_result['mol_pred_mean'] = mol_pred_mean_list_for_sub
        attribution_result['mol_pred_std'] = mol_pred_std_list_for_sub
        sub_pred_std_list = result_sub['pred_std']
        attribution_result['attribution'] = attribution_result['mol_pred_mean'] - attribution_result['sub_pred_mean']
        attribution_result['attribution_normalized'] = (np.exp(attribution_result['attribution'].values) - np.exp(
            -attribution_result['attribution'].values)) / (np.exp(attribution_result['attribution'].values) + np.exp(
            -attribution_result['attribution'].values))
        if 'index' in result_sub.columns and 'smarts' in result_sub.columns:
            attribution_result['index'] = result_sub['index']

        attribution_result['smarts'] = result_sub['smarts']
        dirs = '/HOME/scz4306/run/SME/optimization/prediction/attribution/'
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        # attribution_result.to_csv('/HOME/scz4306/run/SME/optimization/prediction/attribution/{}_{}_B_20_mol_generator_attribution_summary.csv'.format(task_name, sub_type), index=False)
        attribution_result.to_csv('/HOME/scz4306/run/SME/optimization/prediction/attribution/{}_{}_attribution_summary.csv'.format(task_name, sub_type), index=False)

