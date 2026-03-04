import pandas as pd
import os
import traceback

task_list = ['TYK2'] # 'ESOL','Mutagenicity', 'hERG','drd2','BBBP_pzl'
sub_type_list = ['brics_combine','brics' ,'mol'   ]  # 'brics_combine','brics' ,'mol' 

for task_name in task_list:
    for sub_type in sub_type_list:
        try:
            print(f'Processing {task_name} {sub_type}...')
            # 将训练集，验证集，测试集数据合并
            result_summary = pd.DataFrame()
            for i in range(10):
                seed = i + 1
                # 读取文件
                try:
                    result_train = pd.read_csv(
                        f'/HOME/scz4306/run/SME/optimization/prediction/{sub_type}/{task_name}_{sub_type}_{seed}_train_prediction.csv'
                    )
                    result_val = pd.read_csv(
                        f'/HOME/scz4306/run/SME/optimization/prediction/{sub_type}/{task_name}_{sub_type}_{seed}_val_prediction.csv'
                    )
                    result_test = pd.read_csv(
                        f'/HOME/scz4306/run/SME/optimization/prediction/{sub_type}/{task_name}_{sub_type}_{seed}_test_prediction.csv'
                    )
                   
                except FileNotFoundError as e:
                    print(f"File not found for seed {seed}: {e}")
                    raise
                except Exception as e:
                    print(f"Error reading files for seed {seed}: {e}")
                    raise
                
                # 合并数据并打印列
                print(f"Columns in result_train (seed {seed}): {result_train.columns}")
                print(f"Columns in result_val (seed {seed}): {result_val.columns}")
                print(f"Columns in result_test (seed {seed}): {result_test.columns}")
                group_list = ['training'] * len(result_train) + ['val'] * len(result_val) + ['test'] * len(result_test)
                result = pd.concat([result_train, result_val, result_test], axis=0)
                print(f"Columns in merged result (seed {seed}): {result.columns}")


                
                result['group'] = group_list
                if sub_type == 'mol':
                    result.sort_values(by='smiles', inplace=True)

                # 初始化结果汇总
                if seed == 1:
                    result_summary['smiles'] = result['smiles']
                    result_summary['label'] = result['label']
                    result_summary['sub_name'] = result['sub_name']
                    result_summary['group'] = result['group']


                    result_summary['antecedents'] = result['antecedents']
                    result_summary['consequents'] = result['consequents']
                    result_summary['support'] = result['support']
                    result_summary['confidence'] = result['confidence']
                    result_summary['lift'] = result['lift']
                    result_summary['leverage'] = result['leverage']
                    result_summary['conviction'] = result['conviction']
                    result_summary['zhangs_metric'] = result['zhangs_metric']
                    result_summary['brics_indices'] = result['brics_indices']
                    result_summary['brics_comb_indices'] = result['brics_comb_indices']
                    result_summary['ante_indices'] = result['ante_indices']
                    result_summary['ante_comb_indices'] = result['ante_comb_indices']
                    result_summary['conse_indices'] = result['conse_indices']
                    result_summary['conse_comb_indices'] = result['conse_comb_indices']
                    result_summary['ante_conse_indices'] = result['ante_conse_indices']
                    result_summary['smarts'] = result['smarts']


                    result_summary[f'pred_{seed}'] = result['pred'].tolist()
                else:
                    result_summary[f'pred_{seed}'] = result['pred'].tolist()
            
                    if 'index' in result.columns and 'smarts' in result.columns:
                        result_summary['index'] = result['index']
                        result_summary['smarts'] = result['smarts']

                    
            # 计算均值和标准差
            pred_columns = [f'pred_{i + 1}' for i in range(10)]
            data_pred = result_summary[pred_columns]
            result_summary['pred_mean'] = data_pred.mean(axis=1)
            result_summary['pred_std'] = data_pred.std(axis=1)
            
            # 保存结果
            output_dir = '/HOME/scz4306/run/SME/optimization/prediction/summary/'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_file = f'{output_dir}{task_name}_{sub_type}_prediction_summary.csv'
            # output_file = f'{output_dir}{task_name}_{sub_type}_T_20_mol_generator_prediction_summary.csv'
            result_summary.to_csv(output_file, index=False)
            print(f'{task_name} {sub_type} sum succeed. Results saved to {output_file}')
        
        except Exception as e:
            print(f'{task_name} {sub_type} sum failed. Error details:')
            traceback.print_exc()



