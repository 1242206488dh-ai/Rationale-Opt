import pandas as pd
import ast

def get_single_attribution(smarts, smiles, antecedents, consequents, single_indices, df_single):
    """
    在 df_single 中查找匹配的 smiles, antecedents, consequents 和 single_indices， 
    然后获取对应 smarts 的 attribution_normalized 值。
    """
    # 查找匹配的行，按 smiles, antecedents, consequents 过滤
    match_rows = df_single[(df_single['smiles'] == smiles) & 
                           (df_single['antecedents'] == antecedents) & 
                           (df_single['consequents'] == consequents)]
    if match_rows.empty:
        print(f"❌ No matching row found for smiles: {smiles}, antecedents: {antecedents}, consequents: {consequents}")
        return None

    # 逐行查找 smarts 中是否包含目标子结构
    for _, row in match_rows.iterrows():
        # 确保 smarts 列是列表类型
        smarts_list = eval(row['smarts']) if isinstance(row['smarts'], str) else row['smarts']
        
        # 将 single_indices 转换为列表类型
        if isinstance(single_indices, str):
            single_indices = eval(single_indices)  # 将字符串转换为列表
        
        # 增加调试，检查 smarts_list 和 single_indices
        print(f"Debugging comparison between smarts_list and single_indices")
        print(f"smarts_list (type {type(smarts_list)}): {repr(smarts_list)}")
        print(f"single_indices (type {type(single_indices)}): {repr(single_indices)}")
        
        # 比较前，确保两个列表的顺序一致
        if sorted(smarts_list) == sorted(single_indices):  # 排序后再比较
            print("✔ Match found!")
            return row['attribution_normalized']  # 返回匹配的 attribution_normalized

    print(f"❌ No matching smarts found for smiles: {smiles}, antecedents: {antecedents}, consequents: {consequents}, single_indices: {single_indices}")
    return None

def process_task(task_name):
    print(f"🔍 Processing task: {task_name}")

    combine_file = f'/HOME/scz4306/run/SME/optimization/prediction/attribution/{task_name}_brics_combine_attribution_summary.csv'
    single_file = f'/HOME/scz4306/run/SME/optimization/prediction/attribution/{task_name}_brics_attribution_summary.csv'
    output_file = f'/HOME/scz4306/run/SME/optimization/prediction/gain/{task_name}_brics_gain_analysis.csv'
    # combine_file = f'/HOME/scz4306/run/SME/optimization/prediction/attribution/{task_name}_brics_combine_T_20_mol_generator_attribution_summary.csv'
    # single_file = f'/HOME/scz4306/run/SME/optimization/prediction/attribution/{task_name}_brics_T_20_mol_generator_attribution_summary.csv'
    # output_file = f'/HOME/scz4306/run/SME/optimization/prediction/gain/{task_name}_brics_T_20_mol_generator_gain_analysis.csv'


    df_combine = pd.read_csv(combine_file)
    df_single = pd.read_csv(single_file).drop_duplicates()

    # 解析为列表
    df_single['smarts'] = df_single['smarts'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    
    # 解析 ante_conse_indices 为列表形式
    df_combine['ante_conse_indices'] = df_combine['ante_conse_indices'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # 合并 ante_comb_indices 和 conse_comb_indices
    df_combine['brics_indices'] = df_combine.apply(lambda row: [row['ante_comb_indices'], row['conse_comb_indices']], axis=1)

    # 打印合并后的 brics_indices，进行调试
    print(f"Debugging brics_indices for task {task_name}:")
    print(df_combine[['smiles', 'brics_indices']].head())  # 打印前几行以验证

    results = []
    for idx, row in df_combine.iterrows():
        smiles = row['smiles']
        combined_attr = row['attribution']
        brics_indices = row['brics_indices']

        # 假设 antecedents 和 consequents 存在于 df_combine 中
        antecedents = row['antecedents']
        consequents = row['consequents']

        single_attrs = []
        indices_list = []

        for single_indices in brics_indices:
            single_attr = get_single_attribution(smarts=None,  # 此参数暂时不需要
                                                smiles=smiles,
                                                antecedents=antecedents,
                                                consequents=consequents,
                                                single_indices=single_indices,
                                                df_single=df_single)
            single_attrs.append(single_attr)
            indices_list.append(single_indices)

        # 计算增益
        if None in single_attrs or len(single_attrs) < 2:
            gain_AB = gain_0 = gain_1 = None
        else:
            sum_A_B = sum(single_attrs)
            gain_AB = ((combined_attr - sum_A_B) / sum_A_B) * 100
            # gain_0 = ((combined_attr - single_attrs[0]) / single_attrs[0]) * 100
            # gain_1 = ((combined_attr - single_attrs[1]) / single_attrs[1]) * 100

        result_data = {
            'smiles': smiles,
            'combined_indexes': brics_indices,
            'combined_attr': combined_attr,
            'single_attrs': single_attrs,
            'gain_AB(%)': gain_AB,
            # 'gain_0(%)': gain_0,
            # 'gain_1(%)': gain_1,
            'index': indices_list,
            # 添加额外的字段到输出文件
            'label': row['label'],
            'sub_name': row['sub_name'],
            # 'group': row['group'],
            'antecedents': row['antecedents'],
            'consequents': row['consequents'],
            'support': row['support'],
            'confidence': row['confidence'],
            'lift': row['lift'],
            'leverage': row['leverage'],
            'conviction': row['conviction'],
            'zhangs_metric': row['zhangs_metric']
        }
        results.append(result_data)

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)
    print(f"✅ Results saved to: {output_file}")

# 执行任务
task_name_list = [ 'TYK2']#'ESOL','Mutagenicity', 'hERG','drd2'
for task_name in task_name_list:
    process_task(task_name)

print("🎉 All tasks processed successfully!")
