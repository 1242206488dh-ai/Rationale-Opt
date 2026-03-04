import pandas as pd
import numpy as np
import os

def calculate_and_append_run_success_rate():
    """
    This script reads a detailed evaluation trajectory file, calculates a new 
    'per-run' success rate, and appends it to the summary file.

    New Success Rate Logic:
    1. A 'run' is defined by a unique (original_smiles, run_index) pair.
    2. A 'run' is considered successful if ANY molecule generated within that 
       run's trajectory meets the final success criteria.
    3. The final metric is the average success rate of runs, calculated only 
       for 'hard' starting molecules.
    """
    
    # --- 1. Configuration: Define file paths ---
    # !! IMPORTANT !!: Please verify these paths match your generated files.
    base_result_path = ''
    
    detailed_file_path = os.path.join(base_result_path, 'evaluation_detailed_full_trajectory.csv')
    summary_file_path = os.path.join(base_result_path, 'evaluation_summary_full_trajectory.csv')
    
    print(f"--- Starting Analysis ---")
    print(f"Reading detailed data from: {detailed_file_path}")
    print(f"Will update summary file: {summary_file_path}")

    # --- 2. Load the Detailed Data ---
    try:
        df_detailed = pd.read_csv(detailed_file_path)
    except FileNotFoundError:
        print(f"\n[ERROR] Detailed data file not found at: {detailed_file_path}")
        print("Please ensure you have run the main evaluation script first.")
        return

    # --- 3. Define Criteria and Identify Hard Molecules ---
    
    # Define what makes a generated molecule "fully successful"
    # This should match the criteria in your main evaluation script (Part D)
    success_conditions = [
        (df_detailed['new_prop_BBB'] > 0.5),
        (df_detailed['new_prop_Ames'] < 0.5),
        # (df_detailed['new_prop_hERG_10uM'] < 0.5),
        # Add other conditions here if your model optimizes more properties, e.g.:
        # (df_detailed['new_prop_logS'] > (df_detailed['original_prop_logS'] + 0.5))
    ]
    df_detailed['is_molecule_successful'] = np.logical_and.reduce(success_conditions)

    # Define what makes a starting molecule "hard"
    hard_molecule_conditions = [
        (df_detailed['original_prop_Ames'] >= 0.5),
        (df_detailed['original_prop_BBB'] <= 0.5),
        # (df_detailed['original_prop_hERG_10uM'] >= 0.5)
    ]
    is_from_hard_molecule = np.logical_and.reduce(hard_molecule_conditions)
    
    # Get a unique list of the hard starting molecules
    hard_smiles_list = df_detailed[is_from_hard_molecule]['original_smiles'].unique()
    
    if len(hard_smiles_list) == 0:
        print("\n[INFO] No 'hard' molecules found in the dataset based on the criteria.")
        print("Cannot calculate the new success rate. Exiting.")
        return
        
    print(f"\nFound {len(hard_smiles_list)} unique 'hard' starting molecules for analysis.")

    # --- 4. Calculate Per-Run Success Rate for each Hard Molecule ---
    
    # Filter the dataframe to only include trajectories from hard molecules
    df_from_hard = df_detailed[df_detailed['original_smiles'].isin(hard_smiles_list)].copy()

    # The core of the new logic:
    # 1. Group by the starting molecule and the run index.
    # 2. For each group (i.e., each run), check if 'is_molecule_successful' is True for ANY step.
    # 3. This gives a boolean Series: True if the run was successful, False otherwise.
    run_success_status = df_from_hard.groupby(['original_smiles', 'run_index'])['is_molecule_successful'].any()

    # Now, calculate the success rate for each original molecule
    # 1. Group the run success status by the starting molecule.
    # 2. For each group, calculate the mean (True=1, False=0), which is the success rate for that molecule.
    success_rate_per_hard_smi = run_success_status.groupby('original_smiles').mean()
    
    # The final metric is the mean of these individual success rates
    final_mean_run_success_rate = success_rate_per_hard_smi.mean()
    
    new_metric_name = 'Mean_Run_Success_Rate_for_Hard_Molecules'
    print(f"\nCalculation complete.")
    print(f"  > New Metric '{new_metric_name}': {final_mean_run_success_rate:.4f}")

    # --- 5. Update and Save the Summary File ---
    try:
        df_summary = pd.read_csv(summary_file_path)
    except FileNotFoundError:
        print(f"\n[WARNING] Summary file not found at: {summary_file_path}")
        print("Creating a new summary file with the calculated metric.")
        df_summary = pd.DataFrame()

    # Add or update the new metric column
    df_summary[new_metric_name] = [final_mean_run_success_rate]

    # Save the updated dataframe back to the same file, overwriting it
    df_summary.to_csv(summary_file_path, index=False)
    
    print(f"\n[SUCCESS] Successfully appended the new metric to the summary file.")
    print(f"Updated file saved at: {summary_file_path}")
    print("\n--- Final Summary Data ---")
    print(df_summary.to_string(index=False))


if __name__ == '__main__':
    calculate_and_append_run_success_rate()