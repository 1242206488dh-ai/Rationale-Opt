from rdkit import Chem
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer # Make sure this import works in your environment

from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import QED, Descriptors, Crippen 
from rdkit.Chem import AllChem
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env 
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv 
from stable_baselines3.common.callbacks import BaseCallback 
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure as sb3_configure_logger
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.results_plotter import load_results, ts2xy

# --- SB3 Contrib Imports ---
from sb3_contrib import MaskablePPO 
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy 
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback 
from sb3_contrib.common.maskable.evaluation import evaluate_policy as maskable_evaluate_policy 
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement 

from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt 

from optuna import Trial, create_study
import optuna
from typing import Dict, List, Tuple, Optional, Any, Callable, Set

from gymnasium import spaces
import gymnasium as gym
import re
import random
from sklearn.metrics.pairwise import cosine_similarity
import pickle as pkl
from dgllife.utils import smiles_to_bigraph
import argparse
from functools import partial
from torch.utils.data import DataLoader
from model3.utils import collate_molgraphs, load_data 
from utils.opt_mol_generator import SME_opt_sub_detect, sub_data_filter, sme_mol_opt, generate_optimized_molecules
import model3.model_predictor
from sklearn.model_selection import train_test_split

if torch.cuda.is_available():
    print('use GPU')
    device_name = 'cuda'
else:
    print('use CPU')
    device_name = 'cpu'
device = torch.device(device_name) 

# Set global random seeds
seed = 43
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

from model3.featurizers import CanonicalAtomFeaturizer
from model3.featurizers import CanonicalBondFeaturizer
atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='hv')
bond_featurizer = CanonicalBondFeaturizer(bond_data_field='he', self_loop=True)

class Config:
    # --- 1. Core Configurations ---
    MODEL_NAMES = ['BBBP']

    # Optimization goals for all potential models
    OPTIMIZATION_GOALS = {
        'Mutagenicity': 'lower',
        'ESOL': 'higher',
        'hERG': 'lower',
        'BBBP': 'higher',
        'lipop': 'higher'
    }

    # --- 2. Path Configurations ---
    BASE_PATH = '/HOME/scz4306/run/SME/optimization'
    MODEL_NAMES_STR = "_".join(sorted(MODEL_NAMES))
    
    RESULT_PATH = os.path.join(BASE_PATH, f'RL/result_{MODEL_NAMES_STR}_0.2_sme_dataset_eval_freq_2000_n_step(4)') 
    TENSORBOARD_LOG_PATH = os.path.join(RESULT_PATH, 'tensorboard_logs')
    MODEL_SAVE_PATH = os.path.join(RESULT_PATH, 'models')

    # --- 3. Other Parameters ---
    ACTION_BUILDER_PARAMS = {
        'embedding_sim_threshold': 0.2,
        'atom_num_ratio_range': (5/6, 13/6),
    }
    ENV_PARAMS = {
        'max_steps': 10,
        'max_potential_actions': 800 
    }
    PPO_HYPERPARAMS = {
        'policy': 'MultiInputPolicy', 'n_steps': 1024, 'batch_size': 128, 'n_epochs': 10,
        'learning_rate': 3e-4, 'ent_coef': 0.001, 'gamma': 0.99, 'gae_lambda': 0.95,
        'clip_range': 0.2, 'vf_coef': 0.5, 'max_grad_norm': 0.5, 'device': device_name,
    }
    RULE_LOADING_STRATEGY = 0
    TOTAL_TIMESTEPS = 150000
    EVAL_FREQ = 2000
    EARLY_STOPPING_PATIENCE = 10
    N_EVAL_EPISODES = 5
    SME_MODEL_PATH = '/HOME/scz4306/run/SME/optimization/RL/model3/2model_817.pth'
    
    TRAIN_SMILES = []
    TEST_SMILES = []

    # --- 4. Helper Methods ---
    @staticmethod
    def _check_task_name(task_name_str: str, target_models: List[str]) -> bool:
        """
        Check if the task name matches target models exactly.
        """
        if not isinstance(task_name_str, str):
            return False
        return set(task_name_str.split()) == set(target_models)

    @staticmethod
    def setup_directories():
        """Ensure all required output directories exist."""
        os.makedirs(Config.RESULT_PATH, exist_ok=True)
        os.makedirs(Config.TENSORBOARD_LOG_PATH, exist_ok=True)
        os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)

# --- Data Loading Function ---
def load_datasets_into_config():
    """
    Load training and test datasets into Config based on specified group column.
    """
    print("\n[INFO] Loading training and test data...")
    
    if len(Config.MODEL_NAMES) != 1:
        print(f"[ERROR] Current logic only supports a single model name, configured: {Config.MODEL_NAMES}")
        Config.TRAIN_SMILES = []
        Config.TEST_SMILES = []
        return

    model_name = Config.MODEL_NAMES[0]
    data_path = f'/HOME/scz4306/run/SME/optimization/data/origin_data_raw/{model_name}.csv'
    print(f"       Target task: {model_name}")
    print(f"       Data file path: {data_path}")

    try:
        df_all = pd.read_csv(data_path)
        df_filtered_by_task = df_all

        train_df = df_filtered_by_task[df_filtered_by_task['group'].isin(['training', 'valid'])]
        Config.TRAIN_SMILES = train_df['smiles'].dropna().unique().tolist()
        print(f"[INFO] Training set loaded. Unique molecules: {len(Config.TRAIN_SMILES)}")

        test_df = df_filtered_by_task[df_filtered_by_task['group'] == 'test']
        Config.TEST_SMILES = test_df['smiles'].dropna().unique().tolist()
        print(f"[INFO] Test set loaded. Unique molecules: {len(Config.TEST_SMILES)}")

    except FileNotFoundError:
        print(f"[ERROR] Data file not found: {data_path}")
        Config.TRAIN_SMILES, Config.TEST_SMILES = [], []
    except KeyError as e:
        print(f"[ERROR] Missing required column in {data_path}: {e}")
        Config.TRAIN_SMILES, Config.TEST_SMILES = [], []
    except Exception as e:
        print(f"[ERROR] Error processing data file: {e}")
        Config.TRAIN_SMILES, Config.TEST_SMILES = [], []


# Setup configuration upon script execution
Config.setup_directories()
load_datasets_into_config()


class RuleManager:
    def __init__(self, model_names: List[str]):
        self.model_names = model_names
        self.potential_actions_library: List[str] = [] 
        self.max_n_actions: int = 0 

        self._load_all_fragments()
        self._load_resources()

    def _load_all_fragments(self):
        print("\n=== Initializing Fragment Library (RuleManager) ===")
        self.all_fragments: Set[str] = set()
        lib_path = '/HOME/scz4306/run/SME/optimization/data/origin_data/PubChem+SAR3.0_brics_lib.csv'
        try:
            df = pd.read_csv(lib_path)
            if 'frag_smiles' not in df.columns:
                raise ValueError("Missing 'frag_smiles' column in CSV")

            sampled_fragments = df['frag_smiles'].dropna().unique().tolist()
            if len(sampled_fragments) < 838:
                raise ValueError(f"Insufficient fragments, only {len(sampled_fragments)} found")

            selected_fragments = random.sample(sampled_fragments, 800)
            self.all_fragments.update(selected_fragments)
            print(f"✅ Randomly selected 800 fragments from {lib_path} (Total unique: {len(self.all_fragments)})")

        except FileNotFoundError:
            print(f"❌ Fragment library file not found: {lib_path}")
        except Exception as e:
            print(f"❌ Failed to load {lib_path}: {str(e)}")

        if self.all_fragments:
            self.potential_actions_library = sorted(list(self.all_fragments)) 
            self.max_n_actions = len(self.potential_actions_library)
            
            if Config.ENV_PARAMS['max_potential_actions'] < self.max_n_actions:
                print(f"⚠️ Config max_potential_actions ({Config.ENV_PARAMS['max_potential_actions']}) is smaller than loaded fragments ({self.max_n_actions}).")
            elif Config.ENV_PARAMS['max_potential_actions'] > self.max_n_actions:
                 print(f"ℹ️ Config max_potential_actions ({Config.ENV_PARAMS['max_potential_actions']}) is larger than loaded fragments ({self.max_n_actions}).")

            print(f"Total fragments for action library: {self.max_n_actions}")
            example_frags = self.potential_actions_library[:3] if self.max_n_actions >=3 else self.potential_actions_library
            print(f"Example fragments: {example_frags}")
        else:
            print(f"⚠️ Failed to load any BRICS fragments. ActionBuilder might not function.")
            self.max_n_actions = 0
            if Config.ENV_PARAMS['max_potential_actions'] == 0 :
                print("ERROR: max_potential_actions is 0. Setting to 1 to prevent env crash.")

    def _load_resources(self):
        print("🔧 Loading hyperparameters and rule resources (RuleManager)...")
        self.hyperparameter: Dict[str, Dict] = {} 
        all_rule_dfs_processed: List[pd.DataFrame] = [] 
        counts_from_strategy1: Dict[str, int] = {}

        for name in self.model_names:
            print(f"➡ Loading resources for model: {name}")
            
            hyper_path = os.path.join(Config.BASE_PATH, f'result/hyperparameter_{name}.pkl')
            try:
                with open(hyper_path, 'rb') as f:
                    self.hyperparameter[name] = pkl.load(f)
            except FileNotFoundError:
                self.hyperparameter[name] = {}
            except Exception as e:
                self.hyperparameter[name] = {}

            rule_path = os.path.join(Config.BASE_PATH, f'prediction/gain/{name}_brics_gain.csv')
            try:
                rule_df_original = pd.read_csv(rule_path) 
                current_rule_df = rule_df_original.copy()

                if 'frag_total_list' in current_rule_df.columns:
                    current_rule_df['frag_total_list'] = current_rule_df['frag_total_list'].apply(eval)
                if 'embedding' in current_rule_df.columns:
                    current_rule_df['embedding'] = current_rule_df['embedding'].apply(eval)

                if 'antecedents' in current_rule_df.columns and 'consequents' in current_rule_df.columns:
                    current_rule_df['rule_key'] = current_rule_df['antecedents'].astype(str) + '->' + current_rule_df['consequents'].astype(str)
                    rule_counts = current_rule_df['rule_key'].value_counts()
                    unique_keys = rule_counts[rule_counts == 1].index
                else:
                    print(f"  ⚠️ Missing 'antecedents' or 'consequents' in {rule_path}.")
                    unique_keys = pd.Index([]) 

                processed_df_for_model = pd.DataFrame() 

                if Config.RULE_LOADING_STRATEGY == 1: 
                    print(f"  Strategy 1 (BIASED_UNIQUE) applied to {name}...")
                    if 'gain_AB_avg(%)' in current_rule_df.columns and 'rule_key' in current_rule_df.columns:
                        if 'Mutagenicity' in name:
                            processed_df_for_model = current_rule_df[
                                (current_rule_df['rule_key'].isin(unique_keys)) &
                                (current_rule_df['gain_AB_avg(%)'] < 0)
                            ]
                            print(f"    Mutagenicity: Filtered {len(processed_df_for_model)} unique rules with gain < 0.")
                        elif 'ESOL' in name:
                            processed_df_for_model = current_rule_df[
                                (current_rule_df['rule_key'].isin(unique_keys)) &
                                (current_rule_df['gain_AB_avg(%)'] > 0)
                            ]
                            print(f"    ESOL: Filtered {len(processed_df_for_model)} unique rules with gain > 0.")
                        else:
                            processed_df_for_model = current_rule_df
                            print(f"    {name}: Kept all {len(processed_df_for_model)} rules.")
                        counts_from_strategy1[name] = len(processed_df_for_model) 
                    else:
                        print(f"  ⚠️ {name}: Missing required columns for Strategy 1.")
                        processed_df_for_model = current_rule_df
                        counts_from_strategy1[name] = len(processed_df_for_model)

                elif Config.RULE_LOADING_STRATEGY == 2: 
                    print(f"  Strategy 2 (RANDOM_MATCHING_COUNT) applied to {name}...")
                    processed_df_for_model = current_rule_df 

                elif Config.RULE_LOADING_STRATEGY == 0: 
                    print(f"  Strategy 0 (NONE) applied to {name}...")
                    if 'gain_AB_avg(%)' in current_rule_df.columns and 'rule_key' in current_rule_df.columns:
                        temp_df = current_rule_df.copy()
                        if 'Mutagenicity' in name or 'hERG' in name :
                            temp_df = temp_df[~((temp_df['rule_key'].isin(unique_keys)) & (temp_df['gain_AB_avg(%)'] < 0))]
                        elif 'ESOL' in name or 'BBBP' in name:
                            temp_df = temp_df[~((temp_df['rule_key'].isin(unique_keys)) & (temp_df['gain_AB_avg(%)'] < 0))]
                        processed_df_for_model = temp_df
                        print(f"    {name}: Remaining rules after default exclusions: {len(processed_df_for_model)}.")
                    else:
                        processed_df_for_model = current_rule_df
                        print(f"    {name}: Kept all {len(processed_df_for_model)} rules.")
                else:
                    print(f"  ⚠️ Unknown RULE_LOADING_STRATEGY: {Config.RULE_LOADING_STRATEGY}. Applying no filters.")
                    processed_df_for_model = current_rule_df
                
                all_rule_dfs_processed.append(processed_df_for_model)
            except FileNotFoundError:
                print(f"  ❌ Rule file not found: {rule_path}")
            except Exception as e:
                print(f"  ❌ Failed to process rule file {rule_path}: {e}")

        if not all_rule_dfs_processed:
            self.rule_df = pd.DataFrame()
            print("⚠️ All rule files failed to load. Rule library is empty!")
        else:
            combined_df = pd.concat(all_rule_dfs_processed).reset_index(drop=True)
            if Config.RULE_LOADING_STRATEGY == 2:
                print("\n  Applying random sampling for Strategy 2...")
                _counts_strat1_temp: Dict[str, int] = {}
                for name_s1 in self.model_names:
                    rule_path_s1 = os.path.join(Config.BASE_PATH, f'prediction/gain/{name_s1}_brics_gain.csv')
                    try:
                        df_s1 = pd.read_csv(rule_path_s1)
                        if 'antecedents' in df_s1.columns and 'consequents' in df_s1.columns:
                             df_s1['rule_key'] = df_s1['antecedents'].astype(str) + '->' + df_s1['consequents'].astype(str)
                        else: continue 
                        rule_counts_s1 = df_s1['rule_key'].value_counts()
                        unique_keys_s1 = rule_counts_s1[rule_counts_s1 == 1].index
                        count = 0
                        if 'gain_AB_avg(%)' in df_s1.columns:
                            if 'Mutagenicity' in name_s1:
                                count = len(df_s1[(df_s1['rule_key'].isin(unique_keys_s1)) & (df_s1['gain_AB_avg(%)'] < 0)])
                            elif 'ESOL' in name_s1:
                                count = len(df_s1[(df_s1['rule_key'].isin(unique_keys_s1)) & (df_s1['gain_AB_avg(%)'] > 0)])
                            else: 
                                count = len(df_s1[df_s1['rule_key'].isin(unique_keys_s1)])
                        _counts_strat1_temp[name_s1] = count
                    except: 
                        _counts_strat1_temp[name_s1] = 0 
                
                print(f"    Target counts determined by Strategy 1: {_counts_strat1_temp}")
                final_dfs_for_concat_strat2 = []
                temp_all_rule_dfs_full = [] 
                
                for name_idx, name_model in enumerate(self.model_names):
                    rule_path_full = os.path.join(Config.BASE_PATH, f'prediction/gain/{name_model}_brics_gain.csv')
                    try:
                        df_full = pd.read_csv(rule_path_full)
                        if 'frag_total_list' in df_full.columns: df_full['frag_total_list'] = df_full['frag_total_list'].apply(eval)
                        if 'embedding' in df_full.columns: df_full['embedding'] = df_full['embedding'].apply(eval)
                        if 'antecedents' in df_full.columns and 'consequents' in df_full.columns:
                            df_full['rule_key'] = df_full['antecedents'].astype(str) + '->' + df_full['consequents'].astype(str)
                        temp_all_rule_dfs_full.append(df_full)
                    except Exception as e_load:
                        print(f"    Failed to load full rules for {name_model}: {e_load}")
                        temp_all_rule_dfs_full.append(pd.DataFrame())

                for df_model_full, model_name_iter in zip(temp_all_rule_dfs_full, self.model_names):
                    if df_model_full.empty:
                        final_dfs_for_concat_strat2.append(pd.DataFrame())
                        continue
                    target_count = _counts_strat1_temp.get(model_name_iter, 0)
                    if target_count == 0:
                        print(f"    {model_name_iter}: Target count is 0, skipping.")
                        final_dfs_for_concat_strat2.append(pd.DataFrame())
                        continue
                    if len(df_model_full) > target_count:
                        sampled_df = df_model_full.sample(n=target_count, random_state=Config.RANDOM_STATE if hasattr(Config, 'RANDOM_STATE') else 42) 
                        print(f"    {model_name_iter}: Sampled {target_count} rules from {len(df_model_full)} total.")
                    else:
                        sampled_df = df_model_full 
                        print(f"    {model_name_iter}: Rules available ({len(df_model_full)}) < target ({target_count}), keeping all.")
                    final_dfs_for_concat_strat2.append(sampled_df)
                if final_dfs_for_concat_strat2:
                    combined_df = pd.concat(final_dfs_for_concat_strat2).reset_index(drop=True)
                else:
                    combined_df = pd.DataFrame()

            if not combined_df.empty and 'antecedents' in combined_df.columns and 'consequents' in combined_df.columns:
                 self.rule_df = combined_df.drop_duplicates(subset=['antecedents', 'consequents']).reset_index(drop=True)
            elif not combined_df.empty: 
                self.rule_df = combined_df 
            else: 
                self.rule_df = pd.DataFrame()

            if not self.rule_df.empty:
                print(f"✅ Final rule count (Strategy {Config.RULE_LOADING_STRATEGY}): {len(self.rule_df)}")
                if 'embedding' in self.rule_df.columns and not self.rule_df['embedding'].empty:
                    try: 
                        first_emb = self.rule_df['embedding'].iloc[0]
                        if hasattr(first_emb, '__iter__') and first_emb is not None and len(list(first_emb)) > 0:
                            print(f"  Example rule embedding (first 5 dims): {str(list(first_emb)[:5])}")
                        else:
                            print(f"  Example rule embedding (raw): {first_emb}")
                    except IndexError:
                        print("  ⚠️ Rule embedding column is not empty, but failed to fetch first element.")
                else:
                    print("  ⚠️ Rule embedding column missing or empty.")
            else:
                print("⚠️ No valid rules available after applying strategies and merging!")

        self.frag_dict: Dict[str, List[float]] = {} 
        frag_emb_all = []

        frag_emb_path = os.path.join(Config.BASE_PATH, f'RL/embedding/PubChem+SAR3.0_frag_emb.csv')
        try:
            if os.path.exists(frag_emb_path):
                frag_df = pd.read_csv(frag_emb_path)
                if 'embedding' in frag_df.columns: 
                    frag_df['embedding'] = frag_df['embedding'].apply(eval)
                    frag_emb_all.append(frag_df)
                else:
                    print(f"  ⚠️ Fragment embedding file {frag_emb_path} missing 'embedding' column.")
            else:
                print(f"  ⚠️ Fragment embedding file not found: {frag_emb_path}")
        except Exception as e:
            print(f"  ❌ Failed to process fragment embeddings {frag_emb_path}: {e}")
        
        if frag_emb_all:
            frag_df_combined = pd.concat(frag_emb_all).drop_duplicates(subset=['smiles']).reset_index(drop=True)
            if 'smiles' in frag_df_combined.columns and 'embedding' in frag_df_combined.columns:
                self.frag_dict = dict(zip(frag_df_combined['smiles'], frag_df_combined['embedding']))
                if self.frag_dict:
                    sample_smiles = list(self.frag_dict.keys())[0]
                    sample_emb = self.frag_dict[sample_smiles]
                    print(f"✅ Fragment dictionary built successfully, total: {len(self.frag_dict)}. Example key: {sample_smiles}")
                else:
                    print("⚠️ Fragment dictionary empty after merge.")
            else:
                print("⚠️ Merged fragment DataFrame missing 'smiles' or 'embedding' columns.")
        else:
            print("⚠️ All fragment embedding files failed to load. ActionBuilder may fail.")


# ================== Utility Functions ==================

def get_heavy_atom_count(smiles: str) -> int:
    """Calculate the number of heavy atoms in a SMILES, excluding attachment points ([*]) and hydrogens."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() > 1:
                count += 1
        return count
    except:
        return 0

def calculate_tanimoto(smi1: str, smi2: str) -> float:
    """Calculate Tanimoto similarity (Morgan Fingerprints) between two SMILES."""
    try:
        mol1 = Chem.MolFromSmiles(smi1.replace('*','')) 
        mol2 = Chem.MolFromSmiles(smi2.replace('*',''))
        if not mol1 or not mol2: return 0.0
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, 1024)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 1024)
        return TanimotoSimilarity(fp1, fp2)
    except:
        return 0.0


class ActionBuilder:
    def __init__(self, rule_manager: RuleManager): 
        self.rule_manager = rule_manager
        self.params = Config.ACTION_BUILDER_PARAMS
        self.frag_to_idx_map = {frag: i for i, frag in enumerate(self.rule_manager.potential_actions_library)}
        print(f"ActionBuilder initialized with params: {self.params}")
        print(f"ActionBuilder potential action library size: {self.rule_manager.max_n_actions}")

    def build_actions(self, context_emb_list: List[np.ndarray], sub_emb: np.ndarray,
                      target_num_attachment_points: int, exclude_frag: Optional[str] = None):
        print(f"\n=== Building Action Mask and Valid Fragments (ActionBuilder) ===")
        print(f"  Inputs: target_attachments={target_num_attachment_points}, exclude_frag='{exclude_frag}', num_context_embs={len(context_emb_list)}")

        action_mask = np.zeros(self.rule_manager.max_n_actions, dtype=bool)

        if target_num_attachment_points == 0 and exclude_frag and '*' not in exclude_frag:
            print(f"  ⚠️ target_attachments is 0 and exclude_frag ('{exclude_frag}') looks like a full molecule. Filtering out.")
            return action_mask, [] 

        primary_filtered_mask = self._primary_filter_to_mask(
            target_num_attachment_points=target_num_attachment_points,
            exclude_frag=exclude_frag
        )
        num_after_primary = np.sum(primary_filtered_mask)
        print(f"  Valid actions after primary filter: {num_after_primary}")
        if num_after_primary == 0:
            return action_mask, []

        final_candidate_mask = self._embedding_filter_on_mask(
            initial_mask=primary_filtered_mask,
            context_emb_list=context_emb_list,
            sub_emb_for_combined=sub_emb
        )
        num_after_embedding = np.sum(final_candidate_mask)
        print(f"  Valid actions after embedding filter: {num_after_embedding}")
        if num_after_embedding == 0:
            return action_mask, []

        valid_frag_smiles_for_step = []
        true_indices = np.where(final_candidate_mask)[0]
        
        for frag_idx in true_indices:
            frag_smiles = self.rule_manager.potential_actions_library[frag_idx]
            if frag_smiles in self.rule_manager.frag_dict:
                valid_frag_smiles_for_step.append(frag_smiles)
            else:
                print(f"    CRITICAL WARNING: Fragment '{frag_smiles}' (idx {frag_idx}) missing in frag_dict. Removing from mask.")
                final_candidate_mask[frag_idx] = False 

        if np.sum(final_candidate_mask) != len(valid_frag_smiles_for_step):
            print(f"    CRITICAL WARNING: Mismatch after final check. Mask sum: {np.sum(final_candidate_mask)}, Valid SMILES: {len(valid_frag_smiles_for_step)}.")
            valid_frag_smiles_for_step = [
                self.rule_manager.potential_actions_library[i] 
                for i in np.where(final_candidate_mask)[0] 
                if self.rule_manager.potential_actions_library[i] in self.rule_manager.frag_dict
            ]

        print(f"  Final valid fragment SMILES generated: {len(valid_frag_smiles_for_step)}")
        return final_candidate_mask, valid_frag_smiles_for_step

    def _primary_filter_to_mask(self, target_num_attachment_points: int, exclude_frag: Optional[str]): 
        potential_actions = self.rule_manager.potential_actions_library
        mask = np.zeros(len(potential_actions), dtype=bool)

        exclude_frag_heavy_atoms = get_heavy_atom_count(exclude_frag) if exclude_frag else 0
        min_atoms = 0.0 
        max_atoms = float('inf')
        if exclude_frag_heavy_atoms > 0: 
            min_atoms = exclude_frag_heavy_atoms * self.params['atom_num_ratio_range'][0]
            max_atoms = exclude_frag_heavy_atoms * self.params['atom_num_ratio_range'][1]
        
        for i, frag_smi in enumerate(potential_actions):
            if frag_smi == exclude_frag:
                continue
            if frag_smi.count('*') != target_num_attachment_points:
                continue

            current_frag_heavy_atoms = get_heavy_atom_count(frag_smi)
            if exclude_frag_heavy_atoms > 0: 
                if not (min_atoms <= current_frag_heavy_atoms <= max_atoms):
                    continue
            mask[i] = True 
        return mask

    def _embedding_filter_on_mask(self, initial_mask: np.ndarray,
                                  context_emb_list: List[np.ndarray],
                                  sub_emb_for_combined: np.ndarray): 
        if not np.any(initial_mask) or self.rule_manager.rule_df.empty or 'embedding' not in self.rule_manager.rule_df.columns:
            print("    ⚠️ Embedding filter skipped: No valid fragments or rule database unavailable.")
            return initial_mask 

        rule_embs_list = [
            np.array(e, dtype=np.float32)
            for e in self.rule_manager.rule_df['embedding'].tolist()
            if isinstance(e, (list, np.ndarray)) and hasattr(e, '__len__') and len(e) > 0 and np.array(e).ndim > 0
        ]
        if not rule_embs_list:
            print("    ⚠️ Rule database has no valid embeddings, skipping similarity calculation.")
            return initial_mask
        
        try:
            rule_embs_matrix = np.array(rule_embs_list)
            if rule_embs_matrix.ndim == 1 and rule_embs_matrix.shape[0] > 0: 
                 rule_embs_matrix = rule_embs_matrix.reshape(1, -1)
            elif rule_embs_matrix.ndim == 0 or rule_embs_matrix.shape[0] == 0 or rule_embs_matrix.size == 0:
                print("    ⚠️ Rule embedding matrix is empty, skipping embedding filter.")
                return initial_mask
        except ValueError as e:
            print(f"    ❌ Failed to create embedding matrix: {e}. Skipping embedding filter.")
            return initial_mask

        embedding_passed_mask = np.copy(initial_mask)
        print(f"  ActionBuilder: Input context embs for combined_emb calc: {len(context_emb_list)}")

        for frag_idx in np.where(initial_mask)[0]:
            frag_smiles = self.rule_manager.potential_actions_library[frag_idx]

            if frag_smiles not in self.rule_manager.frag_dict:
                print(f"[DEBUG] Fragment '{frag_smiles}' missing from frag_dict. Disabling in embedding mask.")
                embedding_passed_mask[frag_idx] = False 
                continue

            frag_emb = np.array(self.rule_manager.frag_dict[frag_smiles], dtype=np.float32)
            if not (frag_emb.ndim == 1 and frag_emb.shape[0] == 128):
                print(f"    ❌ Fragment '{frag_smiles}': invalid embedding shape {frag_emb.shape}.")
                embedding_passed_mask[frag_idx] = False
                continue
            
            try:
                all_embs_for_combined = []
                for ctx_e in context_emb_list: 
                    all_embs_for_combined.append(ctx_e)
                all_embs_for_combined.append(frag_emb) 

                if not all_embs_for_combined: 
                    print(f"    ❌ Fragment '{frag_smiles}': missing embeddings for combined_emb.")
                    embedding_passed_mask[frag_idx] = False
                    continue
                
                combined_emb_stacked = np.stack(all_embs_for_combined, axis=0) 
                combined_emb = np.mean(combined_emb_stacked, axis=0) 
                
                sim_vector = cosine_similarity(combined_emb[np.newaxis, :], rule_embs_matrix)[0]
                max_sim = np.max(sim_vector)

                if max_sim < self.params['embedding_sim_threshold']:
                    embedding_passed_mask[frag_idx] = False
            except Exception as e:
                print(f"    ❌ Failed embedding filter for '{frag_smiles}': {e}")
                import traceback
                traceback.print_exc()
                embedding_passed_mask[frag_idx] = False
                continue
        
        return embedding_passed_mask
    

class MoleculeOptimEnv(gym.Env): 
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, model_names: List[str], smiles_list: List[str],
                 max_steps: int = Config.ENV_PARAMS['max_steps'],
                 mode: str = 'higher', is_eval_env: bool = False):
        super().__init__()
        self.model_names = model_names
        self.max_steps = max_steps
        self.initial_smiles_list = smiles_list
        self.current_smiles_pool = list(smiles_list) 
        self.smiles_idx = 0
        self.current_step = 0
        self.mode = mode
        self.is_eval_env = is_eval_env
        self.last_reset_info: Dict = {} 

        self.label_name = []
        self.model_to_label_map = {"Mutagenicity": "Ames", "ESOL": "logS", "hERG": "hERG_10uM", "BBBP": "BBB"}
        for name in self.model_names:
            if name in self.model_to_label_map:
                self.label_name.append(self.model_to_label_map[name])
        
        if not self.is_eval_env: 
            print(f"Env targeting properties: {self.label_name}")

        self.rule_manager = RuleManager(self.model_names)
        self._sme_hyperparams = {
            model: self.rule_manager.hyperparameter.get(model, {}) for model in self.model_names
        }

        if hasattr(self.rule_manager, 'max_n_actions') and self.rule_manager.max_n_actions > 0:
            self.max_potential_actions = self.rule_manager.max_n_actions
            if not self.is_eval_env or self.smiles_idx <= 1 : 
                 print(f"  INFO: max_potential_actions set to {self.max_potential_actions}.")
        else:
            print(f"  CRITICAL ERROR: RuleManager.max_n_actions is {getattr(self.rule_manager, 'max_n_actions', 'N/A')}. "
                  f"Action space cannot be determined correctly. Check fragment loading.")

        self.action_builder = ActionBuilder(self.rule_manager)
        self.property_predictor = self._load_property_predictor()
        self._property_cache: Dict[str, Dict[str, float]] = {}
        self._graph_cache: Dict[str, Tuple] = {}

        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(128,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.max_potential_actions,), dtype=np.bool_)
        })
        self.action_space = spaces.Discrete(self.max_potential_actions)
        
        if not self.is_eval_env or self.smiles_idx <=1 : 
            print(f"  Environment observation_space['action_mask'] shape: {self.observation_space['action_mask'].shape}")
            print(f"  Environment action_space: Discrete({self.action_space.n})")

        self.optimization_paths_episode: List[Dict] = []
        self.all_optimization_results: List[Dict] = []

        self.original_mol_for_episode: Optional[str] = None
        self.current_mol: Optional[str] = None
        self.rm_atom_idx: Optional[List[int]] = None
        self.sub_connect_num: Optional[int] = None
        self.current_sme_pred_value: Optional[float] = None
        self.current_opt_subs: List[str] = []    

        self.current_state_numeric = np.zeros(self.observation_space["observation"].shape, dtype=self.observation_space["observation"].dtype)
        self.current_action_mask_bool = np.zeros(self.max_potential_actions, dtype=bool)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.optimization_paths_episode = []
        info_to_return: Dict = {"actions_available": False, "error": "initialization_pending"}
        
        max_molecule_init_attempts = len(self.current_smiles_pool) * 3 if self.current_smiles_pool else 10
        if not self.current_smiles_pool and not self.initial_smiles_list:
            self.current_state_numeric = np.zeros((128,), dtype=np.float32)
            self.current_action_mask_bool = np.zeros(self.max_potential_actions, dtype=bool)
            self.current_opt_subs = []
            return self._get_current_observation(), {"error": "empty_smiles_pool", "actions_available": False}

        for attempt in range(max_molecule_init_attempts):
            if not self.current_smiles_pool:
                self.current_smiles_pool = list(self.initial_smiles_list)
                if not self.current_smiles_pool:
                    self.current_state_numeric = np.zeros((128,), dtype=np.float32)
                    self.current_action_mask_bool = np.zeros(self.max_potential_actions, dtype=bool)
                    return self._get_current_observation(), {"error": "empty_initial_smiles_list_critical", "actions_available": False}
                random.shuffle(self.current_smiles_pool)
                self.smiles_idx = 0
            
            if self.smiles_idx >= len(self.current_smiles_pool):
                self.smiles_idx = 0 
                random.shuffle(self.current_smiles_pool) 

            self.original_mol_for_episode = self.current_smiles_pool[self.smiles_idx]
            self.current_mol = self.original_mol_for_episode
            self.smiles_idx += 1

            all_detect_outputs = []
            for model_name in self.model_names:
                sme_hp = self._sme_hyperparams.get(model_name, {})
                if model_name == 'Mutagenicity' or model_name == 'hERG':
                    model_mode = 'lower'
                elif model_name == 'ESOL' or model_name == 'BBBP':
                    model_mode = 'higher'
                else:
                    model_mode = self.mode 

                try:
                    outputs = SME_opt_sub_detect( 
                        smiles=self.current_mol, model_name=model_name,
                        rgcn_hidden_feats=sme_hp.get('rgcn_hidden_feats'),
                        ffn_hidden_feats=sme_hp.get('ffn_hidden_feats'),
                        lr=sme_hp.get('lr'), classification=sme_hp.get('classification'),
                        mode=model_mode
                    )
                    if outputs and outputs != [-1,-1]:
                        all_detect_outputs.extend(outputs)
                    elif outputs == [-1,-1]:
                        print(f"  DEBUG ENV RESET: SME_opt_sub_detect returned [-1,-1] for {self.current_mol}.")
                except Exception as e:
                    print(f"  DEBUG ENV RESET: SME_opt_sub_detect failed: {e}")
                    continue

            if not all_detect_outputs:
                info_to_return = {"error": f"no_sme_outputs_for_{self.current_mol}", "actions_available": False} 
                continue

            for sme_output_raw in all_detect_outputs:
                pred_value, sub_smi_to_change, _, current_rm_atom_idx, \
                _, current_sub_connect_num, \
                sub_smi_embedding_raw, context_embedding_raw_list_or_single = sme_output_raw

                if sub_smi_to_change == -1 or not sub_smi_to_change: continue
                if current_sub_connect_num == 0 and sub_smi_to_change and '*' not in sub_smi_to_change:
                    continue
                
                print(f"  DEBUG ENV RESET: State calculation for sub_smi='{sub_smi_to_change}'")

                if not (isinstance(sub_smi_embedding_raw, np.ndarray) and sub_smi_embedding_raw.shape == (128,)):
                    print(f"    Skipping site due to invalid sub_smi_embedding_raw shape.")
                    continue
                sub_emb_reshaped = sub_smi_embedding_raw.astype(np.float32)[:, np.newaxis] 

                temp_context_list_for_processing = []
                processed_context_source = context_embedding_raw_list_or_single
                if isinstance(processed_context_source, np.ndarray) and processed_context_source.shape == (128,):
                    temp_context_list_for_processing = [processed_context_source.astype(np.float32)]
                elif isinstance(processed_context_source, list):
                    for item in processed_context_source:
                        if isinstance(item, np.ndarray) and item.shape == (128,):
                            temp_context_list_for_processing.append(item.astype(np.float32))
                
                if not temp_context_list_for_processing:
                    print(f"    Skipping site: No valid context embeddings.")
                    continue
                
                context_embs_stacked = np.stack(temp_context_list_for_processing, axis=-1) 
                all_embs_for_state_stacked = np.concatenate([sub_emb_reshaped, context_embs_stacked], axis=1) 
                calculated_state_for_this_site = np.mean(all_embs_for_state_stacked, axis=1) 

                if calculated_state_for_this_site.shape != (128,):
                    print(f"    Skipping site due to shape mismatch: {calculated_state_for_this_site.shape}")
                    continue
                
                action_mask, valid_frags = self.action_builder.build_actions(
                    context_emb_list=temp_context_list_for_processing,
                    sub_emb=sub_smi_embedding_raw.astype(np.float32),
                    target_num_attachment_points=current_sub_connect_num,
                    exclude_frag=sub_smi_to_change
                )

                if np.any(action_mask):
                    self.current_state_numeric = calculated_state_for_this_site.astype(np.float32)
                    self.current_action_mask_bool = action_mask.astype(np.bool_)
                    self.current_opt_subs = valid_frags
                    self.rm_atom_idx = current_rm_atom_idx
                    self.sub_connect_num = current_sub_connect_num
                    self.current_sme_pred_value = pred_value
                    print(f"  DEBUG ENV RESET: Success. Current mol: {self.current_mol}. Available actions: {np.sum(self.current_action_mask_bool)}")
                    info_to_return = {"actions_available": True, "error": ""}
                    self.last_reset_info = info_to_return 
                    return self._get_current_observation(), info_to_return
        
        self.current_state_numeric = np.zeros((128,), dtype=np.float32)
        self.current_action_mask_bool = np.zeros(self.max_potential_actions, dtype=bool)
        self.current_opt_subs = []
        
        error_msg_reset = "max_reset_attempts_reached"
        if not all_detect_outputs and attempt == max_molecule_init_attempts - 1:
             error_msg_reset = "max_reset_attempts_reached_likely_no_sme_sites_found"

        print(f"DEBUG ENV RESET: Max attempts reached for {self.original_mol_for_episode}. Error: {info_to_return.get('error', error_msg_reset)}")
        final_error_info = info_to_return if "no_sme_outputs" in info_to_return.get("error","") else {"error": error_msg_reset, "actions_available": False}
        self.last_reset_info = final_error_info 
        return self._get_current_observation(), final_error_info
    
    def _prepare_next_state_from_mol(self, smiles_to_process: str) -> Tuple[Dict[str, np.ndarray], Dict]:
        all_detect_outputs_next = []
        for model_name in self.model_names:
            sme_hp = self._sme_hyperparams.get(model_name, {})
            if model_name == 'Mutagenicity' or model_name == 'hERG':
                model_mode = 'lower'
            elif model_name == 'ESOL' or model_name == 'BBBP':
                model_mode = 'higher'
            else:
                model_mode = self.mode

            try:
                outputs = SME_opt_sub_detect( 
                    smiles=smiles_to_process, model_name=model_name,
                    rgcn_hidden_feats=sme_hp.get('rgcn_hidden_feats'),
                    ffn_hidden_feats=sme_hp.get('ffn_hidden_feats'),
                    lr=sme_hp.get('lr'), classification=sme_hp.get('classification'),
                    mode=model_mode
                )
                if outputs and outputs != [-1, -1]: 
                    all_detect_outputs_next.extend(outputs)
            except Exception as e:
                print(f"  DEBUG ENV PREP_NEXT_STATE: SME_opt_sub_detect failed: {e}")
                continue

        if not all_detect_outputs_next:
            self.current_action_mask_bool = np.zeros(self.max_potential_actions, dtype=bool)
            self.current_opt_subs = []
            return self._get_current_observation(), {"error": "sme_no_points_next_state", "actions_available": False}

        random.shuffle(all_detect_outputs_next)

        for sme_output_raw in all_detect_outputs_next:
            _pred_val, sub_smi, _, rm_idx_next, \
            _, connect_num_next, sub_emb_next_raw, ctx_emb_next_raw_list_or_single = sme_output_raw

            if sub_smi == -1 or not sub_smi: continue
            if connect_num_next == 0 and sub_smi and '*' not in sub_smi: continue

            print(f"  DEBUG ENV PREP_NEXT_STATE: State calculation for sub_smi='{sub_smi}'")

            if not (isinstance(sub_emb_next_raw, np.ndarray) and sub_emb_next_raw.shape == (128,)):
                continue
            sub_emb_next_reshaped = sub_emb_next_raw.astype(np.float32)[:, np.newaxis]
            
            contexts_to_process_next = []
            processed_context_source_next = ctx_emb_next_raw_list_or_single
            if isinstance(processed_context_source_next, np.ndarray) and processed_context_source_next.shape == (128,):
                contexts_to_process_next = [processed_context_source_next.astype(np.float32)]
            elif isinstance(processed_context_source_next, list):
                for item in processed_context_source_next:
                    if isinstance(item, np.ndarray) and item.shape == (128,):
                        contexts_to_process_next.append(item.astype(np.float32))
            
            if not contexts_to_process_next:
                continue
            
            context_embs_next_stacked = np.stack(contexts_to_process_next, axis=-1)
            all_embs_for_state_next_stacked = np.concatenate([sub_emb_next_reshaped, context_embs_next_stacked], axis=1)
            next_calculated_state_arr = np.mean(all_embs_for_state_next_stacked, axis=1)

            if next_calculated_state_arr.shape != (128,):
                continue
            
            action_mask_next, valid_frags_next = self.action_builder.build_actions(
                context_emb_list=contexts_to_process_next,
                sub_emb=sub_emb_next_raw.astype(np.float32),
                target_num_attachment_points=connect_num_next,
                exclude_frag=sub_smi
            )

            if np.any(action_mask_next):
                self.current_state_numeric = next_calculated_state_arr.astype(np.float32)
                self.current_action_mask_bool = action_mask_next.astype(np.bool_)
                self.current_opt_subs = valid_frags_next
                self.rm_atom_idx = rm_idx_next
                self.sub_connect_num = connect_num_next
                self.current_sme_pred_value = _pred_val 
                print(f"  DEBUG ENV PREP_NEXT_STATE: Found {np.sum(action_mask_next)} actions for '{smiles_to_process}'.")
                return self._get_current_observation(), {"actions_available": True}
        
        self.current_action_mask_bool = np.zeros(self.max_potential_actions, dtype=bool)
        self.current_opt_subs = []
        print(f"  DEBUG ENV PREP_NEXT_STATE: No valid actions yielded.")
        return self._get_current_observation(), {"error": "no_actions_for_any_sme_next_state", "actions_available": False}

    def _get_current_observation(self):
        return {
            "observation": self.current_state_numeric.astype(np.float32),
            "action_mask": self.current_action_mask_bool.astype(np.bool_)
        }

    def action_masks(self) -> np.ndarray:
        return self.current_action_mask_bool.astype(np.bool_)
    
    def _load_property_predictor(self):
        try:
            model = model3.model_predictor.ModelPredictor(
                node_feat_size=atom_featurizer.feat_size('hv'),
                edge_feat_size=bond_featurizer.feat_size('he'),
                num_layers=2, num_timesteps=1, graph_feat_size=256,
                predictor_hidden_feats=256, n_tasks=108
            )
            model.load_state_dict(torch.load(Config.SME_MODEL_PATH, map_location=device))
            model = model.to(device)
            model.eval()
            print("✅ Property predictor model loaded successfully.")
            return model
        except Exception as e:
            print(f"❌ Error loading property predictor model: {e}")
            raise

    def _prepare_graph_for_prediction(self, smi: str):
        if smi in self._graph_cache: return self._graph_cache[smi]
        df = pd.DataFrame([[smi, 0]], columns=['SMILES', 'dummy_label'])
        try:
            dataset = load_data(df, id=None)
            dataloader = DataLoader(dataset=dataset, batch_size=1, collate_fn=collate_molgraphs)
            batch = next(iter(dataloader))
            _, bg, _, _ = batch
            bg = bg.to(device)
            n_feats = bg.ndata.pop('hv').to(device)
            e_feats = bg.edata.pop('he').to(device)
            self._graph_cache[smi] = (bg, n_feats, e_feats)
            return bg, n_feats, e_feats
        except Exception: return None, None, None

    def _predict_property(self, smi: str) -> Dict[str, float]:
        """
        Predict properties for a given SMILES.
        Processes Ames (classification), hERG (classification), and logS (regression).
        - Ames/hERG: apply sigmoid.
        - logS: use raw logit values.
        """
        if smi in self._property_cache:
            return self._property_cache[smi]

        bg, n_feats, e_feats = self._prepare_graph_for_prediction(smi)
        if bg is None:
            error_results = {name: 0.0 for name in self.label_name}
            self._property_cache[smi] = error_results
            return error_results

        # Mapping based on original model output indices
        property_to_idx = {
            "Ames": 62,
            "hERG_10uM": 34,
            "logS": 93,
            "BBB": 80,
        }

        try:
            with torch.no_grad():
                all_logits = self.property_predictor(bg, n_feats, e_feats) 

            results = {}
            for prop_name in self.label_name: 
                model_idx = property_to_idx.get(prop_name)
                if model_idx is None:
                    print(f"  [Prop Predict] WARNING: Property '{prop_name}' not found. Defaulting to 0.0")
                    results[prop_name] = 0.0
                    continue

                raw_logit = all_logits[0, model_idx]

                if prop_name == "logS":
                    final_value = raw_logit.item()
                    print(f"  [Prop Predict] {prop_name:<10}: Logit={raw_logit.item():.4f} -> Final={final_value:.4f}")
                else: 
                    final_value = torch.sigmoid(raw_logit).item()
                    print(f"  [Prop Predict] {prop_name:<10}: Logit={raw_logit.item():.4f} -> Final={final_value:.4f}")
                
                results[prop_name] = final_value

            self._property_cache[smi] = results
            return results

        except Exception as e:
            print(f"  [Prop Predict] CRITICAL ERROR during prediction: {e}")
            error_results = {name: 0.0 for name in self.label_name}
            self._property_cache[smi] = error_results
            return error_results

    def _calculate_reward(self, original_smi: str, new_smi: str) -> Tuple[float, float, float, float, Dict, Dict]:
        """
        Calculate rewards based on property changes.
        - Ames/hERG: Classification (lower is better).
        - BBBP: Classification (higher is better).
        - ESOL: Regression (higher is better).
        """
        print(f"  --- Calculating Reward for '{original_smi[:20]}...' -> '{new_smi[:20]}...' ---")
        orig_props = self._predict_property(original_smi)
        new_props = self._predict_property(new_smi)

        total_reward: float = 0.0
        raw_delta_muta: float = 0.0
        raw_delta_esol: float = 0.0
        raw_delta_herg: float = 0.0 
        raw_delta_bbbp: float = 0.0

        # Constants for reward scaling
        LOWER_IS_BETTER_EXP_SCALER = 5.0
        LOWER_IS_BETTER_CROSSING_BONUS = 5.0
        LOWER_IS_BETTER_PENALTY_FACTOR = 2.0

        HIGHER_IS_BETTER_EXP_SCALER = 5.0
        HIGHER_IS_BETTER_CROSSING_BONUS = 5.0
        HIGHER_IS_BETTER_PENALTY_FACTOR = 2.0

        ESOL_EXP_SCALER = 2.0
        ESOL_PENALTY_FACTOR = 1.0
        ESOL_SURPRISE_BONUS = 5.0
        ESOL_BONUS_THRESHOLD = 0.5
        
        SAFE_THRESHOLD = 0.5

        # --- 1. Ames / hERG Reward (Classification, lower is better) ---
        for prop_name, label in [("Ames", "Ames"), ("hERG_10uM", "hERG_10uM")]:
            if label in self.label_name:
                orig_val = orig_props.get(label)
                new_val = new_props.get(label)
                
                if orig_val is not None and new_val is not None:
                    delta = orig_val - new_val  # Positive means improvement
                    if prop_name == "Ames":
                        raw_delta_muta = delta
                    else: 
                        raw_delta_herg = delta 
                        raw_delta_muta = delta 
                    
                    reward_contribution = 0.0
                    if delta > 0:  
                        reward_contribution += np.exp(LOWER_IS_BETTER_EXP_SCALER * delta) - 1
                    elif delta < 0:  
                        reward_contribution += delta * LOWER_IS_BETTER_PENALTY_FACTOR
                    
                    if orig_val >= SAFE_THRESHOLD and new_val < SAFE_THRESHOLD:
                        reward_contribution += LOWER_IS_BETTER_CROSSING_BONUS
                        print(f"    [{prop_name} Reward] CROSSING BONUS applied.")

                    total_reward += reward_contribution
                    print(f"    [{prop_name} Reward] Delta = {delta:.4f}, Contribution: {reward_contribution:.4f}")

        # --- 2. BBBP Reward (Classification, higher is better) ---
        if "BBB" in self.label_name:
            orig_val = orig_props.get("BBB")
            new_val = new_props.get("BBB")

            if orig_val is not None and new_val is not None:
                raw_delta_bbbp = new_val - orig_val  
                
                reward_contribution = 0.0
                if raw_delta_bbbp > 0:  
                    reward_contribution += np.exp(HIGHER_IS_BETTER_EXP_SCALER * raw_delta_bbbp) - 1
                elif raw_delta_bbbp < 0:  
                    reward_contribution += raw_delta_bbbp * HIGHER_IS_BETTER_PENALTY_FACTOR
                
                if orig_val < SAFE_THRESHOLD and new_val >= SAFE_THRESHOLD:
                    reward_contribution += HIGHER_IS_BETTER_CROSSING_BONUS
                    print(f"    [BBB Reward] CROSSING BONUS applied.")

                total_reward += reward_contribution
                print(f"    [BBB Reward] Delta = {raw_delta_bbbp:.4f}, Contribution: {reward_contribution:.4f}")

        # --- 3. ESOL Reward (Regression, higher is better) ---
        if "logS" in self.label_name:
            orig_val_neg_logs = orig_props.get("logS")
            new_val_neg_logs = new_props.get("logS")
            
            if orig_val_neg_logs is not None and new_val_neg_logs is not None:
                raw_delta_esol = new_val_neg_logs - orig_val_neg_logs
                
                reward_contribution = 0.0
                if raw_delta_esol > 0:
                    reward_contribution += np.exp(ESOL_EXP_SCALER * raw_delta_esol) - 1
                    if raw_delta_esol > ESOL_BONUS_THRESHOLD:
                        reward_contribution += ESOL_SURPRISE_BONUS
                        print(f"    [ESOL Reward] SURPRISE BONUS applied.")
                elif raw_delta_esol < 0:
                    reward_contribution += raw_delta_esol * ESOL_PENALTY_FACTOR
                
                total_reward += reward_contribution
                print(f"    [ESOL Reward] Delta = {raw_delta_esol:.4f}, Contribution: {reward_contribution:.4f}")

        print(f"  --- Total Calculated Reward: {total_reward:.4f} ---")
        return total_reward, raw_delta_muta, raw_delta_esol, raw_delta_bbbp, orig_props, new_props

    def step(self, action_id: np.ndarray):
        self.current_step += 1
        action_id = int(action_id) 
        print(f"\n[ENV_STEP {self.current_step}/{self.max_steps}] Current Mol: {self.current_mol}, Received Action ID: {action_id}")
        info = {'action_id_raw': action_id}

        # --- 1. Validate action ---
        if not (0 <= action_id < self.max_potential_actions and self.current_action_mask_bool[action_id]):
            print(f"  ERROR: Invalid action_id {action_id} or action is masked out.")
            reward_for_sb3 = -1.0 
            terminated = True
            truncated = False
            info.update({"error": "invalid_action_id_or_masked_out", "reward_agent": reward_for_sb3})
            return self._get_current_observation(), reward_for_sb3, terminated, truncated, info

        true_indices_count = np.sum(self.current_action_mask_bool[:action_id + 1])
        relative_action_idx = true_indices_count - 1

        if not (0 <= relative_action_idx < len(self.current_opt_subs)):
            print(f"  ERROR: Action out of bounds for current_opt_subs.")
            reward_for_sb3 = -1.0 
            terminated = True
            truncated = False
            info.update({"error": "action_mapping_oob", "reward_agent": reward_for_sb3})
            return self._get_current_observation(), reward_for_sb3, terminated, truncated, info

        # --- 2. Execute action and generate new molecule ---
        opt_sub_smiles = self.current_opt_subs[relative_action_idx]
        original_smiles_before_step = self.current_mol

        mapped_new_smi_list = generate_optimized_molecules(smiles=self.current_mol, match=self.rm_atom_idx, optimized_fg_smiles=opt_sub_smiles)
        new_smi_list_filtered = list(set(convert_mapped_smiles_to_standard_smiles(s) for s in mapped_new_smi_list if s and convert_mapped_smiles_to_standard_smiles(s) != original_smiles_before_step))

        # --- 3. Evaluate new molecules and select the best ---
        base_calculated_reward = 0.0
        delta_muta, delta_esol, delta_bbbp = 0.0, 0.0, 0.0 
        
        if new_smi_list_filtered:
            candidate_evaluations = []
            for new_smi in new_smi_list_filtered:
                reward, dm, de, db, _, _ = self._calculate_reward(original_smiles_before_step, new_smi)
                candidate_evaluations.append({'smi': new_smi, 'reward': reward, 'delta_muta': dm, 'delta_esol': de, 'delta_bbbp': db})
            
            best_eval = max(candidate_evaluations, key=lambda x: x['reward'])
            self.current_mol = best_eval['smi']
            base_calculated_reward = best_eval['reward']
            delta_muta, delta_esol, delta_bbbp = best_eval['delta_muta'], best_eval['delta_esol'], best_eval['delta_bbbp']
        else:
            print("  NO NEW MOLECULES generated. State remains unchanged.")
            self.current_mol = original_smiles_before_step
            base_calculated_reward, delta_muta, delta_esol, delta_bbbp, _, _ = self._calculate_reward(original_smiles_before_step, original_smiles_before_step)

        reward_for_sb3 = base_calculated_reward

        # --- 4. Prepare next state and check termination ---
        next_obs_dict, prep_info_next = self._prepare_next_state_from_mol(self.current_mol)

        terminated_by_no_actions_next = not prep_info_next.get("actions_available", False)
        terminated_by_max_steps = self.current_step >= self.max_steps
        terminated = terminated_by_max_steps or terminated_by_no_actions_next
        truncated = False 

        # --- 5. Return results ---
        info.update({
            'new_smi': self.current_mol if self.current_mol != original_smiles_before_step else None,
            'reward_agent': reward_for_sb3,
            'delta_muta': delta_muta,
            'delta_esol': delta_esol,
            'delta_bbbp': delta_bbbp 
        })
        
        if terminated:
            if terminated_by_no_actions_next:
                reason = prep_info_next.get("error", "unknown")
                info['final_termination_reason'] = f"no_actions_for_next_state: {reason}"
            if terminated_by_max_steps:
                info['final_termination_reason'] = "max_steps_reached"

        return next_obs_dict, reward_for_sb3, terminated, truncated, info
    
    def _calculate_chemical_properties(self, smi: str) -> Dict:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            return {'valid': False, 'qed': 0, 'sascore': 10, 'logp': 0, 'mw': 0}
        return {
            'valid': True, 'qed': QED.qed(mol),
            'sascore': sascorer.calculateScore(mol) if 'sascorer' in globals() else 0.0,
            'logp': Crippen.MolLogP(mol), 'mw': Descriptors.MolWt(mol)
        }

    def render(self, mode='human'):
        if mode == 'human':
            num_available_actions = np.sum(self.current_action_mask_bool) if hasattr(self, 'current_action_mask_bool') else 0
            print(f"Render - Step: {self.current_step}, Current SMILES: {self.current_mol}, Available Actions: {num_available_actions}")

    def close(self):
        self._property_cache.clear()
        self._graph_cache.clear()


class ActionProbsCallback(BaseCallback):
    def __init__(self, print_freq: int = 1000, verbose: int = 0):
        super(ActionProbsCallback, self).__init__(verbose)
        self.print_freq = print_freq
        self.print_next_log = self.print_freq

    def _on_training_start(self) -> None:
        super()._on_training_start()
        if self.model is not None and hasattr(self.model, 'observation_space'):
            self.observation_space = self.model.observation_space
        if self.model is not None and hasattr(self.model, 'action_space'):
            self.action_space = self.model.action_space

    def _get_obs_dict_from_locals_or_buffer(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Tries to get the current observation dictionary (as numpy arrays)
        from self.locals or the rollout buffer.
        """
        current_obs_dict_tensor = None

        if 'obs' in self.locals:
            obs_from_locals = self.locals['obs']
            if isinstance(obs_from_locals, dict) and all(isinstance(v, torch.Tensor) for v in obs_from_locals.values()):
                current_obs_dict_tensor = {k: v[0].cpu() for k, v in obs_from_locals.items()} 
            elif isinstance(obs_from_locals, torch.Tensor):
                return None 

        elif hasattr(self.model, 'rollout_buffer') and \
             hasattr(self.model.rollout_buffer, 'observations') and \
             self.model.rollout_buffer.pos > 0:
            
            buffer_obs = self.model.rollout_buffer.observations
            idx = (self.model.rollout_buffer.pos - 1 + self.model.rollout_buffer.buffer_size) % self.model.rollout_buffer.buffer_size
            
            if isinstance(buffer_obs, dict) and \
               'observation' in buffer_obs and 'action_mask' in buffer_obs:
                current_obs_dict_tensor = {}
                for key in buffer_obs:
                    if isinstance(buffer_obs[key], torch.Tensor):
                        current_obs_dict_tensor[key] = buffer_obs[key][idx, 0].cpu() 
                    elif isinstance(buffer_obs[key], np.ndarray): 
                        current_obs_dict_tensor[key] = torch.from_numpy(buffer_obs[key][idx, 0]) 
                    else:
                        return None

        if current_obs_dict_tensor is None:
            return None

        try:
            obs_dict_numpy = {k: v.numpy() for k, v in current_obs_dict_tensor.items()}
            return obs_dict_numpy
        except Exception as e:
            return None


    def _on_step(self) -> bool:
        if not hasattr(self, 'observation_space') or self.observation_space is None:
             if self.model is not None and hasattr(self.model, 'observation_space'):
                self.observation_space = self.model.observation_space
             else:
                return True
        if not hasattr(self, 'action_space') or self.action_space is None:
            if self.model is not None and hasattr(self.model, 'action_space'):
                self.action_space = self.model.action_space
            else:
                return True

        if self.num_timesteps >= self.print_next_log:
            self.print_next_log = self.num_timesteps + self.print_freq

            obs_dict_np = self._get_obs_dict_from_locals_or_buffer()
            if obs_dict_np is None: return True 

            try:
                obs_tensor_dict_for_policy = {}
                for key, value_np in obs_dict_np.items():
                    expected_dims = len(self.observation_space.spaces[key].shape)
                    value_np_batched = value_np if value_np.ndim > expected_dims else value_np[np.newaxis, :]
                    
                    target_dtype = torch.float32
                    if key == "action_mask": 
                        target_dtype = torch.bool
                    
                    obs_tensor_dict_for_policy[key] = torch.as_tensor(value_np_batched, device=self.model.device, dtype=target_dtype)

                with torch.no_grad():
                    action_distribution = self.model.policy.get_distribution(obs_tensor_dict_for_policy)
                    
                    probabilities = None
                    if hasattr(action_distribution, 'distribution') and hasattr(action_distribution.distribution, 'probs'):
                        probabilities = action_distribution.distribution.probs.cpu().numpy()
                    elif hasattr(action_distribution, 'probs'): 
                        probabilities = action_distribution.probs.cpu().numpy()
                    
                    if probabilities is not None:
                        probs_to_log = probabilities[0] 
                        
                        action_mask_np = obs_dict_np.get("action_mask")
                        if action_mask_np is None:
                            num_available_actions = probs_to_log.size 
                        else:
                            if action_mask_np.ndim > 1: 
                                action_mask_np = action_mask_np.squeeze(0)
                            num_available_actions = np.sum(action_mask_np)
                        
                        if self.logger is not None:
                            if probs_to_log.size > 0:
                                self.logger.record('custom_train_action_probs/action_0_prob', probs_to_log[0])
                            
                            for i in range(1, min(probs_to_log.size, 5)): 
                                self.logger.record(f'custom_train_action_probs/action_{i}_prob', probs_to_log[i])
                            
                            self.logger.record('custom_train_action_probs/num_available_actions', num_available_actions)
                            
                            if hasattr(action_distribution, 'entropy'):
                                entropy = action_distribution.entropy() 
                                if entropy is not None:
                                     entropy_to_log = entropy.mean().item() if entropy.ndim > 0 else entropy.item()
                                     self.logger.record('custom_train_action_probs/distribution_entropy', entropy_to_log)

            except Exception as e:
                pass
        return True

    
class TensorBoardCallback(BaseCallback): 
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if isinstance(self.training_env, DummyVecEnv): 
            if "reward" in self.training_env.buf_infos[0]: 
                 self.logger.record('custom_metrics/step_reward', self.training_env.buf_infos[0]["reward"])
        return True

def get_scalar_from_tfevents(tfevents_filepath: str, scalar_tag: str) -> tuple[list, list]:
    steps, values = [], []
    try:
        ea = event_accumulator.EventAccumulator(
            tfevents_filepath,
            size_guidance={event_accumulator.SCALARS: 0}
        )
        ea.Reload()
        if scalar_tag in ea.Tags()['scalars']:
            scalar_events = ea.Scalars(scalar_tag)
            steps = [event.step for event in scalar_events]
            values = [event.value for event in scalar_events]
    except Exception as e:
        print(f"    Error reading tfevents file {tfevents_filepath}: {e}")
    return steps, values

def moving_average(data: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(data, np.ndarray): 
        data = np.array(data)
    if len(data) < window_size or window_size <= 1: 
        return np.array([]), np.array([]) 
    
    smoothed_data = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    original_indices_for_smoothed = np.arange(window_size - 1, len(data))
    return smoothed_data, original_indices_for_smoothed


def objective(trial: Trial) -> float:
    print(f"\nOptuna Trial #{trial.number}")

    current_hyperparams = Config.PPO_HYPERPARAMS.copy() 
    current_hyperparams.update({
        'learning_rate': trial.suggest_float('lr', 1e-5, 5e-4, log=True), 
        'ent_coef': trial.suggest_float('ent_coef', 1e-5, 0.005, log=True), 
        'n_steps': trial.suggest_categorical('n_steps', [512, 1024]), 
        'batch_size': trial.suggest_categorical('batch_size', [64, 128]), 
        'gamma': trial.suggest_float('gamma', 0.95, 0.999),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.98),
        'clip_range': trial.suggest_float('clip_range', 0.1, 0.3),
    })
    current_hyperparams['device'] = device_name 
    if 'policy_kwargs' not in current_hyperparams and 'policy_kwargs' in Config.PPO_HYPERPARAMS:
         current_hyperparams['policy_kwargs'] = Config.PPO_HYPERPARAMS['policy_kwargs']


    trial_log_path = os.path.join(Config.TENSORBOARD_LOG_PATH, f"optuna_trial_{trial.number}")
    os.makedirs(trial_log_path, exist_ok=True)

    def make_hpo_train_env():
        env = MoleculeOptimEnv(
            model_names=Config.MODEL_NAMES,
            smiles_list=Config.TRAIN_SMILES, 
            max_steps=Config.ENV_PARAMS['max_steps'],
            is_eval_env=True 
        )
        env = Monitor(env, trial_log_path) 
        return env

    vec_env = DummyVecEnv([make_hpo_train_env])
    
    model = MaskablePPO( 
        policy=current_hyperparams['policy'], 
        env=vec_env,
        n_steps=current_hyperparams['n_steps'],
        batch_size=current_hyperparams['batch_size'],
        n_epochs=current_hyperparams.get('n_epochs', Config.PPO_HYPERPARAMS['n_epochs']), 
        learning_rate=current_hyperparams['learning_rate'],
        ent_coef=current_hyperparams['ent_coef'],
        gamma=current_hyperparams['gamma'],
        gae_lambda=current_hyperparams['gae_lambda'],
        clip_range=current_hyperparams['clip_range'],
        vf_coef=current_hyperparams.get('vf_coef', Config.PPO_HYPERPARAMS['vf_coef']),
        max_grad_norm=current_hyperparams.get('max_grad_norm', Config.PPO_HYPERPARAMS['max_grad_norm']),
        tensorboard_log=None, 
        verbose=0, 
        device=current_hyperparams['device'],
        policy_kwargs=current_hyperparams.get('policy_kwargs')
    )

    def make_hpo_eval_env():
        env = MoleculeOptimEnv(
            model_names=Config.MODEL_NAMES,
            smiles_list=Config.TRAIN_SMILES[:max(10, len(Config.TRAIN_SMILES)//20)], 
            max_steps=Config.ENV_PARAMS['max_steps'],
            is_eval_env=True
        )
        return env
    
    eval_env_hpo_vec = DummyVecEnv([make_hpo_eval_env])

    hpo_eval_callback = MaskableEvalCallback(
        eval_env_hpo_vec,
        best_model_save_path=None, 
        log_path=trial_log_path, 
        eval_freq=max(1, Config.OPTUNA_TIMESTEPS_PER_TRIAL // current_hyperparams['n_steps'] // 2) * current_hyperparams['n_steps'], 
        n_eval_episodes=Config.N_EVAL_EPISODES // 2 or 1, 
        deterministic=True,
        render=False
    )
    
    mean_reward_eval = -float('inf')
    try:
        model.learn(
            total_timesteps=Config.OPTUNA_TIMESTEPS_PER_TRIAL,
            callback=hpo_eval_callback 
        )
        if hpo_eval_callback.best_mean_reward != -np.inf:
             mean_reward_eval = hpo_eval_callback.best_mean_reward
        elif hasattr(hpo_eval_callback, 'last_mean_reward') and hpo_eval_callback.last_mean_reward != -np.inf:
             mean_reward_eval = hpo_eval_callback.last_mean_reward
        else: 
            print(f"  Optuna Trial #{trial.number}: Manual evaluation triggered.")
            episode_rewards, _ = maskable_evaluate_policy(
                model, eval_env_hpo_vec, n_eval_episodes=(Config.N_EVAL_EPISODES // 2 or 1), 
                return_episode_rewards=True, deterministic=True
            )
            mean_reward_eval = np.mean(episode_rewards) if episode_rewards else -float('inf')

    except Exception as e:
        print(f"  Optuna Trial #{trial.number} LEARN FAILED: {e}")
        return -float('inf') 
    finally:
        vec_env.close()
        eval_env_hpo_vec.close()

    print(f"  Optuna Trial #{trial.number} finished. Eval Mean Reward: {mean_reward_eval:.4f}")
    return mean_reward_eval


def train_final_model(hyperparams: Dict):
    print("\n=== Training Final Model ===")
    log_path_final_model = os.path.join(Config.RESULT_PATH, "final_model_training_logs")
    os.makedirs(log_path_final_model, exist_ok=True)

    def make_train_env_final():
        env = MoleculeOptimEnv(
            model_names=Config.MODEL_NAMES,
            smiles_list=Config.TRAIN_SMILES,
            max_steps=Config.ENV_PARAMS['max_steps']
        )
        env = Monitor(env, log_path_final_model) 
        return env
    vec_env = DummyVecEnv([make_train_env_final])

    def make_eval_env_final():
        env = MoleculeOptimEnv(
            model_names=Config.MODEL_NAMES,
            smiles_list=Config.TRAIN_SMILES[:max(20, len(Config.TRAIN_SMILES)//10)],
            max_steps=Config.ENV_PARAMS['max_steps'],
            is_eval_env=True
        )
        return env
    eval_vec_env = DummyVecEnv([make_eval_env_final])

    early_stopping_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=Config.EARLY_STOPPING_PATIENCE,
        min_evals=Config.EARLY_STOPPING_PATIENCE + 2, 
        verbose=1
    )
    eval_callback = MaskableEvalCallback( 
        eval_vec_env,
        best_model_save_path=os.path.join(Config.MODEL_SAVE_PATH, 'best_model_final'),
        log_path=log_path_final_model, 
        eval_freq=max(1, Config.EVAL_FREQ // hyperparams['n_steps']) * hyperparams['n_steps'],
        n_eval_episodes=Config.N_EVAL_EPISODES,
        deterministic=True, 
        render=False,
        callback_after_eval=early_stopping_callback 
    )
    action_probs_callback = ActionProbsCallback(print_freq=1024, verbose=1) 
    combined_callbacks = [eval_callback, action_probs_callback]

    model = MaskablePPO(
        policy=hyperparams['policy'],
        env=vec_env,
        n_steps=hyperparams['n_steps'],
        batch_size=hyperparams['batch_size'],
        n_epochs=hyperparams['n_epochs'],
        learning_rate=hyperparams['learning_rate'],
        ent_coef=hyperparams['ent_coef'],
        gamma=hyperparams['gamma'],
        gae_lambda=hyperparams['gae_lambda'],
        clip_range=hyperparams['clip_range'],
        vf_coef=hyperparams['vf_coef'],
        max_grad_norm=hyperparams['max_grad_norm'],
        tensorboard_log=Config.TENSORBOARD_LOG_PATH, 
        verbose=1,
        device=hyperparams['device'],
        policy_kwargs=hyperparams.get('policy_kwargs')
    )
    
    actual_ppo_run_name_for_tb = "MaskablePPO_MoleculeOptim_Final" 
    print(f"Starting final model training. TB log name: {actual_ppo_run_name_for_tb}")
    model.learn(
        total_timesteps=Config.TOTAL_TIMESTEPS,
        callback=combined_callbacks,
        tb_log_name=actual_ppo_run_name_for_tb 
    )
    
    model.save(os.path.join(Config.MODEL_SAVE_PATH, "final_maskable_ppo_model_completed"))
    print(f"Final MaskablePPO model saved.")
    
    vec_env.close()
    eval_vec_env.close()
    
    return model


def evaluate_trained_model(model: 'MaskablePPO', test_smiles: List[str], n_runs_per_smiles: int = 20):
    """
    Modified evaluation function (single-target full trajectory):
    - Records information of newly generated molecules at each step.
    - Success rate and other metrics are calculated based on all molecules generated across runs.
    - Applicable to a single target within Config.MODEL_NAMES.
    """
    print(f"\n=== Evaluating Trained Model (Single Target, n_runs={n_runs_per_smiles}) ===")

    if len(Config.MODEL_NAMES) != 1:
        print(f"[ERROR] Evaluation strictly expects a single target model, given: {Config.MODEL_NAMES}")
        return
        
    target_model_name = Config.MODEL_NAMES[0]
    optimization_goal = Config.OPTIMIZATION_GOALS.get(target_model_name)

    print(f"       Target property: {target_model_name} (Direction: {optimization_goal})")

    eval_env = MoleculeOptimEnv(
        model_names=Config.MODEL_NAMES, 
        smiles_list=test_smiles, 
        max_steps=Config.ENV_PARAMS['max_steps'],
        is_eval_env=True
    )
    
    label_name = eval_env.model_to_label_map.get(target_model_name)
    if not label_name:
        print(f"[ERROR] Cannot map target model '{target_model_name}' to a valid label name.")
        eval_env.close()
        return

    all_step_data = []
    max_steps_per_episode = Config.ENV_PARAMS['max_steps']

    for smi_idx, original_smi in enumerate(test_smiles):
        print(f"\n  Evaluating original SMILES {smi_idx + 1}/{len(test_smiles)}: {original_smi}")
        
        eval_env.initial_smiles_list = [original_smi]
        eval_env.current_smiles_pool = [original_smi]

        orig_props = eval_env._predict_property(original_smi)
        orig_chem_props = eval_env._calculate_chemical_properties(original_smi)

        for run_idx in range(n_runs_per_smiles):
            obs, reset_info = eval_env.reset()
            
            if not reset_info.get("actions_available", False):
                print(f"    Run {run_idx + 1}: Initialization failed (no valid actions). Skipping.")
                continue

            current_mol_in_run = eval_env.current_mol 
            terminated, truncated = False, False
            step_count = 0

            while not (terminated or truncated) and step_count < max_steps_per_episode:
                step_count += 1
                smi_before_step = current_mol_in_run

                obs_for_model = {k: np.expand_dims(v, axis=0) for k, v in obs.items()}
                action_mask_for_model = obs_for_model.get("action_mask")
                action, _ = model.predict(obs_for_model, deterministic=False, action_masks=action_mask_for_model)
                
                obs, reward_step, terminated, truncated, info_step = eval_env.step(action)
                
                generated_smi = eval_env.current_mol

                is_novel_vs_orig = (generated_smi != original_smi)

                reward_vs_orig, d_muta, d_esol, d_bbbp, _, new_props = \
                    eval_env._calculate_reward(original_smi, generated_smi)

                delta_val = 0.0
                if target_model_name in ['Mutagenicity', 'hERG']: delta_val = d_muta
                elif target_model_name == 'ESOL': delta_val = d_esol
                elif target_model_name == 'BBBP': delta_val = d_bbbp

                new_chem_props = eval_env._calculate_chemical_properties(generated_smi)

                step_data = {
                    'original_smiles': original_smi,
                    'run_index': run_idx + 1,
                    'step_index': step_count,
                    'smiles_before_step': smi_before_step,
                    'generated_smiles': generated_smi,
                    'is_novel_vs_original': is_novel_vs_orig,
                    'recalculated_reward_vs_original': reward_vs_orig,
                    
                    f'orig_prop_{label_name}': orig_props.get(label_name, 0.0),
                    f'new_prop_{label_name}': new_props.get(label_name, 0.0),
                    f'delta_vs_original_{label_name}': delta_val,

                    **{f'orig_chem_{k}': v for k, v in orig_chem_props.items()},
                    **{f'new_chem_{k}': v for k, v in new_chem_props.items()}
                }
                all_step_data.append(step_data)

                current_mol_in_run = generated_smi

    eval_env.close()

    if not all_step_data:
        print("\n[ERROR] No evaluation results generated, unable to compute statistics.")
        return

    df_results = pd.DataFrame(all_step_data)
    
    detailed_path = os.path.join(Config.RESULT_PATH, "evaluation_detailed_full_trajectory.csv")
    df_results.to_csv(detailed_path, index=False)
    print(f"\nDetailed trajectory results saved to: {detailed_path}")

    stats = {}
    
    df_novel = df_results[df_results['is_novel_vs_original']].copy()
    num_total_generations = len(df_results)
    num_novel_generations = len(df_novel)

    # --- A. Basic Metrics ---
    stats['Total_Generation_Steps'] = num_total_generations
    stats['Num_Novel_Molecules_Generated'] = num_novel_generations
    stats['Novelty_Rate_All_Generations'] = num_novel_generations / num_total_generations if num_total_generations > 0 else 0.0

    # --- B. Improvement rate based on novel molecules ---
    if num_novel_generations > 0:
        delta_col = f'delta_vs_original_{label_name}'
        df_novel[f'is_improved_{target_model_name}'] = df_novel[delta_col] > 1e-7
        stats[f'ImprovementRate_{target_model_name}_in_Novel'] = df_novel[f'is_improved_{target_model_name}'].mean()
    else:
        stats[f'ImprovementRate_{target_model_name}_in_Novel'] = 0.0
    
    # --- C. Success Rate based on hard molecules ---
    stats['Mean_Success_Rate_Metric'] = 0.0
    metric_name = "Mean_Success_Rate_Metric" 

    orig_prop_col = f'orig_prop_{label_name}'
    new_prop_col = f'new_prop_{label_name}'

    hard_condition = pd.Series([False] * num_total_generations)
    success_condition = pd.Series([False] * num_total_generations)
    is_metric_applicable = False

    if target_model_name in ['Mutagenicity', 'hERG']:
        is_metric_applicable = True
        hard_condition = df_results[orig_prop_col] >= 0.5
        success_condition = df_results[new_prop_col] < 0.5
        metric_name = f"Mean_Success_Rate(to_Safe)_for_Hard({target_model_name})"
        
    elif target_model_name == 'BBBP':
        is_metric_applicable = True
        hard_condition = df_results[orig_prop_col] < 0.5
        success_condition = df_results[new_prop_col] >= 0.5
        metric_name = f"Mean_Success_Rate(to_Permeable)_for_Hard({target_model_name})"

    elif target_model_name == 'ESOL':
        is_metric_applicable = True
        hard_condition = pd.Series([True] * num_total_generations) 
        delta_col = f'delta_vs_original_{label_name}'
        success_condition = df_results[delta_col] > 0.5
        metric_name = "Mean_Rate_of_Large_Improvement(Delta>0.5)_AllGenerations"

    if is_metric_applicable:
        df_relevant_base = df_results[hard_condition].copy()
        
        if not df_relevant_base.empty:
            unique_relevant_starts = df_relevant_base['original_smiles'].nunique()
            print(f"\nFound {unique_relevant_starts} starting molecules applicable for calculating success metrics.")

            if unique_relevant_starts > 0:
                df_relevant_base['is_successful'] = success_condition[hard_condition]
                success_rates_per_start = df_relevant_base.groupby('original_smiles')['is_successful'].mean()
                stats[metric_name] = success_rates_per_start.mean()

    # --- D. Diversity & Chemical Property Metrics ---
    if not df_novel.empty:
        df_best_per_smi = df_novel.sort_values('recalculated_reward_vs_original', ascending=False).drop_duplicates('original_smiles')
        df_best_per_smi.rename(columns={'new_chem_valid': 'valid', 'new_chem_qed': 'qed', 'new_chem_sascore': 'sascore'}, inplace=True)

        stats['Avg_Validity_of_Best_SMILES'] = df_best_per_smi['valid'].mean()
        
        df_best_valid = df_best_per_smi[df_best_per_smi['valid']]
        if not df_best_valid.empty:
            stats['Avg_QED_of_Best_Valid_SMILES'] = df_best_valid['qed'].mean()
            stats['Avg_SAscore_of_Best_Valid_SMILES'] = df_best_valid['sascore'].mean()
        else:
            stats.update({'Avg_QED_of_Best_Valid_SMILES': 0.0, 'Avg_SAscore_of_Best_Valid_SMILES': 10.0})
        
        stats['Uniqueness_of_Best_SMILES'] = df_best_per_smi['generated_smiles'].nunique() / len(df_best_per_smi) if len(df_best_per_smi) > 0 else 0.0
        
        train_set = set(Config.TRAIN_SMILES)
        novel_against_train_count = sum(1 for s in df_best_per_smi['generated_smiles'] if s not in train_set)
        stats['Novelty_of_Best_SMILES_vs_Train'] = novel_against_train_count / len(df_best_per_smi) if len(df_best_per_smi) > 0 else 0.0

    else:
        stats.update({
            'Avg_Validity_of_Best_SMILES': 0.0, 'Avg_QED_of_Best_Valid_SMILES': 0.0, 'Avg_SAscore_of_Best_Valid_SMILES': 10.0,
            'Uniqueness_of_Best_SMILES': 0.0, 'Novelty_of_Best_SMILES_vs_Train': 0.0
        })

    # --- 3. Consolidate and Export Results ---
    ordered_keys = [
        'Total_Generation_Steps', 'Num_Novel_Molecules_Generated', 'Novelty_Rate_All_Generations',
        f'ImprovementRate_{target_model_name}_in_Novel',
    ]
    if metric_name in stats:
        ordered_keys.append(metric_name)

    ordered_keys.extend([
        'Avg_Validity_of_Best_SMILES', 'Avg_QED_of_Best_Valid_SMILES', 'Avg_SAscore_of_Best_Valid_SMILES',
        'Uniqueness_of_Best_SMILES', 'Novelty_of_Best_SMILES_vs_Train'
    ])

    final_stats = {key: stats.get(key, 0.0) for key in ordered_keys}

    summary_path = os.path.join(Config.RESULT_PATH, "evaluation_summary_full_trajectory.csv")
    pd.DataFrame([final_stats]).to_csv(summary_path, index=False)
    print(f"\nComprehensive trajectory evaluation summary saved to: {summary_path}")
    
    print(f"\n--- Final Evaluation Summary ({target_model_name}) ---")
    for key, value in final_stats.items():
        print(f"  {key:<60}: {value:.4f}" if isinstance(value, (float, np.floating)) else f"  {key:<60}: {value}")


if __name__ == "__main__":
    Config.setup_directories() 

    # --- Use fixed hyperparameters ---
    print("\nUsing fixed hyperparameters for training...")
    fixed_hyperparams = {
        'policy': 'MultiInputPolicy', 
        'n_steps': 1024,
        'batch_size': 128,
        'n_epochs': 10,        
        'learning_rate': 1e-5,
        'ent_coef': 0.05,
        'gamma': 0.99,
        'gae_lambda': 0.964,
        'clip_range': 0.2,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'device': device_name, 
        'policy_kwargs': dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])
    }
    
    best_hpo_params = fixed_hyperparams 

    pd.Series(best_hpo_params).to_csv(os.path.join(Config.RESULT_PATH, "fixed_hyperparams_used.csv"))

    # --- Train Final Model with Best Hyperparameters ---
    print("\nTraining final model with MaskablePPO...")
    final_model = train_final_model(best_hpo_params)
    
    evaluate_trained_model(final_model, Config.TEST_SMILES, n_runs_per_smiles=20) 

    print("\nExperiment finished.")