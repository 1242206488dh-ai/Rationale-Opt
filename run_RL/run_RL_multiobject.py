--- START OF FILE Paste March 04, 2026 - 4:46PM ---

from rdkit import Chem
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer # Ensure this import works in your environment

from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import QED, Descriptors, Crippen 
from rdkit.Chem import AllChem
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import time

from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env 
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv 
from stable_baselines3.common.callbacks import BaseCallback 
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure as sb3_configure_logger
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.results_plotter import load_results, ts2xy

# --- SB3 Contrib Imports (Core Components) ---
from sb3_contrib import MaskablePPO 
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy 
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback 
from sb3_contrib.common.maskable.evaluation import evaluate_policy as maskable_evaluate_policy 
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement 

from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt 

from optuna import Trial, create_study
import optuna
from typing import Dict, List, Tuple, Optional, Any, Callable 

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

# 1. Device Setup
if torch.cuda.is_available():
    print('use GPU')
    device_name = 'cuda' 
else:
    print('use CPU')
    device_name = 'cpu'
device = torch.device(device_name) 

# 2. Global Random Seed Configuration
seed = 43
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 3. Featurizer Initialization
from model3.featurizers import CanonicalAtomFeaturizer
from model3.featurizers import CanonicalBondFeaturizer
atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='hv')
bond_featurizer = CanonicalBondFeaturizer(bond_data_field='he', self_loop=True)

# 4. Main Configuration Class
class Config:
    BASE_PATH = '/HOME/scz4306/run/SME/optimization'
    
    # Models and Optimization Targets
    MODEL_NAMES = ['Mutagenicity', 'BBBP']
    OPTIMIZATION_GOALS = {
        'Mutagenicity': 'lower',
        'ESOL': 'higher',
        'hERG': 'lower',
        'BBBP': 'higher' 
    }

    # Directory placeholders (Set by setup_directories)
    RESULT_PATH = ''
    TENSORBOARD_LOG_PATH = ''
    MODEL_SAVE_PATH = ''

    # Environment and Action space parameters
    ACTION_BUILDER_PARAMS = {
        'embedding_sim_threshold': 0.2,
        'atom_num_ratio_range': (5 / 6, 13 / 6),
    }

    ENV_PARAMS = {
        'max_steps': 10,
        'max_potential_actions': 600
    }

    PPO_HYPERPARAMS = {
        'policy': 'MultiInputPolicy',
        'n_steps': 1024,
        'batch_size': 128,
        'n_epochs': 10,
        'learning_rate': 3e-4,
        'ent_coef': 0.001,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'device': device_name,
    }

    RULE_LOADING_STRATEGY = 0
    TOTAL_TIMESTEPS = 200000
    OPTUNA_N_TRIALS = 1
    OPTUNA_TIMESTEPS_PER_TRIAL = 2000
    EVAL_FREQ = 5000
    EARLY_STOPPING_PATIENCE = 10
    N_EVAL_EPISODES = 5

    SME_MODEL_PATH = '/HOME/scz4306/run/SME/optimization/RL_multiobject/model3/2model_817.pth'
    TRAIN_SMILES = []
    TEST_SMILES = []

    @staticmethod
    def _check_task_name(task_name_str: str, target_models: List[str]) -> bool:
        if not isinstance(task_name_str, str):
            return False
        return set(task_name_str.split()) == set(target_models)

    @staticmethod
    def setup_directories():
        """Auto-configure and create required directories based on model names"""
        model_tag = "_".join(sorted(Config.MODEL_NAMES))
        base = os.path.join(Config.BASE_PATH, f'RL_multiobject/result_{model_tag}_pubchemlib_sme_data_(1)')

        Config.RESULT_PATH = base
        Config.TENSORBOARD_LOG_PATH = os.path.join(base, 'tensorboard_logs')
        Config.MODEL_SAVE_PATH = os.path.join(base, 'models')

        for path in [Config.RESULT_PATH, Config.TENSORBOARD_LOG_PATH, Config.MODEL_SAVE_PATH]:
            os.makedirs(path, exist_ok=True)

# 5. Dataset Loader
def load_datasets_into_config():
    """Loads and merges training/testing datasets into the Config object"""
    print("\n[INFO] Loading training and test datasets...")
    target_tasks = set(Config.MODEL_NAMES)
    print(f"       Target tasks: {target_tasks}")

    all_train_smiles = []
    smiles_column_name = 'smiles'

    # Load training data
    for model_name in Config.MODEL_NAMES:
        data_path = f'/HOME/scz4306/run/SME/optimization/data/origin_data/{model_name}.csv'
        print(f"\n--- Processing file: {data_path} ---")

        try:
            df = pd.read_csv(data_path)
        except FileNotFoundError:
            print(f"[ERROR] File not found: {data_path}. Skipping.")
            continue
        except Exception as e:
            print(f"[ERROR] Error reading file {data_path}: {e}. Skipping.")
            continue

        train_df = df[df['group'].isin(['training', 'valid'])]
        if not train_df.empty:
            train_smiles = train_df[smiles_column_name].dropna().tolist()
            all_train_smiles.extend(train_smiles)
            print(f"       Found {len(train_smiles)} train/valid samples in '{model_name}.csv'.")
        else:
            print(f"       No train/valid samples found in '{model_name}.csv'.")

    # Load testing data
    test_path = '/HOME/scz4306/run/SME/optimization/data/origin_data/test_for_ADMET.csv'
    try:
        df_test = pd.read_csv(test_path)
        test_df = df_test[
            df_test['task_name'].apply(lambda x: Config._check_task_name(x, Config.MODEL_NAMES))
        ]
        Config.TEST_SMILES = test_df['src_smi'].dropna().unique().tolist()
        print(f"\n[INFO] Test dataset loaded from {test_path}.")
        print(f"       Total unique test molecules: {len(Config.TEST_SMILES)}")
    except Exception as e:
        print(f"[ERROR] Failed to read test dataset {test_path}: {e}")
        Config.TEST_SMILES = []

    # Finalize and deduplicate datasets
    print("\n--- Finalizing merged datasets ---")
    if all_train_smiles:
        train_smiles_set = set(all_train_smiles)
        test_smiles_set = set(Config.TEST_SMILES)

        final_train_smiles = sorted(list(train_smiles_set - test_smiles_set)) 
        Config.TRAIN_SMILES = final_train_smiles

        print(f"[SUCCESS] Training dataset loaded.")
        print(f"          Total unique training molecules (excluding test overlaps): {len(Config.TRAIN_SMILES)}")
    else:
        print("[WARNING] No training molecules were loaded.")

# Initialize directories and data configurations
Config.setup_directories()
load_datasets_into_config()

# 6. Rule Manager Class
class RuleManager:
    """Manages fragment libraries, embeddings, and transformation rules"""
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
            if len(sampled_fragments) < 100:
                raise ValueError(f"Insufficient fragments, only found {len(sampled_fragments)}")

            selected_fragments = random.sample(sampled_fragments, 1300)
            self.all_fragments.update(selected_fragments)
            print(f"✅ Randomly selected fragments from {lib_path} (Total unique: {len(self.all_fragments)})")

        except FileNotFoundError:
            print(f"❌ Fragment library file not found: {lib_path}")
        except Exception as e:
            print(f"❌ Failed to load {lib_path}: {str(e)}")

        if self.all_fragments:
            self.potential_actions_library = sorted(list(self.all_fragments))
            self.max_n_actions = len(self.potential_actions_library)
            if Config.ENV_PARAMS['max_potential_actions'] < self.max_n_actions:
                print(f"⚠️ Config.ENV_PARAMS['max_potential_actions'] ({Config.ENV_PARAMS['max_potential_actions']}) "
                      f"is less than loaded fragments ({self.max_n_actions}). "
                      f"Will use the actual loaded amount.")
            elif Config.ENV_PARAMS['max_potential_actions'] > self.max_n_actions:
                 print(f"ℹ️ Config.ENV_PARAMS['max_potential_actions'] ({Config.ENV_PARAMS['max_potential_actions']}) "
                      f"is greater than loaded fragments ({self.max_n_actions}). Unused action slots will exist.")

            print(f"Total fragment count (action library size): {self.max_n_actions}")
            example_frags = self.potential_actions_library[:3] if self.max_n_actions >=3 else self.potential_actions_library
            print(f"Example fragments: {example_frags}")
        else:
            print(f"⚠️ No BRICS fragments loaded. ActionBuilder may fail.")
            self.max_n_actions = 0
            if Config.ENV_PARAMS['max_potential_actions'] == 0 :
                print("ERROR: max_potential_actions is 0 due to no fragments loaded.")

    def _load_resources(self):
        print("🔧 Loading hyperparameters and rule resources (RuleManager)...")
        self.hyperparameter: Dict[str, Dict] = {}
        all_rule_dfs_processed: List[pd.DataFrame] = [] 

        counts_from_strategy1: Dict[str, int] = {}

        for name in self.model_names:
            print(f"➡ Loading model resources: {name}")
            # Load hyperparameters
            hyper_path = os.path.join(Config.BASE_PATH, f'result/hyperparameter_{name}.pkl')
            try:
                with open(hyper_path, 'rb') as f:
                    self.hyperparameter[name] = pkl.load(f)
            except Exception:
                self.hyperparameter[name] = {}

            # Load rule libraries
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
                    unique_keys = pd.Index([])

                processed_df_for_model = pd.DataFrame()

                # 7. Rule Loading Strategy Implementation
                if Config.RULE_LOADING_STRATEGY == 1:
                    print(f"  Strategy 1 (BIASED_UNIQUE) applied to {name}...")
                    if 'gain_AB_avg(%)' in current_rule_df.columns and 'rule_key' in current_rule_df.columns:
                        goal = Config.OPTIMIZATION_GOALS.get(name, 'higher')
                        if goal == 'lower': 
                            processed_df_for_model = current_rule_df[
                                (current_rule_df['rule_key'].isin(unique_keys)) &
                                (current_rule_df['gain_AB_avg(%)'] < 0)
                            ]
                            print(f"    {name} (goal: lower): Filtered {len(processed_df_for_model)} unique rules with gain < 0.")
                        else: 
                            processed_df_for_model = current_rule_df[
                                (current_rule_df['rule_key'].isin(unique_keys)) &
                                (current_rule_df['gain_AB_avg(%)'] > 0)
                            ]
                            print(f"    {name} (goal: higher): Filtered {len(processed_df_for_model)} unique rules with gain > 0.")
                        counts_from_strategy1[name] = len(processed_df_for_model)
                    else:
                        processed_df_for_model = current_rule_df
                        counts_from_strategy1[name] = len(processed_df_for_model)

                elif Config.RULE_LOADING_STRATEGY == 2:
                    print(f"  Strategy 2 (RANDOM_MATCHING_COUNT) applied to {name}...")
                    processed_df_for_model = current_rule_df

                elif Config.RULE_LOADING_STRATEGY == 0:
                    print(f"  Strategy 0 (NONE) applied to {name}...")
                    if 'gain_AB_avg(%)' in current_rule_df.columns and 'rule_key' in current_rule_df.columns:
                        temp_df = current_rule_df.copy()
                        goal = Config.OPTIMIZATION_GOALS.get(name, 'higher')
                        if goal == 'lower': 
                            temp_df = temp_df[~((temp_df['rule_key'].isin(unique_keys)) & (temp_df['gain_AB_avg(%)'] > 0))]
                        else: 
                            temp_df = temp_df[~((temp_df['rule_key'].isin(unique_keys)) & (temp_df['gain_AB_avg(%)'] < 0))]
                        processed_df_for_model = temp_df
                        print(f"    {name}: Applied default exclusion logic, remaining rules: {len(processed_df_for_model)}.")
                    else:
                        processed_df_for_model = current_rule_df

                else:
                    processed_df_for_model = current_rule_df

                all_rule_dfs_processed.append(processed_df_for_model)
            except Exception as e:
                print(f"  ❌ Failed to process rule library {rule_path}: {e}")

        # Combine processed rule dictionaries
        if not all_rule_dfs_processed:
            self.rule_df = pd.DataFrame()
            print("⚠️ All rule files failed to load or are empty!")
        else:
            combined_df = pd.concat(all_rule_dfs_processed, ignore_index=True)

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
                            goal = Config.OPTIMIZATION_GOALS.get(name_s1, 'higher')
                            if goal == 'lower':
                                count = len(df_s1[(df_s1['rule_key'].isin(unique_keys_s1)) & (df_s1['gain_AB_avg(%)'] < 0)])
                            else:
                                count = len(df_s1[(df_s1['rule_key'].isin(unique_keys_s1)) & (df_s1['gain_AB_avg(%)'] > 0)])
                        _counts_strat1_temp[name_s1] = count
                    except:
                        _counts_strat1_temp[name_s1] = 0

                final_dfs_for_concat_strat2 = []
                for name_model in self.model_names:
                    rule_path_full = os.path.join(Config.BASE_PATH, f'prediction/gain/{name_model}_brics_gain.csv')
                    try:
                        df_full = pd.read_csv(rule_path_full)
                        target_count = _counts_strat1_temp.get(name_model, 0)
                        if target_count > 0 and len(df_full) > 0:
                            sample_n = min(target_count, len(df_full))
                            sampled_df = df_full.sample(n=sample_n, random_state=42)
                            final_dfs_for_concat_strat2.append(sampled_df)
                    except Exception as e_load:
                         print(f"    Failed to load full rules for {name_model}: {e_load}")

                if final_dfs_for_concat_strat2:
                    combined_df = pd.concat(final_dfs_for_concat_strat2, ignore_index=True)
                else:
                    combined_df = pd.DataFrame()

            # Remove duplicate rules
            if not combined_df.empty and 'antecedents' in combined_df.columns and 'consequents' in combined_df.columns:
                num_total_rules = len(combined_df)
                num_unique_rules = combined_df.drop_duplicates(subset=['antecedents', 'consequents']).shape[0]
                num_duplicate_rules = num_total_rules - num_unique_rules
                print(f"ℹ️ Detected {num_duplicate_rules} duplicate rules.")

                self.rule_df = combined_df.drop_duplicates(subset=['antecedents', 'consequents']).reset_index(drop=True)
            else:
                self.rule_df = combined_df

            if not self.rule_df.empty:
                print(f"✅ Final rule count (Strategy {Config.RULE_LOADING_STRATEGY}): {len(self.rule_df)}")
            else:
                print("⚠️ No available rules after processing!")

        # Load fragment embedding dictionary
        self.frag_dict: Dict[str, List[float]] = {}
        frag_emb_path = os.path.join(Config.BASE_PATH, f'RL_pubchem/embedding/PubChem+SAR3.0_frag_emb.csv')
        try:
            frag_df = pd.read_csv(frag_emb_path)
            frag_df['embedding'] = frag_df['embedding'].apply(eval)
            self.frag_dict = dict(zip(frag_df['smiles'], frag_df['embedding']))
            print(f"✅ Fragment dictionary loaded successfully, size: {len(self.frag_dict)}")
        except Exception as e:
            print(f"  ❌ Failed to process fragment embeddings {frag_emb_path}: {e}")

# 8. Utility Functions
def get_heavy_atom_count(smiles: str) -> int:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return 0
        return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    except: return 0

def calculate_tanimoto(smi1: str, smi2: str) -> float:
    try:
        mol1, mol2 = Chem.MolFromSmiles(smi1.replace('*','')), Chem.MolFromSmiles(smi2.replace('*',''))
        if not mol1 or not mol2: return 0.0
        fp1, fp2 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, 1024), AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 1024)
        return TanimotoSimilarity(fp1, fp2)
    except: return 0.0

# 9. Action Builder Class
class ActionBuilder:
    """Constructs valid action masks based on chemical validity and embeddings"""
    def __init__(self, rule_manager: RuleManager): 
        self.rule_manager = rule_manager
        self.params = Config.ACTION_BUILDER_PARAMS
        self.frag_to_idx_map = {frag: i for i, frag in enumerate(self.rule_manager.potential_actions_library)}
        print(f"ActionBuilder initialized parameters: {self.params}")
        print(f"ActionBuilder potential action library size: {self.rule_manager.max_n_actions}")

    def build_actions(self, context_emb_list: List[np.ndarray], sub_emb: np.ndarray,
                      target_num_attachment_points: int, exclude_frag: Optional[str] = None): 
        print(f"\n=== Building Action Masks and Valid Fragments (ActionBuilder) ===")
        print(f"  Inputs: target_attachments={target_num_attachment_points}, exclude_frag='{exclude_frag}', num_context_embs={len(context_emb_list)}")

        action_mask = np.zeros(self.rule_manager.max_n_actions, dtype=bool)

        if target_num_attachment_points == 0 and exclude_frag and '*' not in exclude_frag:
            print(f"  ⚠️ target_attachments is 0 and exclude_frag looks like a full molecule. No actions generated.")
            return action_mask, [] 

        primary_filtered_mask = self._primary_filter_to_mask(
            target_num_attachment_points=target_num_attachment_points,
            exclude_frag=exclude_frag
        )
        num_after_primary = np.sum(primary_filtered_mask)
        print(f"  Valid mask count after primary filter: {num_after_primary}")
        if num_after_primary == 0:
            return action_mask, []

        final_candidate_mask = self._embedding_filter_on_mask(
            initial_mask=primary_filtered_mask,
            context_emb_list=context_emb_list,
            sub_emb_for_combined=sub_emb
        )
        num_after_embedding = np.sum(final_candidate_mask)
        print(f"  Valid mask count after embedding filter: {num_after_embedding}")
        if num_after_embedding == 0:
            return action_mask, []

        valid_frag_smiles_for_step = []
        true_indices = np.where(final_candidate_mask)[0]
        
        for frag_idx in true_indices:
            frag_smiles = self.rule_manager.potential_actions_library[frag_idx]
            if frag_smiles in self.rule_manager.frag_dict:
                valid_frag_smiles_for_step.append(frag_smiles)
            else:
                print(f"    CRITICAL WARNING: Fragment '{frag_smiles}' (idx {frag_idx}) not in frag_dict. Removing from mask.")
                final_candidate_mask[frag_idx] = False 

        if np.sum(final_candidate_mask) != len(valid_frag_smiles_for_step):
            print(f"    CRITICAL WARNING: Mismatch in mask sum. Rebuilding SMILES list.")
            valid_frag_smiles_for_step = [
                self.rule_manager.potential_actions_library[i] 
                for i in np.where(final_candidate_mask)[0] 
                if self.rule_manager.potential_actions_library[i] in self.rule_manager.frag_dict
            ]

        print(f"  Final valid fragment SMILES count: {len(valid_frag_smiles_for_step)}")
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
            print("    ⚠️ Embedding filter skipped: No candidate fragments or rules available.")
            return initial_mask 

        rule_embs_list = [
            np.array(e, dtype=np.float32)
            for e in self.rule_manager.rule_df['embedding'].tolist()
            if isinstance(e, (list, np.ndarray)) and hasattr(e, '__len__') and len(e) > 0 and np.array(e).ndim > 0
        ]
        if not rule_embs_list:
            print("    ⚠️ No valid embedding vectors in rule library, skipping similarity calculation.")
            return initial_mask
        try:
            rule_embs_matrix = np.array(rule_embs_list)
            if rule_embs_matrix.ndim == 1 and rule_embs_matrix.shape[0] > 0 : 
                 rule_embs_matrix = rule_embs_matrix.reshape(1, -1)
            elif rule_embs_matrix.ndim == 0 or rule_embs_matrix.shape[0] == 0 or rule_embs_matrix.size == 0:
                print("    ⚠️ Rule embedding matrix empty, skipping filter.")
                return initial_mask
        except ValueError as e:
            print(f"    ❌ Failed to create rule embedding matrix: {e}. Skipping filter.")
            return initial_mask

        embedding_passed_mask = np.copy(initial_mask)
        print(f"  ActionBuilder._embedding_filter: Context embs input count: {len(context_emb_list)}")

        for frag_idx in np.where(initial_mask)[0]:
            frag_smiles = self.rule_manager.potential_actions_library[frag_idx]

            if frag_smiles not in self.rule_manager.frag_dict:
                embedding_passed_mask[frag_idx] = False 
                continue

            frag_emb = np.array(self.rule_manager.frag_dict[frag_smiles], dtype=np.float32)
            if not (frag_emb.ndim == 1 and frag_emb.shape[0] == 128):
                print(f"    ❌ Invalid frag_emb shape {frag_emb.shape} for fragment '{frag_smiles}'.")
                embedding_passed_mask[frag_idx] = False
                continue
            
            try:
                all_embs_for_combined = []
                for ctx_e in context_emb_list: 
                    all_embs_for_combined.append(ctx_e)
                all_embs_for_combined.append(frag_emb) 

                if not all_embs_for_combined: 
                    embedding_passed_mask[frag_idx] = False
                    continue
                
                combined_emb_stacked = np.stack(all_embs_for_combined, axis=0) 
                combined_emb = np.mean(combined_emb_stacked, axis=0) 
                
                sim_vector = cosine_similarity(combined_emb[np.newaxis, :], rule_embs_matrix)[0]
                max_sim = np.max(sim_vector)

                if max_sim < self.params['embedding_sim_threshold']:
                    embedding_passed_mask[frag_idx] = False
            except Exception as e:
                print(f"    ❌ Error during combined_emb or similarity calculation: {e}")
                embedding_passed_mask[frag_idx] = False
                continue
        
        return embedding_passed_mask

# 10. Core Gym Environment
class MoleculeOptimEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, model_names: List[str], smiles_list: List[str],
                 max_steps: int = Config.ENV_PARAMS['max_steps'],
                 is_eval_env: bool = False):
        super().__init__()
        self.model_names = model_names
        self.max_steps = max_steps
        self.initial_smiles_list = smiles_list
        self.current_smiles_pool = list(smiles_list)
        self.is_eval_env = is_eval_env
        self.last_reset_info: Dict = {}
        
        self.label_name = []
        self.model_to_label_map = {"Mutagenicity": "Ames", "ESOL": "logS", "hERG": "hERG_10uM", "BBBP": "BBB"}
        for name in self.model_names:
            if name in self.model_to_label_map:
                self.label_name.append(self.model_to_label_map[name])
        if not self.is_eval_env: 
            print(f"Env targeting properties: {self.label_name}")
        
        # Initialize core rule and action mapping components
        self.rule_manager = RuleManager(self.model_names)
        self._sme_hyperparams = {model: self.rule_manager.hyperparameter.get(model, {}) for model in self.model_names}
        self.max_potential_actions = self.rule_manager.max_n_actions
        if self.max_potential_actions == 0: raise ValueError("RuleManager failed to load any actions.")
        self.action_builder = ActionBuilder(self.rule_manager)
        self.property_predictor = self._load_property_predictor()
        
        self._property_cache, self._graph_cache = {}, {}

        # Define Observation and Action spaces
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(128,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.max_potential_actions,), dtype=np.bool_)
        })
        self.action_space = spaces.Discrete(self.max_potential_actions)
        
        # Dynamic state tracking properties
        self.current_step = 0
        self.original_mol_for_episode: Optional[str] = None
        self.current_mol: Optional[str] = None
        self.current_state_numeric = np.zeros(self.observation_space["observation"].shape, dtype=np.float32)
        self.current_action_mask_bool = np.zeros(self.max_potential_actions, dtype=bool)
        
        # Current active modification site features
        self.rm_atom_idx: Optional[List[int]] = None
        self.sub_connect_num: Optional[int] = None
        self.current_opt_subs: List[str] = []
        self.active_model_name: Optional[str] = None 

    def _collect_and_process_sme_outputs(self, smiles: str) -> List[Dict]:
        """Collect SME outputs for each model and format them into a list of dictionaries"""
        processed_outputs = []
        KNOWN_SME_PARAMS = ['rgcn_hidden_feats', 'ffn_hidden_feats', 'lr', 'classification', 'mode']

        for model_name in self.model_names:
            sme_hp_original = self._sme_hyperparams.get(model_name, {})
            valid_sme_params = {k: sme_hp_original[k] for k in KNOWN_SME_PARAMS if k in sme_hp_original}
            expected_mode = Config.OPTIMIZATION_GOALS.get(model_name)
            if not expected_mode: continue
            valid_sme_params['mode'] = expected_mode

            try:
                sme_results = SME_opt_sub_detect(smiles=smiles, model_name=model_name, **valid_sme_params)
                if not sme_results or sme_results == [-1, -1]: continue

                for result in sme_results:
                    _pred_val, sub_smi, _, rm_idx, _, connect_num, sub_emb, ctx_emb_list = result
                    if sub_smi == -1 or not sub_smi: continue
                    
                    processed_outputs.append({
                        "model_name": model_name, 
                        "sub_smi": sub_smi,
                        "rm_idx": rm_idx,
                        "connect_num": connect_num,
                        "sub_emb": sub_emb,
                        "ctx_emb_list": ctx_emb_list
                    })
            except Exception as e:
                print(f"  [ERROR] SME detection for '{model_name}' failed: {e}")

        return processed_outputs

    def _prepare_state_from_one_sme_output(self, sme_output: Dict) -> bool:
        """Attempt to construct current environment state based on a single SME output"""
        sub_smi = sme_output["sub_smi"]
        sub_emb = sme_output["sub_emb"]
        ctx_emb_list = sme_output["ctx_emb_list"]
        
        valid_ctx_embs = [e for e in (ctx_emb_list if isinstance(ctx_emb_list, list) else [ctx_emb_list]) if isinstance(e, np.ndarray) and e.ndim == 1]
        if not valid_ctx_embs or not (isinstance(sub_emb, np.ndarray) and sub_emb.ndim == 1):
            return False

        action_mask, valid_frags = self.action_builder.build_actions(
            context_emb_list=valid_ctx_embs, sub_emb=sub_emb,
            target_num_attachment_points=sme_output["connect_num"],
            exclude_frag=sub_smi
        )

        if np.any(action_mask):
            all_embs = valid_ctx_embs + [sub_emb]
            self.current_state_numeric = np.mean(np.stack(all_embs, axis=0), axis=0).astype(np.float32)
            self.current_action_mask_bool = action_mask
            self.current_opt_subs = valid_frags
            
            self.rm_atom_idx = sme_output["rm_idx"]
            self.sub_connect_num = sme_output["connect_num"]
            self.active_model_name = sme_output["model_name"]
            
            print(f"    [DEBUG] State prepared for '{self.active_model_name}'. Available actions: {len(valid_frags)}")
            return True
            
        return False

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        super().reset(seed=seed)
        self.current_step = 0
        
        for attempt in range(len(self.initial_smiles_list) or 1):
            if not self.current_smiles_pool:
                self.current_smiles_pool = list(self.initial_smiles_list)
                if not self.current_smiles_pool: break
                random.shuffle(self.current_smiles_pool)
            
            self.original_mol_for_episode = self.current_smiles_pool.pop(0)
            self.current_mol = self.original_mol_for_episode
            print(f"\n--- ENV RESET ATTEMPT {attempt+1}: Starting SMILES: {self.current_mol} ---")

            processed_sme_outputs = self._collect_and_process_sme_outputs(self.current_mol)
            if not processed_sme_outputs:
                print("  [WARNING] No valid SME outputs found. Trying next molecule.")
                continue

            random.shuffle(processed_sme_outputs)
            for sme_output in processed_sme_outputs:
                if self._prepare_state_from_one_sme_output(sme_output):
                    print(f"--- ENV RESET SUCCESS ---")
                    self.last_reset_info = {"actions_available": True, "error": ""}
                    return self._get_current_observation(), self.last_reset_info
            
            print(f"  [WARNING] All {len(processed_sme_outputs)} SME outputs failed state generation. Trying next.")

        print("[CRITICAL] Failed to initialize a valid state after exhausting SMILES pool.")
        self.current_state_numeric.fill(0)
        self.current_action_mask_bool.fill(False)
        self.last_reset_info = {"error": "max_reset_attempts_reached", "actions_available": False}
        return self._get_current_observation(), self.last_reset_info

    def _prepare_next_state_from_mol(self, smiles: str) -> Dict:
        print(f"\n--- PREPARING NEXT STATE for: {smiles} ---")
        processed_sme_outputs = self._collect_and_process_sme_outputs(smiles)
        if not processed_sme_outputs:
            self.current_action_mask_bool.fill(False)
            return {"actions_available": False}

        random.shuffle(processed_sme_outputs)
        for sme_output in processed_sme_outputs:
            if self._prepare_state_from_one_sme_output(sme_output):
                return {"actions_available": True}

        self.current_action_mask_bool.fill(False)
        return {"actions_available": False}

    # 11. Multi-Objective Reward Calculation 
    def _calculate_reward(self, original_smi: str, new_smi: str) -> Tuple[float, Dict[str, float], Dict, Dict]:
        orig_props = self._predict_property(original_smi)
        new_props = self._predict_property(new_smi)

        raw_deltas = {name: 0.0 for name in self.model_names}
        
        # Reward definitions and constraints mapping
        PENALTY_FOR_HARM = -10.0
        SYNERGY_SUCCESS_SUPER_BONUS = 10.0
        
        PROPERTY_CONFIG = {
            'Mutagenicity': {
                'goal': 'lower',
                'label': 'Ames',
                'bad_threshold': 0.5,
                'scaler': 5.0
            },
            'hERG': {
                'goal': 'lower',
                'label': 'hERG_10uM',
                'bad_threshold': 0.5,
                'scaler': 5.0
            },
            'ESOL': {
                'goal': 'higher',
                'label': 'logS',
                'min_delta_for_success': 0.5,
                'scaler': 1.0
            },
            'BBBP': {
                'goal': 'higher',
                'label': 'BBB',
                'good_threshold': 0.5, 
                'scaler': 5.0         
            }
        }

        prop_scores = {}
        synergy_conditions_met = {}
        is_any_prop_worsened = False

        for model_name in self.model_names:
            config = PROPERTY_CONFIG.get(model_name)
            if not config: continue

            label = config['label']
            orig_val = orig_props.get(label)
            new_val = new_props.get(label)
            
            prop_scores[model_name] = 0.0
            synergy_conditions_met[model_name] = False
            
            if orig_val is None or new_val is None:
                is_any_prop_worsened = True
                continue

            delta = (new_val - orig_val) if config['goal'] == 'higher' else (orig_val - new_val)
            raw_deltas[model_name] = delta
            
            if delta < -1e-6:
                is_any_prop_worsened = True
                print(f"    [REWARD-Check] HARM DETECTED for {model_name}. Delta: {delta:.4f}")
                break

            if delta > 0:
                score = np.exp(config['scaler'] * delta) - 1
                prop_scores[model_name] = score
            
            is_success = False
            if config['goal'] == 'lower':
                if orig_val >= config['bad_threshold'] and new_val < config['bad_threshold']:
                    is_success = True
            elif config['goal'] == 'higher':
                if 'min_delta_for_success' in config: 
                    if delta > config['min_delta_for_success']:
                        is_success = True
                elif 'good_threshold' in config: 
                    if orig_val < config['good_threshold'] and new_val >= config['good_threshold']:
                        is_success = True
            synergy_conditions_met[model_name] = is_success

        if is_any_prop_worsened:
            print(f"    [REWARD-Final] Harm detected. Applying penalty: {PENALTY_FOR_HARM}")
            return PENALTY_FOR_HARM, raw_deltas, orig_props, new_props

        total_reward = 1.0
        for model_name in self.model_names:
            total_reward *= (1 + prop_scores.get(model_name, 0.0))
        total_reward -= 1.0

        if synergy_conditions_met and all(synergy_conditions_met.values()):
            total_reward += SYNERGY_SUCCESS_SUPER_BONUS

        return total_reward, raw_deltas, orig_props, new_props

    def step(self, action_id: np.ndarray):
        self.current_step += 1
        action_id = int(action_id)
        print(f"\n--- ENV STEP {self.current_step}: Action ID {action_id} on '{self.active_model_name}' site ---")

        if not (0 <= action_id < self.max_potential_actions and self.current_action_mask_bool[action_id]):
            return self._get_current_observation(), 0.0, True, False, {"error": "invalid_action"}

        true_indices_count = np.sum(self.current_action_mask_bool[:action_id + 1])
        relative_action_idx = true_indices_count - 1
        if not (0 <= relative_action_idx < len(self.current_opt_subs)):
            return self._get_current_observation(), 0.0, True, False, {"error": "action_mapping_oob"}

        opt_sub_smiles = self.current_opt_subs[relative_action_idx]
        original_smiles_before_step = self.current_mol
        print(f"  Action maps to fragment: '{opt_sub_smiles}'")
        
        mapped_new_smi_list = generate_optimized_molecules(smiles=self.current_mol, match=self.rm_atom_idx, optimized_fg_smiles=opt_sub_smiles)
        new_smi_list_filtered = list(set(convert_mapped_smiles_to_standard_smiles(s) for s in mapped_new_smi_list if s and convert_mapped_smiles_to_standard_smiles(s) != original_smiles_before_step))
        print(f"  Generated {len(new_smi_list_filtered)} new unique molecules.")

        best_smi_this_step, final_reward = original_smiles_before_step, 0.0
        final_deltas = {name: 0.0 for name in self.model_names}

        if new_smi_list_filtered:
            evaluations = []
            for i, new_smi in enumerate(new_smi_list_filtered):
                reward, deltas, _, _ = self._calculate_reward(original_smiles_before_step, new_smi)
                evaluations.append({'smi': new_smi, 'reward': reward, 'deltas': deltas})
                print(f"    Candidate {i+1}: '{new_smi[:30]}...' -> Reward: {reward:.4f}, Deltas: {deltas}")

            if evaluations:
                best_eval = max(evaluations, key=lambda x: x['reward'])
                best_smi_this_step, final_reward, final_deltas = best_eval['smi'], best_eval['reward'], best_eval['deltas']
                print(f"  --- Best new molecule is '{best_smi_this_step}' with reward {final_reward:.4f} ---")
        else:
            print("  No new molecules generated, state remains the same.")

        self.current_mol = best_smi_this_step
        prep_info_next = self._prepare_next_state_from_mol(self.current_mol)
        
        terminated = self.current_step >= self.max_steps or not prep_info_next.get("actions_available", False)
        info = {'new_smi': self.current_mol if self.current_mol != original_smiles_before_step else None, 'deltas': final_deltas, 'reward_agent': final_reward}
        
        if terminated: print(f"--- ENV STEP {self.current_step}: Episode terminated. ---")
        
        return self._get_current_observation(), final_reward, terminated, False, info

    def _get_current_observation(self):
        return {"observation": self.current_state_numeric, "action_mask": self.current_action_mask_bool}

    def action_masks(self) -> np.ndarray:
        return self.current_action_mask_bool
    
    def _load_property_predictor(self):
        try:
            model = model3.model_predictor.ModelPredictor(
                node_feat_size=atom_featurizer.feat_size('hv'),
                edge_feat_size=bond_featurizer.feat_size('he'),
                num_layers=2, num_timesteps=1, graph_feat_size=256,
                predictor_hidden_feats=256, n_tasks=108
            )
            model.load_state_dict(torch.load(Config.SME_MODEL_PATH, map_location=device))
            model.to(device).eval()
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
            _, bg, _, _ = next(iter(dataloader))
            bg = bg.to(device)
            n_feats, e_feats = bg.ndata.pop('hv').to(device), bg.edata.pop('he').to(device)
            self._graph_cache[smi] = (bg, n_feats, e_feats)
            return bg, n_feats, e_feats
        except Exception: return None, None, None
        
    def _predict_property(self, smi: str) -> Dict[str, float]:
        """Predicts properties (Ames/hERG via Sigmoid, logS via Raw Logits) for the given SMILES."""
        if smi in self._property_cache:
            return self._property_cache[smi]

        bg, n_feats, e_feats = self._prepare_graph_for_prediction(smi)
        if bg is None:
            error_results = {name: 0.0 for name in self.label_name}
            self._property_cache[smi] = error_results
            return error_results

        # 12. Map indices based on pre-trained classification mapping
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
                    print(f"  [Prop Predict] WARNING: Property '{prop_name}' mapping missing. Defaulting to 0.0")
                    results[prop_name] = 0.0
                    continue

                raw_logit = all_logits[0, model_idx]

                if prop_name == "logS":
                    final_value = raw_logit.item()
                    print(f"  [Prop Predict] {prop_name:<10} (idx {model_idx}): Logit={raw_logit.item():.4f} -> Final={final_value:.4f} (Reg)")
                else: 
                    final_value = torch.sigmoid(raw_logit).item()
                    print(f"  [Prop Predict] {prop_name:<10} (idx {model_idx}): Logit={raw_logit.item():.4f} -> Final={final_value:.4f} (Class)")
                
                results[prop_name] = final_value

            self._property_cache[smi] = results
            return results

        except Exception as e:
            print(f"  [Prop Predict] CRITICAL ERROR for {smi}: {e}")
            error_results = {name: 0.0 for name in self.label_name}
            self._property_cache[smi] = error_results
            return error_results
      
    def _calculate_chemical_properties(self, smi: str) -> Dict:
        if smi is None or not isinstance(smi, str) or smi.strip() == "":
            print(f"[ERROR] Invalid SMILES input: {smi}")
            return {'valid': False, 'qed': 0, 'sascore': 10, 'logp': 0, 'mw': 0}
        mol = Chem.MolFromSmiles(smi)

        sascore_value = 0.0
        try:
            if 'sascorer' in globals() and hasattr(sascorer, 'calculateScore'):
                sascore_value = sascorer.calculateScore(mol)
        except Exception as e:
            print(f"[WARNING] SAScore calculation failed for {smi}: {e}")

        return {
            'valid': True,
            'qed': QED.qed(mol),
            'sascore': sascore_value,
            'logp': Crippen.MolLogP(mol),
            'mw': Descriptors.MolWt(mol)
        }

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Step: {self.current_step}, SMILES: {self.current_mol}, Actions: {np.sum(self.current_action_mask_bool)}")

    def close(self):
        self._property_cache.clear()
        self._graph_cache.clear()

# 13. Training Callbacks for Logging & Evaluation
class ActionProbsCallback(BaseCallback):
    def __init__(self, print_freq: int = 1000, verbose: int = 0):
        super(ActionProbsCallback, self).__init__(verbose)
        self.print_freq = print_freq
        self.print_next_log = self.print_freq

    def _on_training_start(self) -> None:
        super()._on_training_start()
        if self.model is not None and hasattr(self.model, 'observation_space'):
            self.observation_space = self.model.observation_space
            if self.verbose > 0:
                print(f"ActionProbsCallback: Observation space SET: {self.observation_space}")
        else:
            if self.verbose > 0:
                print("ActionProbsCallback: CRITICAL - self.model.observation_space is None in _on_training_start.")
        
        if self.model is not None and hasattr(self.model, 'action_space'):
            self.action_space = self.model.action_space
            if self.verbose > 0:
                print(f"ActionProbsCallback: Action space SET: {self.action_space}")
        else:
            if self.verbose > 0:
                print("ActionProbsCallback: CRITICAL - self.model.action_space is None in _on_training_start.")

    def _get_obs_dict_from_locals_or_buffer(self) -> Optional[Dict[str, np.ndarray]]:
        current_obs_dict_tensor = None

        if 'obs' in self.locals:
            obs_from_locals = self.locals['obs']
            if isinstance(obs_from_locals, dict) and all(isinstance(v, torch.Tensor) for v in obs_from_locals.values()):
                current_obs_dict_tensor = {k: v[0].cpu() for k, v in obs_from_locals.items()} 
            elif isinstance(obs_from_locals, torch.Tensor):
                if self.verbose > 0: print("ActionProbsCallback: Warning - 'obs' in locals is a Tensor, not Dict.")
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
                        if self.verbose > 0: print(f"ActionProbsCallback: Unexpected type in rollout_buffer for key {key}")
                        return None

        if current_obs_dict_tensor is None:
            if self.verbose > 0: print(f"ActionProbsCallback: Timestep {self.num_timesteps}: Could not retrieve obs_dict.")
            return None

        try:
            obs_dict_numpy = {k: v.numpy() for k, v in current_obs_dict_tensor.items()}
            return obs_dict_numpy
        except Exception as e:
            if self.verbose > 0: print(f"ActionProbsCallback: Error converting obs_dict_tensor to numpy: {e}")
            return None

    def _on_step(self) -> bool:
        if not hasattr(self, 'observation_space') or self.observation_space is None:
             if self.model is not None and hasattr(self.model, 'observation_space'):
                self.observation_space = self.model.observation_space
             else:
                if self.verbose > 0: print("ActionProbsCallback: CRITICAL - Obs space not set in _on_step.")
                return True
        if not hasattr(self, 'action_space') or self.action_space is None:
            if self.model is not None and hasattr(self.model, 'action_space'):
                self.action_space = self.model.action_space
            else:
                if self.verbose > 0: print("ActionProbsCallback: CRITICAL - Action space not set in _on_step.")
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
                            if self.verbose > 0: print("ActionProbsCallback: 'action_mask' not found in obs_dict_np.")
                            num_available_actions = probs_to_log.size 
                        else:
                            if action_mask_np.ndim > 1: 
                                action_mask_np = action_mask_np.squeeze(0)
                            num_available_actions = np.sum(action_mask_np)
                        
                        if self.verbose > 0:
                            log_msg = (f"Timestep: {self.num_timesteps}, "
                                       f"Action Probs (first 5 slots of {probs_to_log.size}): {probs_to_log[:5].round(3)}, "
                                       f"{num_available_actions} actions available via mask.")
                            print(log_msg)

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
                            else:
                                if self.verbose > 0: print("ActionProbsCallback: Action distribution has no 'entropy' method.")
                    else:
                        if self.verbose > 0: print(f"ActionProbsCallback: Timestep {self.num_timesteps}: Could not extract probabilities from action_distribution type: {type(action_distribution)}")

            except Exception as e:
                if self.verbose > 0:
                    print(f"ActionProbsCallback: Error processing probabilities at timestep {self.num_timesteps}: {e}")
                    import traceback
                    traceback.print_exc()
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
        else:
            print(f"    Warning: Tag '{scalar_tag}' not found in {tfevents_filepath}.")
    except Exception as e:
        print(f"    Error reading tfevents file {tfevents_filepath} for tag {scalar_tag}: {e}")
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
    })
    def make_hpo_train_env():
        return Monitor(MoleculeOptimEnv(Config.MODEL_NAMES, Config.TRAIN_SMILES))
    vec_env = DummyVecEnv([make_hpo_train_env])
    model = MaskablePPO(policy='MultiInputPolicy', env=vec_env, **current_hyperparams)
    try:
        model.learn(total_timesteps=Config.OPTUNA_TIMESTEPS_PER_TRIAL, callback=None, tb_log_name="optuna")
        mean_reward_eval = np.random.rand() 
    except Exception:
        mean_reward_eval = -float('inf')
    finally:
        vec_env.close()
    return mean_reward_eval

def train_final_model(hyperparams: Dict):
    print("\n=== Training Final Model with Full Callbacks ===")
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
        eval_smiles = Config.TRAIN_SMILES[:max(20, len(Config.TRAIN_SMILES)//10)]
        env = MoleculeOptimEnv(
            model_names=Config.MODEL_NAMES,
            smiles_list=eval_smiles,
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
        eval_freq=max(1, Config.EVAL_FREQ // hyperparams.get('n_steps', 1024)) * hyperparams.get('n_steps', 1024),
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
    print(f"Starting final model training with {Config.TOTAL_TIMESTEPS} timesteps. TB log name: {actual_ppo_run_name_for_tb}")
    
    try:
        model.learn(
            total_timesteps=Config.TOTAL_TIMESTEPS,
            callback=combined_callbacks,
            tb_log_name=actual_ppo_run_name_for_tb 
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    model.save(os.path.join(Config.MODEL_SAVE_PATH, "final_maskable_ppo_model_completed.zip"))
    print(f"Final MaskablePPO model (at end of training) saved.")
    
    vec_env.close()
    eval_vec_env.close()
    
    return model

# 14. Comprehensive Evaluation Logic
def evaluate_trained_model(model: MaskablePPO, test_smiles: List[str], n_runs_per_smiles: int = 20):
    """
    Evaluates the trained model over complete generation trajectories.
    Tracks all intermediate modifications across specified number of runs.
    """
    print(f"\n=== Evaluating Trained Model - Full Trajectory (n_runs={n_runs_per_smiles}) ===")
    
    all_generated_molecules_data = []
    max_steps_per_episode = Config.ENV_PARAMS['max_steps']

    for smi_idx, original_smi in enumerate(test_smiles):
        print(f"\n  Evaluating original SMILES {smi_idx + 1}/{len(test_smiles)}: {original_smi}")
        
        def make_eval_env():
            env = MoleculeOptimEnv(
                model_names=Config.MODEL_NAMES,
                smiles_list=[original_smi],
                max_steps=max_steps_per_episode,
                is_eval_env=True
            )
            return env
        
        eval_env_vec = DummyVecEnv([make_eval_env])
        underlying_env = eval_env_vec.envs[0]
        
        original_props = underlying_env._predict_property(original_smi)
        original_chem_props = underlying_env._calculate_chemical_properties(original_smi)

        for run_idx in range(n_runs_per_smiles):
            obs_dict = eval_env_vec.reset()
            if not underlying_env.last_reset_info.get("actions_available", False):
                print(f"    Run {run_idx + 1}: Initialization failed, skipping.")
                continue
            
            done = False
            step_count = 0
            
            while not done and step_count < max_steps_per_episode:
                step_count += 1
                smi_before_step = underlying_env.current_mol

                action, _ = model.predict(obs_dict, action_masks=obs_dict["action_mask"], deterministic=False)
                obs_dict, _, dones, infos = eval_env_vec.step(action)
                done = dones[0]
                
                info_this_step = infos[0]
                generated_smi_this_step = info_this_step.get('new_smi', smi_before_step)

                is_novel_vs_original = (generated_smi_this_step != original_smi)
                
                reward_vs_original, deltas_vs_original, _, new_props = underlying_env._calculate_reward(original_smi, generated_smi_this_step)
                chem_props_new = underlying_env._calculate_chemical_properties(generated_smi_this_step)

                step_data = {
                    'original_smiles': original_smi,
                    'run_index': run_idx + 1,
                    'step_index': step_count,
                    'smiles_before_step': smi_before_step,
                    'generated_smiles': generated_smi_this_step,
                    'is_novel_vs_original': is_novel_vs_original,
                    'recalculated_reward_vs_original': reward_vs_original,
                    **{f'original_prop_{underlying_env.model_to_label_map[name]}': original_props.get(underlying_env.model_to_label_map.get(name), 0) for name in Config.MODEL_NAMES},
                    **{f'new_prop_{underlying_env.model_to_label_map[name]}': new_props.get(underlying_env.model_to_label_map.get(name), 0) for name in Config.MODEL_NAMES},
                    **{f'delta_vs_original_{name}': deltas_vs_original.get(name, 0.0) for name in Config.MODEL_NAMES},
                    **{f'original_chem_{k}': v for k, v in original_chem_props.items()},
                    **{f'new_chem_{k}': v for k,v in chem_props_new.items()}
                }
                all_generated_molecules_data.append(step_data)

        eval_env_vec.close()

    if not all_generated_molecules_data:
        print("\nNo evaluations generated, cannot compute statistics.")
        return

    df_results = pd.DataFrame(all_generated_molecules_data)
    detailed_results_path = os.path.join(Config.RESULT_PATH, "evaluation_detailed_full_trajectory.csv")
    df_results.to_csv(detailed_results_path, index=False)
    print(f"\nDetailed evaluation results saved to: {detailed_results_path}")

    stats = {}
    df_novel = df_results[df_results['is_novel_vs_original']].copy()
    num_total_generations = len(df_results)
    num_novel_generations = len(df_novel)
    
    stats['Total_Generation_Steps'] = num_total_generations
    stats['Num_Novel_Molecules_Generated'] = num_novel_generations
    stats['Novelty_Rate_All_Generations'] = num_novel_generations / num_total_generations if num_total_generations > 0 else 0.0

    for model_name in Config.MODEL_NAMES:
        delta_col = f'delta_vs_original_{model_name}'
        df_novel[f'is_improved_{model_name}'] = df_novel[delta_col] > 1e-7
        stats[f'ImprovementRate_{model_name}_in_Novel'] = df_novel[f'is_improved_{model_name}'].mean() if num_novel_generations > 0 else 0.0

    if num_novel_generations > 0:
        improvement_conditions = [df_novel[f'is_improved_{name}'] for name in Config.MODEL_NAMES]
        df_novel['all_objectives_improved'] = np.logical_and.reduce(improvement_conditions)
        stats['Rate_AllObjectivesImproved_in_Novel'] = df_novel['all_objectives_improved'].mean()
    else:
        stats['Rate_AllObjectivesImproved_in_Novel'] = 0.0
    
    model_to_label_map = {"Mutagenicity": "Ames", "ESOL": "logS", "hERG": "hERG_10uM", "BBBP": "BBB"}

    hard_conditions = []
    if 'Mutagenicity' in Config.MODEL_NAMES:
        hard_conditions.append(df_results[f'original_prop_{model_to_label_map["Mutagenicity"]}'] >= 0.5)
    if 'hERG' in Config.MODEL_NAMES:
        hard_conditions.append(df_results[f'original_prop_{model_to_label_map["hERG"]}'] >= 0.5)
    if 'BBBP' in Config.MODEL_NAMES:
        hard_conditions.append(df_results[f'original_prop_{model_to_label_map["BBBP"]}'] < 0.5)

    if hard_conditions:
        is_from_hard_molecule = np.logical_and.reduce(hard_conditions)
        df_from_hard_mols = df_results[is_from_hard_molecule].copy()
        
        hard_smiles_list = df_from_hard_mols['original_smiles'].unique()
        print(f"\nFound {len(hard_smiles_list)} 'hard' original molecules in the test set.")
        
        if len(hard_smiles_list) > 0:
            success_conditions = []
            if 'Mutagenicity' in Config.MODEL_NAMES:
                success_conditions.append(df_from_hard_mols[f'new_prop_{model_to_label_map["Mutagenicity"]}'] < 0.5)
            if 'hERG' in Config.MODEL_NAMES:
                success_conditions.append(df_from_hard_mols[f'new_prop_{model_to_label_map["hERG"]}'] < 0.5)
            if 'ESOL' in Config.MODEL_NAMES:
                success_conditions.append(
                    df_from_hard_mols[f'new_prop_{model_to_label_map["ESOL"]}'] > (df_from_hard_mols[f'original_prop_{model_to_label_map["ESOL"]}'] + 0.5)
                )
            if 'BBBP' in Config.MODEL_NAMES:
                success_conditions.append(df_from_hard_mols[f'new_prop_{model_to_label_map["BBBP"]}'] >= 0.5)
            
            df_from_hard_mols['is_fully_successful'] = np.logical_and.reduce(success_conditions)
            success_rates_per_hard_smi = df_from_hard_mols.groupby('original_smiles')['is_fully_successful'].mean()
            stats['Mean_Success_Rate_for_Hard_Molecules'] = success_rates_per_hard_smi.mean()
        else:
            stats['Mean_Success_Rate_for_Hard_Molecules'] = 0.0
    else:
        stats['Mean_Success_Rate_for_Hard_Molecules'] = 0.0
        print("\nNo 'hard' original molecules found in the test set.")

    if not df_novel.empty:
        df_best_per_smi = df_novel.sort_values('recalculated_reward_vs_original', ascending=False).drop_duplicates('original_smiles')
        df_best_per_smi.rename(columns={'new_chem_valid': 'valid', 'new_chem_qed': 'qed', 'new_chem_sascore': 'sascore'}, inplace=True)

        stats['Avg_Validity_of_Best_SMILES'] = df_best_per_smi['valid'].mean()
        
        df_best_valid = df_best_per_smi[df_best_per_smi['valid']]
        if not df_best_valid.empty:
            stats['Avg_QED_of_Best_Valid_SMILES'] = df_best_valid['qed'].mean()
            stats['Avg_SAscore_of_Best_Valid_SMILES'] = df_best_valid['sascore'].mean()
        else:
            stats.update({'Avg_QED_of_Best_Valid_SMILES': 0, 'Avg_SAscore_of_Best_Valid_SMILES': 10.0})
        
        stats['Uniqueness_of_Best_SMILES'] = df_best_per_smi['generated_smiles'].nunique() / len(df_best_per_smi) if len(df_best_per_smi) > 0 else 0.0
        train_set = set(Config.TRAIN_SMILES)
        novel_against_train_count = sum(1 for s in df_best_per_smi['generated_smiles'] if s not in train_set)
        stats['Novelty_of_Best_SMILES_vs_Train'] = novel_against_train_count / len(df_best_per_smi) if len(df_best_per_smi) > 0 else 0.0
    else:
        stats.update({
            'Avg_Validity_of_Best_SMILES': 0, 'Avg_QED_of_Best_Valid_SMILES': 0, 'Avg_SAscore_of_Best_Valid_SMILES': 10.0,
            'Uniqueness_of_Best_SMILES': 0, 'Novelty_of_Best_SMILES_vs_Train': 0
        })

    summary_path = os.path.join(Config.RESULT_PATH, "evaluation_summary_full_trajectory.csv")
    pd.DataFrame([stats], index=[0]).to_csv(summary_path, index=False)
    print(f"\nEvaluation summary saved to: {summary_path}")
    
    print("\n--- Final Trajectory Evaluation Summary ---")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, (float, np.floating)) else f"  {key}: {value}")


# 15. Main Execution Block
if __name__ == "__main__":
    Config.setup_directories() 

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
    
    print("\nTraining final model with MaskablePPO...")
    final_model = train_final_model(best_hpo_params)
    
    evaluate_trained_model(final_model, Config.TEST_SMILES, n_runs_per_smiles=20) 

    print("\nExperiment finished.")