from dgl import DGLGraph
import pandas as pd
from rdkit.Chem import MolFromSmiles
import numpy as np
from dgl.data.graph_serialize import save_graphs, load_graphs, load_labels
import torch as th
from rdkit.Chem import BRICS
import random
import os
from rdkit import Chem
from rdkit.Chem import RDConfig
from rdkit.Chem import FragmentCatalog
from rdkit.Chem.Scaffolds import MurckoScaffold
import itertools
import ast


# read FunctionGroup information
# with open(fName) as f:
#     file = f.read()

def return_fg_without_c_i_wash(fg_with_c_i, fg_without_c_i):
    # the fragment genereated from smarts would have a redundant carbon, here to remove the redundant carbon
    fg_without_c_i_wash = []
    for fg_with_c in fg_with_c_i:
        for fg_without_c in fg_without_c_i:
            if set(fg_without_c).issubset(set(fg_with_c)):
                fg_without_c_i_wash.append(list(fg_without_c))
    return fg_without_c_i_wash


def return_fg_hit_atom(smiles, fg_name_list, fg_with_ca_list, fg_without_ca_list):
    mol = Chem.MolFromSmiles(smiles)
    hit_at = []
    hit_fg_name = []
    all_hit_fg_at = []
    for i in range(len(fg_with_ca_list)):
        fg_with_c_i = mol.GetSubstructMatches(fg_with_ca_list[i])
        fg_without_c_i = mol.GetSubstructMatches(fg_without_ca_list[i])
        fg_without_c_i_wash = return_fg_without_c_i_wash(fg_with_c_i, fg_without_c_i)
        if len(fg_without_c_i_wash) > 0:
            hit_at.append(fg_without_c_i_wash)
            hit_fg_name.append(fg_name_list[i])
            all_hit_fg_at += fg_without_c_i_wash
    # sort function group atom by atom number
    sorted_all_hit_fg_at = sorted(all_hit_fg_at,
                                  key=lambda fg: len(fg),
                                  reverse=True)
    # remove small function group (wrongly matched), they are part of other big function groups
    remain_fg_list = []
    for fg in sorted_all_hit_fg_at:
        if fg not in remain_fg_list:
            if len(remain_fg_list) == 0:
                remain_fg_list.append(fg)
            else:
                i = 0
                for remain_fg in remain_fg_list:
                    if set(fg).issubset(set(remain_fg)):
                        break
                    else:
                        i += 1
                if i == len(remain_fg_list):
                    remain_fg_list.append(fg)
    # wash the hit function group atom by using the remained fg, remove the small wrongly matched fg
    hit_at_wash = []
    hit_fg_name_wash = []
    for j in range(len(hit_at)):
        hit_at_wash_j = []
        for fg in hit_at[j]:
            if fg in remain_fg_list:
                hit_at_wash_j.append(fg)
        if len(hit_at_wash_j) > 0:
            hit_at_wash.append(hit_at_wash_j)
            hit_fg_name_wash.append(hit_fg_name[j])
    return hit_at_wash, hit_fg_name_wash


def getAllBricsBondSubset(BricsBond):
    all_brics_bond_subset = []
    N = len(BricsBond)
    for i in range(2 ** N):
        brics_bond_subset = []
        for j in range(N):
            if (i >> j) % 2:
                brics_bond_subset.append(BricsBond[j])
        if len(brics_bond_subset) > 0:
            all_brics_bond_subset.append(brics_bond_subset)
        if len(all_brics_bond_subset) > 10000:
            break
    return all_brics_bond_subset


def return_brics_structure_all_substructure(smiles):
    m = Chem.MolFromSmiles(smiles)
    res = list(BRICS.FindBRICSBonds(m))  # [((1, 2), ('1', '5'))]

    # return brics_bond
    all_brics_bond = [set(res[i][0]) for i in range(len(res))]
    all_brics_bond_subset = getAllBricsBondSubset(all_brics_bond)

    all_brics_substructure_subset = dict()
    for i, brics_bond_subset in enumerate(all_brics_bond_subset):
        # return atom in brics_bond_subset
        all_brics_atom = []
        for brics_bond in brics_bond_subset:
            all_brics_atom = list(set(all_brics_atom + list(brics_bond)))

        # return all break atom (the break atoms did'n appear in the same substructure)
        all_break_atom = dict()
        for brics_atom in all_brics_atom:
            brics_break_atom = []
            for brics_bond in brics_bond_subset:
                if brics_atom in brics_bond:
                    brics_break_atom += list(set(brics_bond))
            brics_break_atom = [x for x in brics_break_atom if x != brics_atom]
            all_break_atom[brics_atom] = brics_break_atom

        substrate_idx = dict()
        used_atom = []
        for initial_atom_idx, break_atoms_idx in all_break_atom.items():
            if initial_atom_idx not in used_atom:
                neighbor_idx = [initial_atom_idx]
                substrate_idx_i = neighbor_idx
                begin_atom_idx_list = [initial_atom_idx]
                while len(neighbor_idx) != 0:
                    for idx in begin_atom_idx_list:
                        initial_atom = m.GetAtomWithIdx(idx)
                        neighbor_idx = neighbor_idx + [neighbor_atom.GetIdx() for neighbor_atom in
                                                       initial_atom.GetNeighbors()]
                        exlude_idx = all_break_atom[initial_atom_idx] + substrate_idx_i
                        if idx in all_break_atom.keys():
                            exlude_idx = all_break_atom[initial_atom_idx] + substrate_idx_i + all_break_atom[idx]
                        neighbor_idx = [x for x in neighbor_idx if x not in exlude_idx]
                        substrate_idx_i += neighbor_idx
                        begin_atom_idx_list += neighbor_idx
                    begin_atom_idx_list = [x for x in begin_atom_idx_list if x not in substrate_idx_i]
                substrate_idx[initial_atom_idx] = substrate_idx_i
                used_atom += substrate_idx_i
            else:
                pass
        all_brics_substructure_subset['substructure_subset_{}'.format(i + 1)] = substrate_idx
        all_brics_substructure_subset['substructure_subset_{}_bond'.format(i + 1)] = brics_bond_subset
    substrate_idx = dict()
    substrate_idx[0] = [x for x in range(m.GetNumAtoms())]
    all_brics_substructure_subset['substructure_subset_{}'.format(0)] = substrate_idx
    all_brics_substructure_subset['substructure_subset_{}_bond'.format(0)] = []
    return all_brics_substructure_subset


def return_brics_leaf_structure(smiles):
    m = Chem.MolFromSmiles(smiles)
    res = list(BRICS.FindBRICSBonds(m))  # [((1, 2), ('1', '5'))]

    # return brics_bond
    all_brics_bond = [set(res[i][0]) for i in range(len(res))]

    all_brics_substructure_subset = dict()
    # return atom in all_brics_bond
    all_brics_atom = []
    for brics_bond in all_brics_bond:
        all_brics_atom = list(set(all_brics_atom + list(brics_bond)))

    if len(all_brics_atom) > 0:
        # return all break atom (the break atoms did'n appear in the same substructure)
        all_break_atom = dict()
        for brics_atom in all_brics_atom:
            brics_break_atom = []
            for brics_bond in all_brics_bond:
                if brics_atom in brics_bond:
                    brics_break_atom += list(set(brics_bond))
            brics_break_atom = [x for x in brics_break_atom if x != brics_atom]
            all_break_atom[brics_atom] = brics_break_atom

        substrate_idx = dict()
        used_atom = []
        for initial_atom_idx, break_atoms_idx in all_break_atom.items():
            if initial_atom_idx not in used_atom:
                neighbor_idx = [initial_atom_idx]
                substrate_idx_i = neighbor_idx
                begin_atom_idx_list = [initial_atom_idx]
                while len(neighbor_idx) != 0:
                    for idx in begin_atom_idx_list:
                        initial_atom = m.GetAtomWithIdx(idx)
                        neighbor_idx = neighbor_idx + [neighbor_atom.GetIdx() for neighbor_atom in
                                                       initial_atom.GetNeighbors()]
                        exlude_idx = all_break_atom[initial_atom_idx] + substrate_idx_i
                        if idx in all_break_atom.keys():
                            exlude_idx = all_break_atom[initial_atom_idx] + substrate_idx_i + all_break_atom[idx]
                        neighbor_idx = [x for x in neighbor_idx if x not in exlude_idx]
                        substrate_idx_i += neighbor_idx
                        begin_atom_idx_list += neighbor_idx
                    begin_atom_idx_list = [x for x in begin_atom_idx_list if x not in substrate_idx_i]
                substrate_idx[initial_atom_idx] = substrate_idx_i
                used_atom += substrate_idx_i
            else:
                pass
    else:
        substrate_idx = dict()
        substrate_idx[0] = [x for x in range(m.GetNumAtoms())]
    all_brics_substructure_subset['substructure'] = substrate_idx
    all_brics_substructure_subset['substructure_bond'] = all_brics_bond
    return all_brics_substructure_subset


def reindex_substructure(substructure_dir):
    # get all atoms in substructure-substructure(ss) bond
    all_atom_list = []
    for bond in substructure_dir['substructure_bond']:
        all_atom_list += list(bond)
    all_atom_list = list(set(all_atom_list))

    # reindex substructure
    substructure_reindex = dict()
    sub_structure = substructure_dir['substructure']
    for i, sub in enumerate(sub_structure.items()):
        substructure_reindex[i] = sub[1]
    substructure_dir['substructure_reindex'] = substructure_reindex

    # get substructure-bond atom'reindex substructure dir
    new_sub_bond_dir = dict()
    for atom in all_atom_list:
        for j, reindex_sub in substructure_reindex.items():
            if atom in reindex_sub:
                new_sub_bond_dir[atom] = j
                break

    # change the substructure_bond to reindex ss-bond
    ss_bond = []
    for bond in substructure_dir['substructure_bond']:
        list_bond = list(bond)
        ss_bond.append([new_sub_bond_dir[list_bond[0]], new_sub_bond_dir[list_bond[1]]])
    substructure_dir['ss_bond'] = ss_bond
    return substructure_dir


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom, use_chirality=True):
    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        [
            'B',
            'C',
            'N',
            'O',
            'F',
            'Si',
            'P',
            'S',
            'Cl',
            'As',
            'Se',
            'Br',
            'Te',
            'I',
            'At',
            'other'
        ]) + one_of_k_encoding(atom.GetDegree(),
                               [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                  Chem.rdchem.HybridizationType.SP3D2, 'other']) + [atom.GetIsAromatic()]
    results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                                 ] + [atom.HasProp('_ChiralityPossible')]

    return np.array(results)


def one_of_k_atompair_encoding(x, allowable_set):
    for atompair in allowable_set:
        if x in atompair:
            x = atompair
            break
        else:
            if atompair == allowable_set[-1]:
                x = allowable_set[-1]
            else:
                continue
    return [x == s for s in allowable_set]


def bond_features(bond, use_chirality=True):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])

    return np.array(bond_feats).astype(float)


def etype_features(bond, use_chirality=True):
    bt = bond.GetBondType()
    bond_feats_1 = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
    ]
    for i, m in enumerate(bond_feats_1):
        if m == True:
            a = i

    bond_feats_2 = bond.GetIsConjugated()
    if bond_feats_2 == True:
        b = 1
    else:
        b = 0

    bond_feats_3 = bond.IsInRing
    if bond_feats_3 == True:
        c = 1
    else:
        c = 0

    index = a * 1 + b * 4 + c * 8
    if use_chirality:
        bond_feats_4 = one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
        for i, m in enumerate(bond_feats_4):
            if m == True:
                d = i
        index = index + d * 16
    return index


def substructure_features(m, substructure):
    features_list = []
    for i in substructure:
        features_list.append(atom_features(m.GetAtomWithIdx(i)))
    features_np = np.array(features_list)
    features = np.sum(features_np, axis=0)
    return features


# 化学键 分为单键 双键 三键 方向键，转化为整数
BT_MAPPING_INT = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 1.5,
}


def generate_substructure_features(mol, substruct):
    """
    Generates a vector with mapping for a substructure
    :param substruct: The given substructure
    :param mol: RDKit molecule
    :param structure_type: The type of a structure (one of STRUCT_TO_NUM)
    :return: An encoding vector
    """
    atoms = [mol.GetAtomWithIdx(i) for i in substruct]
    substruct_atomic_encoding = np.array([one_of_k_encoding_unk(atom.GetSymbol(),
                                                                [
                                                                    'B',
                                                                    'C',
                                                                    'N',
                                                                    'O',
                                                                    'F',
                                                                    'Si',
                                                                    'P',
                                                                    'S',
                                                                    'Cl',
                                                                    'As',
                                                                    'Se',
                                                                    'Br',
                                                                    'Te',
                                                                    'I',
                                                                    'At',
                                                                    'other'
                                                                ]) for atom in atoms])
    substruct_atomic_sum = np.sum(substruct_atomic_encoding, axis=0).tolist()
    substruct_atomic_sum_norm = [0.1 * atomic_sum for atomic_sum in substruct_atomic_sum]

    # implicit_substruct_valence 计算子结构共价键的数量，单键为1，双键为2，三键为3，方向键为1.5，如：苯环即为9，注意，这里计算的是子结构内部的化合价
    implicit_substruct_valence = 0
    for i in range(len(atoms)):
        for j in range(i, len(atoms)):
            bond = mol.GetBondBetweenAtoms(substruct[i], substruct[j])
            if bond:
                implicit_substruct_valence += BT_MAPPING_INT[
                    mol.GetBondBetweenAtoms(substruct[i], substruct[j]).GetBondType()]

    # 子结构的形式电荷总和
    substruct_formal_charge = sum(atom.GetFormalCharge() for atom in atoms)

    # 子结构的H总和
    substruct_num_Hs = sum(atom.GetTotalNumHs() for atom in atoms)

    # substruct_valence 子结构化合价，这里指的是子结构与其他子结构连接时表现出来的化合价
    substruct_valence = sum(atom.GetExplicitValence() for atom in atoms) - 2 * implicit_substruct_valence

    # 子结构包不包含芳香性
    substruct_is_aromatic = 1 if sum(atom.GetIsAromatic() for atom in atoms) > 0 else 0

    # 子结构原子质量之和
    substruct_mass = sum(atom.GetMass() for atom in atoms)

    # 子结构内部的键之和
    substruct_edges_sum = implicit_substruct_valence

    # 特征加和
    features = substruct_atomic_sum_norm + [substruct_num_Hs * 0.1, substruct_valence, substruct_formal_charge,
                                            substruct_is_aromatic, substruct_mass * 0.01, substruct_edges_sum * 0.1]
    return features


def construct_RGCN_mol_graph_from_smiles(smiles, smask):
    g = DGLGraph()

    # Add nodes
    mol = MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)
    atoms_feature_all = []
    smask_list = []
    for i, atom in enumerate(mol.GetAtoms()):
        atom_feature = atom_features(atom)
        atoms_feature_all.append(atom_feature)
        if i in smask:
            smask_list.append(0)
        else:
            smask_list.append(1)
    g.ndata["node"] = th.tensor(atoms_feature_all)
    g.ndata["smask"] = th.tensor(smask_list).float()

    # Add edges
    src_list = []
    dst_list = []
    etype_feature_all = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        etype_feature = etype_features(bond)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src_list.extend([u, v])
        dst_list.extend([v, u])
        etype_feature_all.append(etype_feature)
        etype_feature_all.append(etype_feature)

    g.add_edges(src_list, dst_list)
    g.edata["edge"] = th.tensor(etype_feature_all)
    return g


def build_mol_graph_data(dataset_smiles, labels_name, smiles_name):
    dataset_gnn = []
    failed_molecule = []
    labels = dataset_smiles[labels_name]
    split_index = dataset_smiles['group']
    smilesList = dataset_smiles[smiles_name]
    molecule_number = len(smilesList)

    # 打印数据集前5行，检查结构
    print("First 5 rows of the dataset:")
    print(dataset_smiles.head())
    
    for i, smiles in enumerate(smilesList):
        try:
            # 打印每个 SMILES 和索引
            print(f"Processing molecule {i + 1}/{molecule_number}: {smiles}")
            
            g_rgcn = construct_RGCN_mol_graph_from_smiles(smiles, smask=[])
            
            # 打印构建的图结构
            print(f"Graph for molecule {i + 1}: {g_rgcn}")
            
            molecule = [smiles, g_rgcn, labels.loc[i], split_index.loc[i]]
            
            # 打印每个 molecule 的内容
            print(f"molecule {i + 1} content: {molecule}")
            
            dataset_gnn.append(molecule)
            print(f'{i + 1}/{molecule_number} molecule is transformed to mol graph! {len(failed_molecule)} molecules failed so far.')
        
        except Exception as e:
            # 打印错误的 SMILES 分子
            print(f"Error with molecule {i + 1}: {smiles} - {e}")
            failed_molecule.append(smiles)
            molecule_number -= 1

    # 打印所有失败的分子和失败数量
    print(f'{len(failed_molecule)} molecules failed to transform: {failed_molecule}')
    
    return dataset_gnn


def find_murcko_link_bond(mol):
    core = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_index = mol.GetSubstructMatch(core)
    link_bond_list = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        link_score = 0
        if u in scaffold_index:
            link_score += 1
        if v in scaffold_index:
            link_score += 1
        if link_score == 1:
            link_bond_list.append([u, v])
    return link_bond_list


def return_murcko_leaf_structure(smiles):
    m = Chem.MolFromSmiles(smiles)

    # return murcko_link_bond
    all_murcko_bond = find_murcko_link_bond(m)

    all_murcko_substructure_subset = dict()
    # return atom in all_murcko_bond
    all_murcko_atom = []
    for murcko_bond in all_murcko_bond:
        all_murcko_atom = list(set(all_murcko_atom + murcko_bond))

    if len(all_murcko_atom) > 0:
        # return all break atom (the break atoms did'n appear in the same substructure)
        all_break_atom = dict()
        for murcko_atom in all_murcko_atom:
            murcko_break_atom = []
            for murcko_bond in all_murcko_bond:
                if murcko_atom in murcko_bond:
                    murcko_break_atom += list(set(murcko_bond))
            murcko_break_atom = [x for x in murcko_break_atom if x != murcko_atom]
            all_break_atom[murcko_atom] = murcko_break_atom

        substrate_idx = dict()
        used_atom = []
        for initial_atom_idx, break_atoms_idx in all_break_atom.items():
            if initial_atom_idx not in used_atom:
                neighbor_idx = [initial_atom_idx]
                substrate_idx_i = neighbor_idx
                begin_atom_idx_list = [initial_atom_idx]
                while len(neighbor_idx) != 0:
                    for idx in begin_atom_idx_list:
                        initial_atom = m.GetAtomWithIdx(idx)
                        neighbor_idx = neighbor_idx + [neighbor_atom.GetIdx() for neighbor_atom in
                                                       initial_atom.GetNeighbors()]
                        exlude_idx = all_break_atom[initial_atom_idx] + substrate_idx_i
                        if idx in all_break_atom.keys():
                            exlude_idx = all_break_atom[initial_atom_idx] + substrate_idx_i + all_break_atom[idx]
                        neighbor_idx = [x for x in neighbor_idx if x not in exlude_idx]
                        substrate_idx_i += neighbor_idx
                        begin_atom_idx_list += neighbor_idx
                    begin_atom_idx_list = [x for x in begin_atom_idx_list if x not in substrate_idx_i]
                substrate_idx[initial_atom_idx] = substrate_idx_i
                used_atom += substrate_idx_i
            else:
                pass
    else:
        substrate_idx = dict()
        substrate_idx[0] = [x for x in range(m.GetNumAtoms())]
    all_murcko_substructure_subset['substructure'] = substrate_idx
    all_murcko_substructure_subset['substructure_bond'] = all_murcko_bond
    return all_murcko_substructure_subset


def build_mol_graph_data_for_brics_smarts(dataset_smiles, labels_name, smiles_name, substructure_data):
    dataset_gnn = []
    failed_molecule = []
    labels = dataset_smiles[labels_name]
    smilesList = dataset_smiles[smiles_name]
    split_index = dataset_smiles['group']
    molecule_number = len(smilesList)

    for i, smiles in enumerate(smilesList):
        # 从CSV文件中获取当前分子的子结构
        # substructure_indices_str = substructure_data.loc[substructure_data[smiles_name] == smiles, 'substructure_indices']
        current_index = i
        substructure_indices_str = substructure_data.loc[current_index, 'substructure_indices']
        # 打印正在处理的 substructure_indices_str
        print(f"正在处理分子 {smiles} 的 substructure_indices_str: {substructure_indices_str}")
        
        try:
            # 将字符串转换为字典
            substructure_indices = eval(substructure_indices_str)  
            
            # 打印转换后的 substructure_indices
            print(f"转换后的 substructure_indices: {substructure_indices}")
        except Exception as e:
            print(f"Error parsing substructure indices for {smiles}: {e}")
            continue


        atom_mask = []
        brics_substructure_mask = []
        
        mol = Chem.MolFromSmiles(smiles)  # 假设使用 RDKit 来处理 SMILES
        atom_mask = [atom.GetIdx() for atom in mol.GetAtoms()]  # 获取当前分子所有原子的索引

        # 遍历substructure_indices中的每个substructure_list
        for substructure_key, substructure_list in substructure_indices.items():
            print(f"Processing substructure: {substructure_key}, List: {substructure_list}")

            # 遍历每个子结构的索引列表
            for substructure in substructure_list:  
                print(f"Adding substructure {substructure} to the mask")
                brics_substructure_mask.append(substructure)  # 将当前子结构的索引列表添加到 brics_substructure_mask
        
                smask = substructure  # 将 substructure 存为 smask
                print(f"Substructure mask for {smiles}: {smask}")  # 调试输出

                # 生成分子图
                try:
                    g_rgcn = construct_RGCN_mol_graph_from_smiles(smiles, smask=smask)
                    molecule = [smiles, g_rgcn, labels.loc[i], split_index.loc[i], smask, substructure_key, substructure]
                    dataset_gnn.append(molecule)

                    print(f'{smiles} is transformed to mol graph successfully!')
                except Exception as e:
                    print(f'Error processing molecule {smiles}: {e}')
                    failed_molecule.append(smiles)

    return dataset_gnn



def build_mol_graph_data_for_murcko(dataset_smiles, labels_name, smiles_name):
    dataset_gnn = []
    failed_molecule = []
    labels = dataset_smiles[labels_name]
    split_index = dataset_smiles['group']
    smilesList = dataset_smiles[smiles_name]
    molecule_number = len(smilesList)
    for i, smiles in enumerate(smilesList):
        substructure_dir = return_murcko_leaf_structure(smiles)
        atom_mask = []
        murcko_substructure_mask = []
        for _, substructure in substructure_dir['substructure'].items():
            murcko_substructure_mask.append(substructure)
            atom_mask = atom_mask + substructure
        smask = murcko_substructure_mask
        for j, smask_i in enumerate(smask):
            try:
                g_rgcn = construct_RGCN_mol_graph_from_smiles(smiles, smask=smask_i)
                molecule = [smiles, g_rgcn, labels.loc[i], split_index.loc[i], smask_i]
                dataset_gnn.append(molecule)
                print('{}/{}, {}/{} molecule is transformed to mol graph! {} is transformed failed!'.format(j + 1,
                                                                                                            len(smask),
                                                                                                            i + 1,
                                                                                                            molecule_number,
                                                                                                            len(failed_molecule)))
            except:
                print('{} is transformed to mol graph failed!'.format(smiles))
                molecule_number = molecule_number - 1
                failed_molecule.append(smiles)
    print('{}({}) is transformed to mol graph failed!'.format(failed_molecule, len(failed_molecule)))
    return dataset_gnn


def build_mol_graph_data_for_fg(dataset_smiles, labels_name, smiles_name, task_name):
    # 39 function group config
    fName = os.path.join(RDConfig.RDDataDir, 'FunctionalGroups.txt')
    fparams = FragmentCatalog.FragCatParams(1, 6, fName)
    fg_without_ca_smart = ['[N;D2]-[C;D3](=O)-[C;D1;H3]', 'C(=O)[O;D1]', 'C(=O)[O;D2]-[C;D1;H3]',
                           'C(=O)-[H]', 'C(=O)-[N;D1]', 'C(=O)-[C;D1;H3]', '[N;D2]=[C;D2]=[O;D1]',
                           '[N;D2]=[C;D2]=[S;D1]', '[N;D3](=[O;D1])[O;D1]', '[N;R0]=[O;D1]', '[N;R0]-[O;D1]',
                           '[N;R0]-[C;D1;H3]', '[N;R0]=[C;D1;H2]', '[N;D2]=[N;D2]-[C;D1;H3]', '[N;D2]=[N;D1]',
                           '[N;D2]#[N;D1]', '[C;D2]#[N;D1]', '[S;D4](=[O;D1])(=[O;D1])-[N;D1]',
                           '[N;D2]-[S;D4](=[O;D1])(=[O;D1])-[C;D1;H3]', '[S;D4](=O)(=O)-[O;D1]',
                           '[S;D4](=O)(=O)-[O;D2]-[C;D1;H3]', '[S;D4](=O)(=O)-[C;D1;H3]', '[S;D4](=O)(=O)-[Cl]',
                           '[S;D3](=O)-[C;D1]', '[S;D2]-[C;D1;H3]', '[S;D1]', '[S;D1]', '[#9,#17,#35,#53]',
                           '[C;D4]([C;D1])([C;D1])-[C;D1]',
                           '[C;D4](F)(F)F', '[C;D2]#[C;D1;H]', '[C;D3]1-[C;D2]-[C;D2]1', '[O;D2]-[C;D2]-[C;D1;H3]',
                           '[O;D2]-[C;D1;H3]', '[O;D1]', '[O;D1]', '[N;D1]', '[N;D1]', '[N;D1]']
    fg_without_ca_list = [Chem.MolFromSmarts(smarts) for smarts in fg_without_ca_smart]
    fg_with_ca_list = [fparams.GetFuncGroup(i) for i in range(39)]
    fg_name_list = [fg.GetProp('_Name') for fg in fg_with_ca_list]

    # build mol graph for function group
    dataset_gnn = []
    failed_molecule = []
    labels = dataset_smiles[labels_name]
    split_index = dataset_smiles['group']
    smilesList = dataset_smiles[smiles_name]
    molecule_number = len(smilesList)
    for i, smiles in enumerate(smilesList):
        fg_mask = []
        fg_name = []
        hit_fg_at, hit_fg_name = return_fg_hit_atom(smiles, fg_name_list, fg_with_ca_list, fg_without_ca_list)
        for a, hit_fg in enumerate(hit_fg_at):
            for b, hit_fg_b in enumerate(hit_fg):
                fg_mask.append(hit_fg_b)
                fg_name.append(hit_fg_name[a])
        for j, fg_mask_j in enumerate(fg_mask):
            try:
                g_rgcn = construct_RGCN_mol_graph_from_smiles(smiles, smask=fg_mask_j)
                molecule = [smiles, g_rgcn, labels.loc[i], split_index.loc[i], fg_mask_j, fg_name[j]]
                dataset_gnn.append(molecule)
                print('{}/{}, {}/{} molecule is transformed to mol graph! {} is transformed failed!'.format(j + 1,
                                                                                                            len(fg_mask),
                                                                                                            i + 1,
                                                                                                            molecule_number,
                                                                                                            len(failed_molecule)))
            except:
                print('{} is transformed to mol graph failed!'.format(smiles))
                molecule_number = molecule_number - 1
                failed_molecule.append(smiles)
    print('{}({}) is transformed to mol graph failed!'.format(failed_molecule, len(failed_molecule)))
    return dataset_gnn


def cal_pair_index(max_index_len):
    all_pair_combination = []
    for i in range(int(max_index_len / 2) + 1):
        if i * 2 != max_index_len:
            all_sub_index = [x for x in range(max_index_len)]
            combination = [sorted(x) for x in itertools.combinations(all_sub_index, i)]
            pair_index_combination = [sorted([sub_index, list(set(all_sub_index).difference(set(sub_index)))]) for
                                      sub_index in combination]
            all_pair_combination = all_pair_combination + pair_index_combination
        else:
            all_sub_index = [x for x in range(max_index_len)]
            combination = [sorted(x) for x in itertools.combinations(all_sub_index, i)]
            pair_index_combination = [sorted([sub_index, list(set(all_sub_index).difference(set(sub_index)))]) for
                                      sub_index in combination]
            pair_index_combination.sort()
            pair_index_combination_norep = [pair_index_combination[i] for i in range(len(pair_index_combination)) if
                                            i % 2 == 0]
            all_pair_combination = all_pair_combination + pair_index_combination_norep
    return all_pair_combination


def emerge_sub(smask, pair_index):
    # 根据子结构的index来合并子结构, 并根据剩余的子结构index合并另一子结构。即将整个分子一分为二。
    emerge_sub_index_1 = []
    emerge_sub_index_2 = []
    for index in pair_index[0]:
        emerge_sub_index_1 = emerge_sub_index_1 + smask[index]
    for index in pair_index[1]:
        emerge_sub_index_2 = emerge_sub_index_2 + smask[index]
    emerge_pair_sub = [emerge_sub_index_1, emerge_sub_index_2]
    return emerge_pair_sub


def build_mol_graph_data_for_brics_combine(dataset_smiles, labels_name, smiles_name):
    dataset_gnn = []
    failed_molecule = []
    labels = dataset_smiles[labels_name]
    smilesList = dataset_smiles[smiles_name]
    molecule_number = len(smilesList)
    split_index = dataset_smiles['group']

    for i, smiles in enumerate(smilesList):
        try:
            # 解析 `brics_comb_indices`
            smask = eval(dataset_smiles.loc[i, 'brics_comb_indices'])
        except Exception as e:
            print(f"❌ Error parsing brics_comb_indices for {smiles}: {e}")
            continue

        print(f"\n--- Processing SMILES: {smiles} ---")
        print(f"🔹 Extracted smask: {smask}")

        # **生成分子图**
        g_rgcn = construct_RGCN_mol_graph_from_smiles(smiles, smask=smask)

        # **存入 `dataset_gnn`**
        molecule = [
            smiles, g_rgcn, labels.loc[i], split_index.loc[i], smask, 
            dataset_smiles.loc[i, 'antecedents'],
            dataset_smiles.loc[i, 'consequents'],
            dataset_smiles.loc[i, 'support'],
            dataset_smiles.loc[i, 'confidence'],
            dataset_smiles.loc[i, 'lift'],
            dataset_smiles.loc[i, 'leverage'],
            dataset_smiles.loc[i, 'conviction'],
            dataset_smiles.loc[i, 'zhangs_metric'],
            dataset_smiles.loc[i, 'brics_indices'],
            dataset_smiles.loc[i, 'brics_comb_indices'],
            dataset_smiles.loc[i, 'ante_indices'],
            dataset_smiles.loc[i, 'ante_comb_indices'],
            dataset_smiles.loc[i, 'conse_indices'],
            dataset_smiles.loc[i, 'conse_comb_indices'],
            dataset_smiles.loc[i, 'ante_conse_indices']
        ]

        dataset_gnn.append(molecule)
        print(f"✅ {i + 1}/{molecule_number} molecule is transformed successfully!")


    print(f'{len(failed_molecule)} molecules failed to transform.')
    return dataset_gnn  



def build_mol_graph_data_for_murcko_emerge(dataset_smiles, labels_name, smiles_name):
    dataset_gnn = []
    failed_molecule = []
    labels = dataset_smiles[labels_name]
    split_index = dataset_smiles['group']
    smilesList = dataset_smiles[smiles_name]
    molecule_number = len(smilesList)
    for i, smiles in enumerate(smilesList):
        substructure_dir = return_murcko_leaf_structure(smiles)
        atom_mask = []
        murcko_substructure_mask = []
        for _, substructure in substructure_dir['substructure'].items():
            murcko_substructure_mask.append(substructure)
            atom_mask = atom_mask + substructure
        smask = murcko_substructure_mask
        max_index_len = len(smask)
        all_pair_index = cal_pair_index(max_index_len)
        random.shuffle(all_pair_index)
        all_pair_index = all_pair_index[:100]
        for j, pair_index in enumerate(all_pair_index):
            pair_sub = emerge_sub(smask, pair_index)
            try:
                g_rgcn_1 = construct_RGCN_mol_graph_from_smiles(smiles, smask=pair_sub[0])
                molecule_1 = [smiles, g_rgcn_1, labels.loc[i], split_index.loc[i], pair_sub[0], 'emerge_{}'.format(j)]
                dataset_gnn.append(molecule_1)
                g_rgcn_2 = construct_RGCN_mol_graph_from_smiles(smiles, smask=pair_sub[1])
                molecule_2 = [smiles, g_rgcn_2, labels.loc[i], split_index.loc[i], pair_sub[1], 'emerge_{}'.format(j)]
                dataset_gnn.append(molecule_2)
                print('{}/{}, {}/{} molecule is transformed to mol graph! {} is transformed failed!'.format(j + 1,
                                                                                                            len(smask),
                                                                                                            i + 1,
                                                                                                            molecule_number,
                                                                                                            len(failed_molecule)))
            except:
                print('{} is transformed to mol graph failed!'.format(smiles))
                molecule_number = molecule_number - 1
                failed_molecule.append(smiles)
    print('{}({}) is transformed to mol graph failed!'.format(failed_molecule, len(failed_molecule)))
    return dataset_gnn




#子结构做预测的时候用
def load_graph_from_csv_bin_for_splited(
        bin_path='g_atom.bin',
        group_path='g_group.csv',
        smask_path=None,
        classification=True,
        random_shuffle=True,
        seed=2022
):
    data = pd.read_csv(group_path)
    smiles = data.smiles.values  

    antecedents = data['antecedents'].values
    consequents = data['consequents'].values
    support = data['support'].values
    confidence = data['confidence'].values
    lift = data['lift'].values
    leverage = data['leverage'].values
    conviction = data['conviction'].values
    zhangs_metric = data['zhangs_metric'].values
    brics_indices = data['brics_indices'].values
    brics_comb_indices = data['brics_comb_indices'].values
    ante_indices = data['ante_indices'].values
    ante_comb_indices = data['ante_comb_indices'].values
    conse_indices = data['conse_indices'].values
    conse_comb_indices = data['conse_comb_indices'].values
    ante_conse_indices = data['ante_conse_indices'].values

    group = data.group.to_list()
    # load substructure name
    if 'sub_name' in data.columns.tolist():
        sub_name = data['sub_name']
    else:
        sub_name = ['noname' for x in group]

    if random_shuffle:
        random.seed(seed)
        random.shuffle(group)
    homog, detailed_information = load_graphs(bin_path)
    labels = detailed_information['labels']

    # load smask
    if smask_path is None:
        smask = [-1 for x in range(len(group))]
    else:
        smask = np.load(smask_path, allow_pickle=True)

    # calculate not_use index
    train_index = []
    val_index = []
    test_index = []
    T_20_mol_generator_index= []
    B_20_mol_generator_index= []

    for index, group_index in enumerate(group):
        if group_index == 'training':
            train_index.append(index)
        elif group_index == 'valid':
            val_index.append(index)
        elif group_index == 'test':
            test_index.append(index)
        elif group_index == 'T_20_mol_generator':
            T_20_mol_generator_index.append(index)
        elif group_index == 'B_20_mol_generator':
            B_20_mol_generator_index.append(index)

    task_number = 1
    train_set = []
    val_set = []
    test_set = []
    T_20_mol_generator_set= []
    B_20_mol_generator_set= []


    for i in train_index:
        molecule = [smiles[i], homog[i], labels[i], smask[i], sub_name[i], antecedents[i],
    consequents[i],
    support[i],
    confidence[i],
    lift[i],
    leverage[i],
    conviction[i],
    zhangs_metric[i],
    brics_indices[i],
    brics_comb_indices[i],
    ante_indices[i],
    ante_comb_indices[i],
    conse_indices[i],
    conse_comb_indices[i],
    ante_conse_indices[i]]
        train_set.append(molecule)

    for i in val_index:
        molecule = [smiles[i], homog[i], labels[i], smask[i], sub_name[i], antecedents[i],
    consequents[i],
    support[i],
    confidence[i],
    lift[i],
    leverage[i],
    conviction[i],
    zhangs_metric[i],
    brics_indices[i],
    brics_comb_indices[i],
    ante_indices[i],
    ante_comb_indices[i],
    conse_indices[i],
    conse_comb_indices[i],
    ante_conse_indices[i]]
        val_set.append(molecule)

    for i in test_index:
        molecule = [smiles[i], homog[i], labels[i], smask[i], sub_name[i], antecedents[i],
    consequents[i],
    support[i],
    confidence[i],
    lift[i],
    leverage[i],
    conviction[i],
    zhangs_metric[i],
    brics_indices[i],
    brics_comb_indices[i],
    ante_indices[i],
    ante_comb_indices[i],
    conse_indices[i],
    conse_comb_indices[i],
    ante_conse_indices[i]]
        test_set.append(molecule)

    for i in T_20_mol_generator_index:
        molecule = [smiles[i], homog[i], labels[i], smask[i], sub_name[i], ]
        T_20_mol_generator_set.append(molecule)

    for i in B_20_mol_generator_index:
        molecule = [smiles[i], homog[i], labels[i], smask[i], sub_name[i],  ]
        B_20_mol_generator_set.append(molecule)

    print(len(train_set), len(val_set), len(test_set), task_number)
    return train_set, val_set, test_set, T_20_mol_generator_set,B_20_mol_generator_set,task_number


def build_mol_graph_data_for_brics(dataset_smiles, labels_name, smiles_name, substructure_data):
    dataset_gnn = []
    failed_molecule = []
    labels = dataset_smiles[labels_name]
    smilesList = dataset_smiles[smiles_name]
    split_index = dataset_smiles['group']

    for i, smiles in enumerate(smilesList):
        print(f"\n--- Processing SMILES {i+1}/{len(smilesList)}: {smiles} ---")

        # 读取 ante_comb_indices 和 conse_comb_indices
        try:
            ante_comb_indices_str = substructure_data.loc[i, 'ante_comb_indices']
            conse_comb_indices_str = substructure_data.loc[i, 'conse_comb_indices']
            ante_comb_indices = eval(ante_comb_indices_str)
            conse_comb_indices = eval(conse_comb_indices_str)
            print(f"Parsed ante_comb_indices: {ante_comb_indices}")
            print(f"Parsed conse_comb_indices: {conse_comb_indices}")
        except Exception as e:
            print(f"[Error] Parsing ante_comb_indices or conse_comb_indices failed for {smiles}: {e}")
            failed_molecule.append(smiles)
            continue

        # 创建一个包含 ante_comb_indices 和 conse_comb_indices 的列表
        comb_indices_list = [ante_comb_indices, conse_comb_indices]

        # 遍历 ante_comb_indices 和 conse_comb_indices
        for comb_indices in comb_indices_list:
            print(f"Processing combination of indices: {comb_indices}")
            smask=comb_indices
            # 生成分子图
            try:
                g_rgcn = construct_RGCN_mol_graph_from_smiles(smiles, smask=smask)
                print(f"[Success] Generated molecule graph for mask: {smask}")

                # 组装 molecule 数据
                molecule = [
                    smiles, g_rgcn, labels.loc[i], split_index.loc[i], smask,
                    substructure_data.loc[i, 'antecedents'],
                    substructure_data.loc[i, 'consequents'],
                    substructure_data.loc[i, 'support'],
                    substructure_data.loc[i, 'confidence'],
                    substructure_data.loc[i, 'lift'],
                    substructure_data.loc[i, 'leverage'],
                    substructure_data.loc[i, 'conviction'],
                    substructure_data.loc[i, 'zhangs_metric'],
                    substructure_data.loc[i, 'brics_indices'],
                    substructure_data.loc[i, 'brics_comb_indices'],
                    substructure_data.loc[i, 'ante_indices'],
                    substructure_data.loc[i, 'ante_comb_indices'],
                    substructure_data.loc[i, 'conse_indices'],
                    substructure_data.loc[i, 'conse_comb_indices'],
                    substructure_data.loc[i, 'ante_conse_indices']
                ]

                dataset_gnn.append(molecule)
                print(f"[Success] Added molecule data for {smiles}")
            except Exception as e:
                print(f"[Error] Failed to generate molecule graph for {smiles} with mask {smask}: {e}")
                failed_molecule.append(smiles)

    print(f"\n[Summary] Processed {len(dataset_gnn)} molecules successfully, {len(failed_molecule)} molecules failed.")
    return dataset_gnn


def built_mol_graph_data_and_save(
    task_name='None',
    origin_data_path='/HOME/scz4306/run/SME/optimization/data/subnear_association_rules_withsmiles/Mutagenicity_subnear_association_rules_withsmiles.csv',
    labels_name='label',
    save_g_path='No_name_mol_graph.bin',
    save_g_group_path='No_name_mol_graph_group.csv',
    save_g_for_brics_smarts_path='No_name_mol_graph_for_brics.csv',  # 这里是存储 SMARTS 的路径
    save_g_smask_for_brics_smarts_path='No_name_mol_graph_smask_for_brics.npy',
    save_g_group_for_brics_smarts_path='No_name_mol_graph_group_for_brics.csv',  # 这里是存储 group 的路径
    save_g_for_brics_combine_path='No_name_mol_graph_for_brics_emerge.bin',
    save_g_smask_for_brics_combine_path='No_name_mol_graph_smask_for_brics_combine.npy',
    save_g_group_for_brics_combine_path='No_name_mol_graph_group_for_brics_combine.csv',
    save_g_for_fg_path='No_name_mol_graph_for_fg.bin',
    save_g_group_for_fg_path='No_name_mol_graph_group_for_fg.csv',
    save_g_smask_for_fg_path='No_name_mol_graph_smask_for_fg.npy'
):
    data_origin = pd.read_csv(origin_data_path, index_col=None)
    smiles_name = 'smiles'

 
    data_set_gnn = build_mol_graph_data(dataset_smiles=data_origin, labels_name=labels_name, smiles_name=smiles_name)
    
    # 解包结果并打印
    smiles, g_rgcn, labels, split_index = map(list, zip(*data_set_gnn))
    graph_labels = {'labels': th.tensor(labels)}

    split_index_pd = pd.DataFrame(columns=['smiles', 'group'])
    split_index_pd.smiles = smiles
    split_index_pd.group = split_index
    split_index_pd.label = labels
    split_index_pd.to_csv(save_g_group_path, index=False, columns=None)
    print('Molecule graph group data is saved!')
    save_graphs(save_g_path, g_rgcn, graph_labels)
    
    # build data for brics 
    data_origin = pd.read_csv(origin_data_path, index_col=None)
    smiles_name = 'smiles'
    substructure_data = pd.read_csv(origin_data_path)
    
    data_set_gnn_for_brics = build_mol_graph_data_for_brics(
        dataset_smiles=data_origin, 
        labels_name=labels_name,
        smiles_name=smiles_name,
        substructure_data=substructure_data
    )

    # 解析 `data_set_gnn_for_brics`，包括新添加的列
    (
        smiles, g_rgcn, labels, split_index, smask, 
        antecedents, consequents, support, confidence, 
        lift, leverage, conviction, zhangs_metric, 
        brics_indices, brics_comb_indices,ante_indices,
       ante_comb_indices, conse_indices,conse_comb_indices,ante_conse_indices
    ) = map(list, zip(*data_set_gnn_for_brics))

    # 打印 smask 的部分内容
    print("查看 smask 内容:")
    for i, mask in enumerate(smask[:10]):  # 只查看前10个 smask
        print(f"第 {i+1} 个 smask: {mask}")

    graph_labels = {'labels': th.tensor(labels)}

    # 保存 smask 数据
    smask_np = np.array(smask, dtype=object)
    np.save(save_g_smask_for_brics_smarts_path, smask_np)

    # 保存 `substructure_smart_keys_df`，包含所有新列
    substructure_smart_keys_df = pd.DataFrame({
        'smiles': smiles,  
        'group': split_index, 
        'labels':labels, 
        'sub_smask': smask,
        'antecedents': antecedents,
        'consequents': consequents,
        'support': support,
        'confidence': confidence,
        'lift': lift,
        'leverage': leverage,
        'conviction': conviction,
        'zhangs_metric': zhangs_metric,
        'brics_indices': brics_indices,
        'brics_comb_indices': brics_comb_indices,
        'ante_indices':ante_indices,
        'ante_comb_indices':ante_comb_indices,
        'conse_indices':conse_indices,
        'conse_comb_indices':conse_comb_indices,
        'ante_conse_indices':ante_conse_indices
    })


    # 保存所有数据到 CSV
    substructure_smart_keys_df.to_csv(save_g_group_for_brics_smarts_path, index=False)
    print(f"Substructure SMARTS keys and other data are saved to {save_g_group_for_brics_smarts_path}")

    # 保存图数据
    save_graphs(save_g_for_brics_smarts_path, g_rgcn, graph_labels)  


    # build data for brics_combine
    data_set_gnn = build_mol_graph_data_for_brics_combine(
        dataset_smiles=data_origin, 
        labels_name=labels_name, 
        smiles_name=smiles_name
    )

    (
        smiles, g_rgcn, labels, split_index, smask, 
        antecedents, consequents, support, confidence, 
        lift, leverage, conviction, zhangs_metric, 
        brics_indices, brics_comb_indices,ante_indices,
       ante_comb_indices, conse_indices,conse_comb_indices,ante_conse_indices
    ) = map(list, zip(*data_set_gnn))


    print("查看 smask 内容:")
    for i, mask in enumerate(smask[:10]):  # 只查看前10个 smask
        print(f"第 {i+1} 个 brics_combine_mask: {mask}")

    graph_labels = {'labels': th.tensor(labels)}

    # 保存 `split_index_pd`，包含所有新列
    split_index_pd = pd.DataFrame({
        'smiles': smiles,
        'group': split_index, 
        'labels':labels,
        'smask': smask,
        'antecedents': antecedents,
        'consequents': consequents,
        'support': support,
        'confidence': confidence,
        'lift': lift,
        'leverage': leverage,
        'conviction': conviction,
        'zhangs_metric': zhangs_metric,
        'brics_indices': brics_indices,
        'brics_comb_indices': brics_comb_indices,
        'ante_indices':ante_indices,
        'ante_comb_indices':ante_comb_indices,
        'conse_indices':conse_indices,
        'conse_comb_indices':conse_comb_indices,
        'ante_conse_indices':ante_conse_indices
    })

    # 保存数据到 CSV
    split_index_pd.to_csv(save_g_group_for_brics_combine_path, index=False)
    print(f"Substructure SMARTS keys and other data are saved to {save_g_group_for_brics_combine_path}")

    # 保存子结构的掩码
    brics_combine_mask_np = np.array(smask, dtype=object)
    np.save(save_g_smask_for_brics_combine_path, brics_combine_mask_np)

    # 保存图数据
    save_graphs(save_g_for_brics_combine_path, g_rgcn, graph_labels)

    print('Molecules graph for brics combine is saved!')