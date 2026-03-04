import pandas as pd
import numpy as np
import torch
from functools import partial
import dgl
from dgllife.utils import smiles_to_bigraph
from model3.featurizers import CanonicalAtomFeaturizer
from model3.featurizers import CanonicalBondFeaturizer
from dgllife.data import MoleculeCSVDataset
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski, QED
if torch.cuda.is_available():
    #print('use GPU')
    device = 'cuda'
else:
    #print('use CPU')
    device = 'cpu'

def collate_molgraphs(data):
    assert len(data[0]) in [3, 4], \
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
        masks = None
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)
    return smiles, bg, labels, masks



atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='hv')
bond_featurizer = CanonicalBondFeaturizer(bond_data_field='he', self_loop=True)
def load_data(data,id):
    dataset = MoleculeCSVDataset(data,
                                 smiles_to_graph=partial(smiles_to_bigraph, add_self_loop='self_loop'),
                                 node_featurizer=atom_featurizer,
                                 edge_featurizer=bond_featurizer,
                                 smiles_column='SMILES',
                                 cache_file_path='/HOME/scz4306/run/SME/optimization/RL/model3/'  + 'graph.bin',
                                 load=False,init_mask=True,n_jobs=1
                                 )

    return dataset






def Physicochemical_property_calc(SMILES):
    MW = []
    HBA = []
    HBD = []
    nRot = []
    TPSA = []
    SlogP = []
    nRing = []
    nAtom = []
    nHet = []
    QED_v = []
    for i in SMILES:
        try:
            mol = Chem.MolFromSmiles(i)
            # MW
            mw = Descriptors.MolWt(mol)
            # mw = round(mw, 3)
            MW.append(mw)
            # HBA & HBD
            hba = rdMolDescriptors.CalcNumHBA(mol)
            HBA.append(hba)
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            HBD.append(hbd)
            # nROT
            nrot = Lipinski.NumRotatableBonds(mol)
            nRot.append(nrot)
            # TPSA
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            TPSA.append(tpsa)
            # logP
            logP = Descriptors.MolLogP(mol)
            SlogP.append(logP)
            # nRing
            nR = rdMolDescriptors.CalcNumRings(mol)
            nRing.append(nR)
            # nAtom
            nA = mol.GetNumAtoms()
            nAtom.append(nA)
            # nHet
            nH = Lipinski.NumHeteroatoms(mol)
            nHet.append(nH)
            # QED
            qed = QED.default(mol)
            QED_v.append(qed)
        except Exception:
            MW.append(None)
            HBA.append(None)
            HBD.append(None)
            nRot.append(None)
            TPSA.append(None)
            SlogP.append(None)
            nRing.append(None)
            nAtom.append(None)
            nHet.append(None)
            QED_v.append(None)
    return MW,nAtom,nHet,nRing,nRot,HBA,HBD,TPSA,SlogP,QED_v

def Rule_calc(profile):
    # profile = pd.read_csv(query, index_col='SMILES').values
    n = len(profile)
    L_rules = []
    p_rules = []
    GSK_rules = []
    try:
        for i in range(n):
            j = []
            data_line = profile[i]
            if np.isnan(data_line[0])==True :
                L_rules.append(None)
                p_rules.append(None)
                GSK_rules.append(None)
            else:
                # Lipinski_rule
                # MW
                if data_line[0] > 500:
                    j.append(1)
                else:
                    j.append(0)
                # HBA
                if data_line[1] > 10:
                    j.append(1)
                else:
                    j.append(0)
                # HBD
                if data_line[2] > 5:
                    j.append(1)
                else:
                    j.append(0)
                # logP
                if data_line[4] > 5:
                    j.append(1)
                else:
                    j.append(0)

                result = sum(j)
                if result > 1:
                    Li_rule = 'Not accept'
                else:
                    Li_rule = 'Accept'
                L_rules.append(Li_rule)

                # Pfizer_rule
                if data_line[4] > 3 and data_line[3] < 75:
                    Pfizer_rule = 'Not accept'
                else:
                    Pfizer_rule = 'Accept'
                p_rules.append(Pfizer_rule)

                # GSK_rule
                if data_line[0] <= 400 and data_line[4] <= 4:
                    gsk_rule = 'Accept'
                else:
                    gsk_rule = 'Not accept'
                GSK_rules.append(gsk_rule)

    except Exception:
        L_rules.append(None)
        p_rules.append(None)
        GSK_rules.append(None)

    return L_rules,p_rules,GSK_rules


