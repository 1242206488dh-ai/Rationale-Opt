import numpy as np
import torch.nn as nn
import torch
import dgl
import torch.nn.functional as F
from dgllife.model.gnn import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax
from sklearn import preprocessing
if torch.cuda.is_available():
    print('use GPU')
    device = 'cuda'
else:
    print('use CPU')
    device = 'cpu'



class ModelPredictor(nn.Module):
    def __init__(self, node_feat_size,edge_feat_size, graph_feat_size, num_layers=None,
                 dropout=0., n_tasks=1,num_timesteps=None,predictor_hidden_feats=128):
        super(ModelPredictor, self).__init__()

        self.gnn_drug = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size,
                                  dropout=dropout)

        self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
                                          num_timesteps=num_timesteps,
                                          dropout=dropout)

        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.LayerNorm(predictor_hidden_feats),
            nn.Linear(predictor_hidden_feats, 64),
            nn.LeakyReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, n_tasks),
        )


    def forward(self, bg, n_feats, e_feats, get_node_gradient=False, n=0):
        node_feats = self.gnn_drug(bg, n_feats, e_feats,)
        drug_feats = self.readout(bg, node_feats)
        if get_node_gradient:
            # Calculate graph representation by average readout.
            Final_feature = self.predict(drug_feats)[:, n:n+1]
            baseline = torch.zeros(node_feats.shape).to(device)
            scaled_nodefeats = [baseline + (float(i) / 50) * (node_feats - baseline) for i in range(0, 51)]
            gradients = []
            for scaled_nodefeat in scaled_nodefeats:
                scaled_hg = self.readout(bg, scaled_nodefeat)
                scaled_Final_feature = self.predict(scaled_hg)[:, n:n+1]
                gradient = torch.autograd.grad(scaled_Final_feature[0][0], scaled_nodefeat)[0]
                gradient = gradient.detach().cpu().numpy()
                gradients.append(gradient)
            gradients = np.array(gradients)
            grads = (gradients[:-1] + gradients[1:]) / 2.0
            avg_grads = np.average(grads, axis=0)
            avg_grads = torch.from_numpy(avg_grads).to(device)
            integrated_gradients = (node_feats - baseline) * avg_grads
            phi0 = []
            for j in range(node_feats.shape[0]):
                a = sum(integrated_gradients[j].detach().cpu().numpy().tolist())
                phi0.append(a)
            node_gradient = torch.tensor(phi0)
            node_gradient = node_gradient.reshape(-1, 1)
            node_gradient = preprocessing.MaxAbsScaler().fit_transform(node_gradient)
            return Final_feature, node_gradient
        else:
            Final_feature = self.predict(drug_feats)
            return Final_feature

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)