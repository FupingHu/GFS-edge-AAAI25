import json
import os
import time
import warnings
import torch
import torch.nn.functional as F
from torch.optim import Adam
import pandas
from Models import model
from Utils.Utils import get_antecedent_parameters, Mu_Norm_List
from Utils.Metrics import get_scores, get_acc
from Utils.input_data import load_data
from Utils.preprocessing import *
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    HyperParams_json = open('config/HyperParams_Without.json')
    HyperParamConfig = json.load(HyperParams_json)
    Datasets = HyperParamConfig['Datasets']
    Rules = HyperParamConfig['Rules']
    Epoch = HyperParamConfig['Epoch']
    Cons_Models = HyperParamConfig['Cons_Models']
    Lrs = HyperParamConfig['Lrs']
    L2s = HyperParamConfig['L2s']
    HiddenDims1 = HyperParamConfig['HiddenDims1']
    HiddenDims2 = HyperParamConfig['HiddenDims2']
    Alphas = HyperParamConfig['Alphas']
    Betas = HyperParamConfig['Betas']
    Num_Val_Ratio_Pers = HyperParamConfig['Num_Val_Ratio_Per']
    for Dataset in Datasets:
        for Num_Val_Ratio_Per in Num_Val_Ratio_Pers:
            Num_Train_Ratio_Per = 90.00 - Num_Val_Ratio_Per
            adj, features = load_data(Dataset)
            adj_orig = adj
            adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
            adj_orig.eliminate_zeros()
            adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj, Num_Val_Ratio_Per)
            adj = adj_train
            # Some preprocessing
            adj_norm = preprocess_graph(adj)
            num_nodes = adj.shape[0]
            features = sparse_to_tuple(features.tocoo())
            num_features = features[2][1]
            features_nonzero = features[1].shape[0]
            pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
            norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
            adj_label = adj_train + sp.eye(adj_train.shape[0])
            adj_label = sparse_to_tuple(adj_label)
            adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                                                torch.FloatTensor(adj_norm[1]),
                                                torch.Size(adj_norm[2]))
            adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),
                                                 torch.FloatTensor(adj_label[1]),
                                                 torch.Size(adj_label[2]))
            features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T),
                                                torch.FloatTensor(features[1]),
                                                torch.Size(features[2]))
            weight_mask = adj_label.to_dense().view(-1) == 1
            weight_tensor = torch.ones(weight_mask.size(0))
            weight_tensor[weight_mask] = pos_weight
            input_dim = features.size(1)
            for cons_model in Cons_Models:
                for rule in Rules:
                    for L2 in L2s:
                        for Lr in Lrs:
                            for HiddenDim1 in HiddenDims1:
                                for HiddenDim2 in HiddenDims2:
                                    for Alpha in Alphas:
                                        for Beta in Betas:
                                            features_dense = features.to_dense()
                                            cluster_centers = get_antecedent_parameters(features_dense, rule)
                                            cluster_centers = torch.FloatTensor(cluster_centers).float()
                                            mu_norm_list = Mu_Norm_List(features_dense, cluster_centers)
                                            Model = model.GFS_edge(adj_norm, rule, input_dim, HiddenDim1, HiddenDim2,
                                                                Alpha,
                                                                Beta)
                                            optimizer = Adam(Model.parameters(), lr=Lr, weight_decay=L2)
                                            for epoch in range(Epoch):
                                                start_time = time.time()
                                                A_preds, Struct_Infos = Model(features)
                                                A_pred = torch.cat(
                                                    [A_pred_rule.unsqueeze(2) for A_pred_rule in A_preds], 2)
                                                A_pred = torch.matmul(A_pred, mu_norm_list)
                                                A_pred = A_pred.squeeze(2)
                                                optimizer.zero_grad()
                                                loss = log_lik = norm * F.binary_cross_entropy_with_logits(
                                                    A_pred.view(-1),
                                                    adj_label.to_dense().view(
                                                        -1),
                                                    weight=weight_tensor)
                                                EP_Cons = Model.EP_Cons
                                                count_num = 0
                                                KL_Div = 0
                                                for sub_model in EP_Cons:
                                                    kl_divergence = 0.5 / A_pred.size(0) * (
                                                            1 + 2 * sub_model.logstd - sub_model.mean ** 2 - torch.exp(
                                                        sub_model.logstd) ** 2).sum(
                                                        1).mean()
                                                    KL_Div += kl_divergence
                                                    count_num += 1
                                                KL_Div = KL_Div / count_num
                                                loss -= KL_Div
                                                loss.backward()
                                                optimizer.step()
                                                train_acc = get_acc(A_pred, adj_label)
                                                val_roc, val_ap, val_auc = get_scores(val_edges, val_edges_false,
                                                                                      A_pred.cpu(),
                                                                                      adj_orig)
                                                print(
                                                    "Epoch:" + '%04d' % (epoch + 1) + ", train_loss=" + "{:.5f},".format(
                                                        loss.item()) +
                                                    " train_acc=" + "{:.5f},".format(train_acc) + " val_roc=" +
                                                    "{:.5f},".format(val_roc) +
                                                    " val_ap=" + "{:.5f},".format(
                                                        val_ap) + " val_auc=" + "{:.5f},".format(
                                                        val_auc) +
                                                    " time=" + "{:.5f}".format(time.time() - start_time))

                                            test_roc, test_ap, test_auc = get_scores(test_edges, test_edges_false,
                                                                                     A_pred.cpu(),
                                                                                     adj_orig)
                                            print("End of training!" + " test_roc=" + "{:.5f}".format(test_roc) +
                                                        " test_ap=" + "{:.5f}".format(
                                                test_ap) + " test_auc=" + "{:.5f}".format(
                                                test_auc))


