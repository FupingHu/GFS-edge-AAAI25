import torch
from sklearn.cluster import KMeans, SpectralClustering


def get_antecedent_parameters(inputs_train, rules):
    kmeans = KMeans(n_clusters=rules, random_state=0)
    inputs_train = inputs_train.cpu()
    inputs_train = kmeans.fit(inputs_train)
    cluster_centers_nodes = inputs_train.cluster_centers_
    return cluster_centers_nodes


def mu_norm(mus):
    mus_norm = []
    mus_total = 0
    for mu in mus:
        mus_total = mu + mus_total
    for mu in mus:
        mus_norm.append(mu / mus_total)
    return mus_norm


def Mu_Norm_List(features, centers):
    mu_norm_list = []
    for feature in features:
        mu = []
        feature = torch.unsqueeze(feature, 1)
        for center in centers:
            center = torch.unsqueeze(center, 0)
            feature = feature.cpu()
            tmp_mu = torch.mm(center, feature)
            tmp_mu = torch.sigmoid(tmp_mu)
            tmp_mu = tmp_mu.squeeze(1)
            tmp_mu = tmp_mu.squeeze(0)
            mu.append(tmp_mu.tolist())
        mu = mu_norm(mu)
        mu_norm_list.append(mu)

    mu_norm_list = torch.tensor(mu_norm_list)
    mu_norm_list = mu_norm_list.unsqueeze(2)
    return mu_norm_list


def Mu_Norm_List_Cluster(features, centers):
    mu_norm_list = []
    for feature in features:
        mu = []
        feature = torch.unsqueeze(feature, 1)
        for center in centers:
            center = torch.unsqueeze(center, 0)
            feature = feature.cpu()
            tmp_mu = torch.mm(center, feature)
            tmp_mu = torch.sigmoid(tmp_mu)
            tmp_mu = tmp_mu.squeeze(1)
            tmp_mu = tmp_mu.squeeze(0)
            mu.append(tmp_mu.tolist())
        mu = mu_norm(mu)
        mu_norm_list.append(mu)
    mu_norm_list = torch.tensor(mu_norm_list)
    mu_norm_list = mu_norm_list.unsqueeze(2)
    return mu_norm_list
