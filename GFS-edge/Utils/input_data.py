import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sklearn.preprocessing as preprocess
import scipy.io as sio


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_wiki():
    f = open('../data/graph.txt', 'r')
    adj, xind, yind = [], [], []
    for line in f.readlines():
        line = line.split()

        xind.append(int(line[0]))
        yind.append(int(line[1]))
        adj.append([int(line[0]), int(line[1])])
    f.close()
    # print(len(adj))

    f = open('../data/group.txt', 'r')
    label = []
    for line in f.readlines():
        line = line.split()
        label.append(int(line[1]))
    f.close()

    f = open('../data/tfidf.txt', 'r')
    fea_idx = []
    fea = []
    adj = np.array(adj)
    adj = np.vstack((adj, adj[:, [1, 0]]))
    adj = np.unique(adj, axis=0)

    labelset = np.unique(label)
    labeldict = dict(zip(labelset, range(len(labelset))))
    label = np.array([labeldict[x] for x in label])
    adj = sp.csr_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(len(label), len(label)))

    for line in f.readlines():
        line = line.split()
        fea_idx.append([int(line[0]), int(line[1])])
        fea.append(float(line[2]))
    f.close()

    fea_idx = np.array(fea_idx)
    # features = sp.csr_matrix((fea, (fea_idx[:, 0], fea_idx[:, 1])), shape=(len(label), 4973)).toarray()
    features = sp.csr_matrix((fea, (fea_idx[:, 0], fea_idx[:, 1])), shape=(len(label), 4973)).tolil()
    # scaler = preprocess.MinMaxScaler()
    scaler = preprocess.MaxAbsScaler()
    # features = preprocess.normalize(features, norm='l2')
    features = scaler.fit_transform(features)
    # features = torch.FloatTensor(features)

    return adj, features, label


def aug_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def fetch_normalization(type):
    switcher = {
        'AugNormAdj': aug_normalized_adjacency,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    }
    func = switcher.get(type, lambda: "Invalid normalization technique.")
    return func


def row_normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features


def load_nell(label_rate=0.001):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}.{}".format('nell', label_rate, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.{}.test.index".format('nell', label_rate))

    test_idx_range = np.sort(test_idx_reorder)

    test_idx_range_full = range(allx.shape[0], len(graph))
    isolated_node_idx = np.setdiff1d(test_idx_range_full, test_idx_reorder)
    tx_extended = sp.lil_matrix((len(test_idx_range_full), tx.shape[1]))
    tx_extended[test_idx_range - allx.shape[0], :] = tx
    tx = tx_extended
    ty_extended = np.zeros((len(test_idx_range_full), ty.shape[1]))
    ty_extended[test_idx_range - allx.shape[0], :] = ty
    ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_all = np.setdiff1d(range(len(graph)), isolated_node_idx)

    # features = sp.vstack((allx, tx)).tolil()
    # features[test_idx_reorder, :] = features[test_idx_range, :]
    # adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # labels = np.vstack((ally, ty))
    # labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    adj, features = preprocess_citation(adj, features, normalization="AugNormAdj")

    return adj, features


def load_data(dataset):
    if dataset in (
    "USAir", "NS", "PB", "Yeast", "Celegans", "Power", "Router", "Ecoli", "BUP", "KHN", "YST", "ADV", "LDG", "HPD",
    "GRQ", "ZWL", "EML"):
        GraphData = sio.loadmat('./data/{}.mat'.format(dataset))
        from scipy.sparse import lil_matrix
        adj = GraphData['net']
        features = GraphData['net']
        features = lil_matrix(features)
        return adj, features
    if dataset == 'wiki':
        adj, features, label = load_wiki()
        return adj, features
    if dataset == 'nell':
        adj, features = load_nell()
        return adj, features
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)
    if dataset == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features
