import os
import torch
from torch_geometric.utils import dense_to_sparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(torch.cuda.device_count())
print(torch.cuda.is_available())


import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import abc
import torch.nn.utils as utils
from torch_geometric.data import Data
from sklearn.metrics import classification_report, accuracy_score
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.init as init
import math
import argparse
import pickle
import json
import json, os, time
import threading
import argparse
import config_file
import random
from time import *
from PIL import Image
import sys

from torch_geometric.nn import GINConv, global_mean_pool
from torch.optim.lr_scheduler import ReduceLROnPlateau


sys.path.append('/../image_part')


parser = argparse.ArgumentParser()
parser.description = "ini"
parser.add_argument("-t", "--task", type=str, default="weibo2")
parser.add_argument("-g", "--gpu_id", type=str, default="1")
parser.add_argument("-c", "--config_name", type=str, default="single3.json")
parser.add_argument("-T", "--thread_name", type=str, default="Thread-1")
parser.add_argument("-d", "--description", type=str, default="exp_description")
args = parser.parse_args()


def process_config(config):
    for k, v in config.items():
        config[k] = v[0]
    return config


class PGD(object):
    def __init__(self, model, emb_name, epsilon=1., alpha=0.3):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.wtrans = nn.Parameter(torch.zeros(size=(2 * out_features, out_features)))
        nn.init.xavier_uniform_(self.wtrans.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        h = torch.mm(inp, self.W)
        N = h.size()[0]
        Wh1 = torch.mm(h, self.a[:self.out_features, :])
        Wh2 = torch.mm(h, self.a[self.out_features:, :])
        e = self.leakyrelu(Wh1 + Wh2.T)
        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        negative_attention = torch.where(adj > 0, -e, zero_vec)
        attention = F.softmax(attention, dim=1)
        negative_attention = -F.softmax(negative_attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        negative_attention = F.dropout(negative_attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, inp)
        h_prime_negative = torch.matmul(negative_attention, inp)
        h_prime_double = torch.cat([h_prime, h_prime_negative], dim=1)
        new_h_prime = torch.mm(h_prime_double, self.wtrans)
        if self.concat:
            return F.elu(new_h_prime)
        else:
            return new_h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class Signed_GAT(nn.Module):
    def __init__(self, node_embedding, cosmatrix, nfeat, uV, original_adj, hidden=16,
                 nb_heads=4, n_output=300, dropout=0, alpha=0.3):
        super(Signed_GAT, self).__init__()
        self.dropout = dropout
        self.uV = uV
        embedding_dim = 300
        self.user_tweet_embedding = nn.Embedding(num_embeddings=self.uV, embedding_dim=embedding_dim, padding_idx=0)
        self.user_tweet_embedding.from_pretrained(torch.from_numpy(node_embedding))
        self.original_adj = torch.from_numpy(original_adj.astype(np.float64)).cuda()
        self.potentinal_adj = torch.where(cosmatrix > 0.5, torch.ones_like(cosmatrix),
                                          torch.zeros_like(cosmatrix)).cuda()
        self.adj = self.original_adj + self.potentinal_adj
        self.adj = torch.where(self.adj > 0, torch.ones_like(self.adj), torch.zeros_like(self.adj))

        self.attentions = [GraphAttentionLayer(nfeat, n_output, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nb_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nfeat * nb_heads, n_output, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, X_tid):
        X = self.user_tweet_embedding(torch.arange(0, self.uV).long().cuda()).to(torch.float32)
        x = F.dropout(X, self.dropout, training=self.training)
        adj = self.adj.to(torch.float32)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.sigmoid(self.out_att(x, adj))
        return x[X_tid]


class TransformerBlock(nn.Module):
    def __init__(self, input_size, d_k=16, d_v=16, n_heads=8, is_layer_norm=False, attn_dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else input_size
        self.d_v = d_v if d_v is not None else input_size
        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=input_size)
        self.W_q = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_k = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_v = nn.Parameter(torch.Tensor(input_size, n_heads * d_v))
        self.W_o = nn.Parameter(torch.Tensor(d_v * n_heads, input_size))
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout(attn_dropout)
        self.__init_weights__()

    def __init_weights__(self):
        init.xavier_normal_(self.W_q)
        init.xavier_normal_(self.W_k)
        init.xavier_normal_(self.W_v)
        init.xavier_normal_(self.W_o)
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        output = self.linear2(F.relu(self.linear1(X)))
        output = self.dropout(output)
        return output

    def scaled_dot_product_attention(self, Q, K, V, episilon=1e-6):
        temperature = self.d_k ** 0.5
        Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + episilon)
        Q_K_score = F.softmax(Q_K, dim=-1)
        Q_K_score = self.dropout(Q_K_score)
        V_att = Q_K_score.bmm(V)
        return V_att

    def multi_head_attention(self, Q, K, V):
        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()
        bsz, v_len, _ = V.size()

        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.n_heads, self.d_k)
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.n_heads, self.d_k)
        V_ = V.matmul(self.W_v).view(bsz, v_len, self.n_heads, self.d_v)

        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, q_len, self.d_k)
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, q_len, self.d_k)
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, q_len, self.d_v)

        V_att = self.scaled_dot_product_attention(Q_, K_, V_)
        V_att = V_att.view(bsz, self.n_heads, q_len, self.d_v)
        V_att = V_att.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.n_heads * self.d_v)
        output = self.dropout(V_att.matmul(self.W_o))
        return output

    def forward(self, Q, K, V):
        V_att = self.multi_head_attention(Q, K, V)
        if self.is_layer_norm:
            X = self.layer_morm(Q + V_att)
            output = self.layer_morm(self.FFN(X) + X)
        else:
            X = Q + V_att
            output = self.FFN(X) + X
        return output


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.best_acc = 0
        self.init_clip_max_norm = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    @abc.abstractmethod
    def forward(self, X_tid, X_text):
        pass

    def ginmcl(self, x_tid, x_text, y, loss, i, total, params, pgd_word):
        self.optimizer.zero_grad()
        logit_defense, dist = self.forward(x_tid, x_text)
        loss_classification = loss(logit_defense, y)
        loss_mse = nn.MSELoss()
        loss_dis = loss_mse(dist[0], dist[1])
        loss_defense = loss_classification + loss_dis
        loss_defense.backward()
        K = 3
        pgd_word.backup_grad()

        for t in range(K):
            pgd_word.attack(is_first_attack=(t == 0))
            if t != K - 1:
                self.zero_grad()
            else:
                pgd_word.restore_grad()
            loss_adv, dist = self.forward(x_tid, x_text)
            loss_adv = loss(loss_adv, y)
            loss_adv.backward()
        pgd_word.restore()

        self.optimizer.step()
        corrects = (torch.max(logit_defense, 1)[1].view(y.size()).data == y.data).sum()
        accuracy = 100 * corrects / len(y)
        print(
            'Batch[{}/{}] - loss: {:.6f}  accuracy: {:.4f}%({}/{})'.format(i + 1, total,
                                                                           loss_defense.item(),
                                                                           accuracy,
                                                                           corrects,
                                                                           y.size(0)))

    def fit(self, X_train_tid, X_train, y_train,
            X_dev_tid, X_dev, y_dev):
        if torch.cuda.is_available():
            self.cuda()
        batch_size = self.config['batch_size']
        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-3, weight_decay=0)

        scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3, verbose=True)

        X_train_tid = torch.LongTensor(X_train_tid)
        X_train = torch.LongTensor(X_train)
        y_train = torch.LongTensor(y_train)

        dataset = TensorDataset(X_train_tid, X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        loss = nn.CrossEntropyLoss()
        params = [(name, param) for name, param in self.named_parameters()]
        pgd_word = PGD(self, emb_name='word_embedding', epsilon=6, alpha=1.8)

        best_val_loss = float('inf')
        early_stopping_counter = 0
        early_stopping_patience = 5

        for epoch in range(self.config['epochs']):
            print("\nEpoch ", epoch + 1, "/", self.config['epochs'])
            self.train()
            avg_loss = 0
            avg_acc = 0
            for i, data in enumerate(dataloader):
                total = len(dataloader)
                batch_x_tid, batch_x_text, batch_y = (item.cuda(device=self.device) for item in data)
                self.ginmcl(batch_x_tid, batch_x_text, batch_y, loss, i, total, params, pgd_word)
                if self.init_clip_max_norm is not None:
                    utils.clip_grad_norm_(self.parameters(), max_norm=self.init_clip_max_norm)

            val_loss = self.evaluate(X_dev_tid, X_dev, y_dev)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print("Early stopping triggered.")
                    break

    def evaluate(self, X_dev_tid, X_dev, y_dev):
        X_dev_tid = torch.tensor(X_dev_tid).long().cuda()
        X_dev = torch.tensor(X_dev).long().cuda()
        y_dev = torch.tensor(y_dev).long().cuda()

        y_pred = self.predict(X_dev_tid, X_dev)
        acc = accuracy_score(y_dev.cpu(), y_pred)
        val_loss = F.cross_entropy(self.forward(X_dev_tid, X_dev)[0], y_dev).item()

        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.state_dict(), self.config['save_path'])
            print(classification_report(y_dev.cpu(), y_pred, target_names=self.config['target_names'], digits=5))
            print("Val set acc:", acc)
            print("Best val set acc:", self.best_acc)
            print("save model!!!   at ", self.config['save_path'])

        return val_loss

    def predict(self, X_test_tid, X_test):
        if torch.cuda.is_available():
            self.cuda()

        X_test_tid = torch.tensor(X_test_tid, dtype=torch.long, device=self.device)
        X_test = torch.tensor(X_test, dtype=torch.long, device=self.device)

        self.eval()
        y_pred = []
        dataset = TensorDataset(X_test_tid, X_test)
        dataloader = DataLoader(dataset, batch_size=50)

        for i, data in enumerate(dataloader):
            with torch.no_grad():
                batch_x_tid, batch_x_text = (item.cuda(device=self.device) for item in data)
                logits, dist = self.forward(batch_x_tid, batch_x_text, compute_contrastive_loss=False)
                predicted = torch.max(logits, dim=1)[1]
                y_pred += predicted.data.cpu().numpy().tolist()

        return y_pred


class resnet50():
    def __init__(self):
        self.newid2imgnum = config['newid2imgnum']
        self.model = models.resnet50(pretrained=True).cuda()
        self.model.fc = nn.Linear(2048, 300).cuda()
        torch.nn.init.eye_(self.model.fc.weight)
        self.path = os.path.dirname(os.getcwd()) + '/dataset/weibo/weibo_images/weibo_images_all/'
        self.trans = self.img_trans()

    def img_trans(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])
        return transform

    def forward(self, xtid):
        img_path = []
        img_list = []
        for newid in xtid.cpu().numpy():
            imgnum = self.newid2imgnum[newid]
            imgpath = self.path + imgnum + '.jpg'
            im = np.array(self.trans(Image.open(imgpath)))
            im = torch.from_numpy(np.expand_dims(im, axis=0)).to(torch.float32)
            img_list.append(im)
        batch_img = torch.cat(img_list, dim=0).cuda()
        img_output = self.model(batch_img)
        return img_output


class AttentionGINConv(nn.Module):
    def __init__(self, nn, out_channels):
        super(AttentionGINConv, self).__init__()
        self.nn = nn
        self.att = nn.Linear(2 * out_channels, 1)

    def forward(self, x, edge_index):
        edge_index, _ = dense_to_sparse(edge_index)
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j, edge_index, size):
        x_cat = torch.cat([x_i, x_j], dim=-1)
        alpha = self.att(x_cat)
        alpha = F.softmax(alpha, dim=1)
        return x_j * alpha

    import torch.nn.functional as F


class GINMCL(NeuralNetwork):
    def __init__(self, config, adj, original_adj):
        super(GINMCL, self).__init__()
        self.config = config
        self.uV = adj.shape[0]
        self.original_adj = torch.tensor(original_adj, dtype=torch.float32).cuda()
        self.edge_index, _ = dense_to_sparse(self.original_adj)
        self.edge_index = self.edge_index.cuda()

        embedding_weights = config['embedding_weights']
        V, D = embedding_weights.shape
        maxlen = config['maxlen']
        dropout_rate = config['dropout']

        self.mh_attention = TransformerBlock(input_size=300, n_heads=8, attn_dropout=0)
        self.word_embedding = nn.Embedding(num_embeddings=V, embedding_dim=D, padding_idx=0,
                                           _weight=torch.from_numpy(embedding_weights))
        self.cosmatrix = self.calculate_cos_matrix()

        self.gin1 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(300, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 300),
            torch.nn.BatchNorm1d(300)))

        self.gin2 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(300, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 300),
            torch.nn.BatchNorm1d(300)))

        self.gin3 = GINConv(nn.Sequential(
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.BatchNorm1d(300)))

        self.fc1 = nn.Linear(300, 300)
        self.fc2 = nn.Linear(300, 300)

        self.image_embedding = resnet50()
        self.convs = nn.ModuleList([nn.Conv1d(300, 100, kernel_size=K) for K in config['kernel_sizes']])
        self.max_poolings = nn.ModuleList([nn.MaxPool1d(kernel_size=maxlen - K + 1) for K in config['kernel_sizes']])

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(1800, 900)
        self.fc4 = nn.Linear(900, 600)
        self.fc1 = nn.Linear(600, 300)
        self.fc2 = nn.Linear(in_features=300, out_features=config['num_classes'])
        self.alignfc_g = nn.Linear(in_features=300, out_features=300)
        self.alignfc_t = nn.Linear(in_features=300, out_features=300)

        self.contrastive_loss_weight = 0.1

        self.init_weight()

    def calculate_cos_matrix(self):
        a, b = torch.from_numpy(self.config['node_embedding']), torch.from_numpy(self.config['node_embedding'].T)
        c = torch.mm(a, b)
        aa = torch.mul(a, a)
        bb = torch.mul(b, b)
        asum = torch.sqrt(torch.sum(aa, dim=1, keepdim=True))
        bsum = torch.sqrt(torch.sum(bb, dim=0, keepdim=True))
        norm = torch.mm(asum, bsum)
        res = torch.div(c, norm)
        return res

    def init_weight(self):
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)
        init.xavier_normal_(self.fc3.weight)
        init.xavier_normal_(self.fc4.weight)

    def forward(self, X_tid, X_text, compute_contrastive_loss=False, return_features=False):
        X_text = self.word_embedding(X_text)
        if self.config['user_self_attention']:
            X_text = self.mh_attention(X_text, X_text, X_text)
        X_text = X_text.permute(0, 2, 1)

        x = self.word_embedding(torch.arange(0, self.uV).long().cuda()).to(torch.float32)
        x = F.relu(self.gin1(x, self.edge_index))
        x = F.relu(self.gin2(x, self.edge_index))
        x = F.relu(self.gin3(x, self.edge_index))

        num_nodes_per_batch = x.size(0) // X_tid.size(0)
        batch = torch.arange(X_tid.size(0), device=x.device).repeat_interleave(num_nodes_per_batch)

        if batch.size(0) != x.size(0):
            if batch.size(0) > x.size(0):
                batch = batch[:x.size(0)]
            else:
                additional_indices = torch.arange(x.size(0) - batch.size(0), device=x.device)
                batch = torch.cat([batch, additional_indices])

        rembedding = global_mean_pool(x, batch)
        iembedding = self.image_embedding.forward(X_tid)
        conv_block = [rembedding]
        for Conv, max_pooling in zip(self.convs, self.max_poolings):
            act = self.relu(Conv(X_text))
            pool = max_pooling(act)
            pool = torch.squeeze(pool)
            conv_block.append(pool)

        conv_feature = torch.cat(conv_block, dim=1)
        graph_feature, text_feature = conv_feature[:, :300], conv_feature[:, 300:]
        bsz = text_feature.size()[0]

        self_att_t = self.mh_attention(text_feature.view(bsz, -1, 300), text_feature.view(bsz, -1, 300),
                                       text_feature.view(bsz, -1, 300))
        self_att_g = self.mh_attention(graph_feature.view(bsz, -1, 300), graph_feature.view(bsz, -1, 300),
                                       graph_feature.view(bsz, -1, 300))
        self_att_i = self.mh_attention(iembedding.view(bsz, -1, 300), iembedding.view(bsz, -1, 300),
                                       iembedding.view(bsz, -1, 300))
        text_enhanced = self.mh_attention(self_att_i.view((bsz, -1, 300)), self_att_t.view((bsz, -1, 300)),
                                          self_att_t.view((bsz, -1, 300))).view(bsz, 300)
        align_text = self.alignfc_t(text_enhanced)
        align_rembedding = self.alignfc_g(self_att_g)
        dist = [align_text, align_rembedding]

        self_att_t = text_enhanced.view((bsz, -1, 300))
        co_att_tg = self.mh_attention(self_att_t, self_att_g, self_att_g).view(bsz, 300)
        co_att_gt = self.mh_attention(self_att_g, self_att_t, self_att_t).view(bsz, 300)
        co_att_ti = self.mh_attention(self_att_t, self_att_i, self_att_i).view(bsz, 300)
        co_att_it = self.mh_attention(self_att_i, self_att_t, self_att_t).view(bsz, 300)
        co_att_gi = self.mh_attention(self_att_g, self_att_i, self_att_i).view(bsz, 300)
        co_att_ig = self.mh_attention(self_att_i, self_att_g, self_att_g).view(bsz, 300)

        att_feature = torch.cat((co_att_tg, co_att_gt, co_att_ti, co_att_it, co_att_gi, co_att_ig), dim=1)
        a1 = self.relu(self.dropout(self.fc3(att_feature)))
        a1 = self.relu(self.fc4(a1))
        a1 = self.relu(self.fc1(a1))
        d1 = self.dropout(a1)
        output = self.fc2(d1)

        if return_features:
            final_features = a1.detach().cpu().numpy()
            return output, dist, final_features
        else:
            if compute_contrastive_loss:
                contrastive_loss = self.compute_contrastive_loss(align_text, align_rembedding, iembedding)
                return output, dist, contrastive_loss
            else:
                return output, dist

    def compute_contrastive_loss(self, text_embed, graph_embed, image_embed):
        text_embed = text_embed.view(text_embed.size(0), -1)
        graph_embed = graph_embed.view(graph_embed.size(0), -1)
        image_embed = image_embed.view(image_embed.size(0), -1)

        pos_sim = F.cosine_similarity(text_embed, graph_embed, dim=1)
        neg_sim = F.cosine_similarity(text_embed, image_embed, dim=1)

        contrastive_loss = F.relu(neg_sim - pos_sim + 1.0).mean()
        return contrastive_loss

    def ginmcl(self, x_tid, x_text, y, loss, i, total, params, pgd_word):
        self.optimizer.zero_grad()
        logit_defense, dist, contrastive_loss = self.forward(x_tid, x_text, compute_contrastive_loss=True)
        loss_classification = loss(logit_defense, y)
        loss_mse = nn.MSELoss()
        loss_dis = loss_mse(dist[0], dist[1])
        loss_defense = loss_classification + loss_dis + contrastive_loss
        loss_defense.backward()
        K = 3
        pgd_word.backup_grad()

        for t in range(K):
            pgd_word.attack(is_first_attack=(t == 0))
            if t != K - 1:
                self.zero_grad()
            else:
                pgd_word.restore_grad()
            loss_adv, dist = self.forward(x_tid, x_text)
            loss_adv = loss(loss_adv, y)
            loss_adv.backward()
        pgd_word.restore()

        self.optimizer.step()
        corrects = (torch.max(logit_defense, 1)[1].view(y.size()).data == y.data).sum()
        accuracy = 100 * corrects / len(y)
        print(
            'Batch[{}/{}] - loss: {:.6f}  accuracy: {:.4f}%({}/{})'.format(i + 1, total,
                                                                           loss_defense.item(),
                                                                           accuracy,
                                                                           corrects,
                                                                           y.size(0)))


def load_dataset():
    pre = os.path.dirname(os.getcwd()) + '/dataset/weibo/weibo_files'
    X_train_tid, X_train, y_train, word_embeddings, adj = pickle.load(open(pre + "/train.pkl", 'rb'))
    X_dev_tid, X_dev, y_dev = pickle.load(open(pre + "/dev.pkl", 'rb'))
    X_test_tid, X_test, y_test = pickle.load(open(pre + "/test.pkl", 'rb'))
    config['embedding_weights'] = word_embeddings
    config['node_embedding'] = pickle.load(open(pre + "/node_embedding.pkl", 'rb'))[0]
    print("#nodes: ", adj.shape[0])

    with open(pre + '/new_id_dic.json', 'r') as f:
        newid2mid = json.load(f)
        newid2mid = dict(zip(newid2mid.values(), newid2mid.keys()))
    mid2num = {}
    for file in os.listdir(os.path.dirname(os.getcwd()) + '/dataset/weibo/weibocontentwithimage/original-microblog/'):
        mid2num[file.split('_')[-2]] = file.split('_')[0]
    newid2num = {}
    for id in X_train_tid:
        newid2num[id] = mid2num[newid2mid[id]]
    for id in X_dev_tid:
        newid2num[id] = mid2num[newid2mid[id]]
    for id in X_test_tid:
        newid2num[id] = mid2num[newid2mid[id]]
    config['newid2imgnum'] = newid2num
    return X_train_tid, X_train, y_train, \
           X_dev_tid, X_dev, y_dev, \
           X_test_tid, X_test, y_test, adj


def load_original_adj(adj):
    pre = os.path.dirname(os.getcwd()) + '/dataset/weibo/weibo_files/'
    path = os.path.join(pre, 'original_adj')
    with open(path, 'r') as f:
        original_adj_dict = json.load(f)
    original_adj = np.zeros(shape=adj.shape)
    for i, v in original_adj_dict.items():
        v = [int(e) for e in v]
        original_adj[int(i), v] = 1
    return original_adj


def train_and_test(model):
    model_suffix = model.__name__.lower().strip("text")
    res_dir = 'exp_result'
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    res_dir = os.path.join(res_dir, args.task)
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    res_dir = os.path.join(res_dir, args.description)
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    res_dir = config['save_path'] = os.path.join(res_dir, 'best_model_in_each_config')
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    config['save_path'] = os.path.join(res_dir, args.thread_name + '_' + 'config' + args.config_name.split(".")[
        0] + '_best_model_weights_' + model_suffix)

    dir_path = os.path.join('exp_result', args.task, args.description)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    if os.path.exists(config['save_path']):
        os.system('rm {}'.format(config['save_path']))

    X_train_tid, X_train, y_train, \
    X_dev_tid, X_dev, y_dev, \
    X_test_tid, X_test, y_test, adj = load_dataset()
    original_adj = load_original_adj(adj)
    nn = model(config, adj, original_adj)

    nn.fit(X_train_tid, X_train, y_train,
           X_dev_tid, X_dev, y_dev)

    if os.path.exists('D:\DL\GINMCL\graph_part\exp_result\weibo2\exp_description\best_model_in_each_config\Thread-1_configsingle3_best_model_weights_ginmcl'):
        model.load_state_dict(torch.load('D:\DL\GINMCL\graph_part\exp_result\weibo2\exp_description\best_model_in_each_config\Thread-1_configsingle3_best_model_weights_ginmcl'))
        model.eval()

        with torch.no_grad():
            X_test_tid = torch.tensor(X_test_tid, dtype=torch.long, device=model.device)
            X_test = torch.tensor(X_test, dtype=torch.long, device=model.device)
            output, dist, test_features = model.forward(X_test_tid, X_test, return_features=True)

        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(test_features)

        plt.figure(figsize=(10, 8))
        for i in np.unique(y_test):
            indices = np.where(y_test == i)
            plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label=f'Class {i}')

        plt.title("t-SNE Visualization of Test Features")
        plt.xlabel('t-SNE Feature 1')
        plt.ylabel('t-SNE Feature 2')
        plt.legend()
        plt.show()


config = process_config(config_file.config)
seed = config['seed']
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
model = GINMCL
train_and_test(model)
