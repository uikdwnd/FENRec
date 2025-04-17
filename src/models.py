# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import math
import os
import pickle

import numpy as np
from tqdm import tqdm
import random
import copy

import torch
import torch.nn as nn
import gensim
import faiss
import time
from modules import Encoder, LayerNorm
import torch.nn.functional as F


class KMeans(object):
    def __init__(self, num_cluster, seed, hidden_size, gpu_id=0, device="cpu"):
        """
        Args:
            k: number of clusters
        """
        self.seed = seed
        self.num_cluster = num_cluster
        self.max_points_per_centroid = 4096
        self.min_points_per_centroid = 0
        self.gpu_id = 0
        self.device = device
        self.first_batch = True
        self.hidden_size = hidden_size
        self.clus, self.index = self.__init_cluster(self.hidden_size)
        self.centroids = []

    def __init_cluster(
        self, hidden_size, verbose=False, niter=20, nredo=5, max_points_per_centroid=4096, min_points_per_centroid=0
    ):
        print(" cluster train iterations:", niter)
        clus = faiss.Clustering(hidden_size, self.num_cluster)
        clus.verbose = verbose
        clus.niter = niter
        clus.nredo = nredo
        clus.seed = self.seed
        clus.max_points_per_centroid = max_points_per_centroid
        clus.min_points_per_centroid = min_points_per_centroid

        res = faiss.StandardGpuResources()
        res.noTempMemory()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = self.gpu_id
        index = faiss.GpuIndexFlatL2(res, hidden_size, cfg)
        return clus, index

    def train(self, x):
        # train to get centroids
        if x.shape[0] > self.num_cluster:
            self.clus.train(x, self.index)
        # get cluster centroids
        centroids = faiss.vector_to_array(self.clus.centroids).reshape(self.num_cluster, self.hidden_size)
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).to(self.device)
        self.centroids = nn.functional.normalize(centroids, p=2, dim=1)

    def train_and_returnI(self, x):
        # train to get centroids
        if x.shape[0] > self.num_cluster:
            self.clus.train(x, self.index)
        # get cluster centroids
        centroids = faiss.vector_to_array(self.clus.centroids).reshape(self.num_cluster, self.hidden_size)
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).to(self.device)
        self.centroids = nn.functional.normalize(centroids, p=2, dim=1)
        D, I = self.index.search(x, 2)
        return I[:, 0], I[:, 1], self.centroids.detach().cpu().numpy()

    def query(self, x):
        # self.index.add(x)
        D, I = self.index.search(x, 1)  # for each sample, find cluster distance and assignments
        seq2cluster = [int(n[0]) for n in I]
        # print("cluster number:", self.num_cluster,"cluster in batch:", len(set(seq2cluster)))
        seq2cluster = torch.LongTensor(seq2cluster).to(self.device)
        return seq2cluster, self.centroids[seq2cluster]

    def my_query(self, x, T=0.1):
        def softmax(x):
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        # self.index.add(x)
        D, I = self.index.search(x, self.num_cluster)  # for each sample, find cluster distance and assignments
        _, indices = self.index.search(x, 1)
        # D = softmax(-D / math.sqrt(512))
        # 创建一个全零的目标数组，用于存放结果
        output = np.zeros((I.shape[0], self.num_cluster), dtype=D.dtype)

        # 遍历每一行，将 value 中的值根据 index 中的列索引填入 output
        for i in range(output.shape[0]):
            output[i, I[i]] = D[i]
        output = softmax(-output / (math.sqrt(self.num_cluster) * T))
        return torch.tensor(output), indices

    def query_multiple_intent(self, x, k=3):
        def softmax(x):
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        # self.index.add(x)
        D, I = self.index.search(x, k)  # for each sample, find cluster distance and assignments
        _, indices = self.index.search(x, 1)
        seq2cluster = [int(n[0]) for n in indices]
        # D = softmax(-D / math.sqrt(512))
        # 创建一个全零的目标数组，用于存放结果
        output = np.ones((I.shape[0], self.num_cluster), dtype=D.dtype) * 1e9

        # 遍历每一行，将 value 中的值根据 index 中的列索引填入 output
        for i in range(output.shape[0]):
            output[i, I[i]] = D[i]
        output = softmax(-output)
        output = torch.tensor(output).to(self.centroids.device)
        centers = torch.matmul(output, self.centroids)
        seq2cluster = torch.LongTensor(seq2cluster).to(self.device)
        return seq2cluster, centers

    def get_intensity_cossim(self, x, T1=0.1, T2=1.1):
        def softmax(x):
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        def create_one_hot_matrix(ind, num_classes):
            eye_matrix = torch.eye(num_classes)  # 创建单位矩阵
            # 使用索引操作将单位矩阵的每行按照 ind 张量的值进行索引
            # 最终得到的 one_hot_matrix 将满足要求
            one_hot_matrix = eye_matrix[ind]

            return one_hot_matrix

        # self.index.add(x)
        D, I = self.index.search(x, self.num_cluster)  # for each sample, find cluster distance and assignments
        _, indices = self.index.search(x, 1)
        # D = softmax(-D / math.sqrt(512))
        # 创建一个全零的目标数组，用于存放结果
        output = np.zeros((I.shape[0], self.num_cluster), dtype=D.dtype)

        # 遍历每一行，将 value 中的值根据 index 中的列索引填入 output
        for i in range(output.shape[0]):
            output[i, I[i]] = D[i]
        output = softmax(-output / (math.sqrt(self.num_cluster) * T1))
        output = torch.tensor(output)
        true_distribution = create_one_hot_matrix(indices.reshape(-1), self.num_cluster)
        # penalty = torch.exp(F.cosine_similarity(output, true_distribution, dim=-1) * 2 - 1)
        penalty = F.cosine_similarity(output, true_distribution, dim=-1) * T2
        # all_distirbution = torch.cat([output, true_distribution], dim=0)
        # all_penalty = 1 - F.cosine_similarity(all_distirbution.unsqueeze(1), all_distirbution.unsqueeze(0), dim=2)

        return penalty #, all_penalty

    def get_intensity_jsd(self, x, T1=0.1, T2=1.15):
        class JSD(nn.Module):
            def __init__(self):
                super(JSD, self).__init__()
                self.kl = nn.KLDivLoss(reduction='none', log_target=True)

            def forward(self, p: torch.tensor, q: torch.tensor):
                epsilon = 1e-10
                p = np.maximum(p, epsilon)
                q = np.maximum(q, epsilon)
                p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
                m = (0.5 * (p + q)).log()
                return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))

        def softmax(x):
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        def create_one_hot_matrix(ind, num_classes):
            """
            輸入一個tensor ind, shape=(n,)
            可以得到一個output, output[i][ind[i]] = 1 其他地方=0
            :param ind:
            :param num_classes:
            :return:
            """
            eye_matrix = torch.eye(num_classes)  # 创建单位矩阵
            # 使用索引操作将单位矩阵的每行按照 ind 张量的值进行索引
            # 最终得到的 one_hot_matrix 将满足要求
            one_hot_matrix = eye_matrix[ind]

            return one_hot_matrix

        # self.index.add(x)
        D, I = self.index.search(x, self.num_cluster)  # for each sample, find "all" cluster distance and assignments
        _, indices = self.index.search(x, 1)  # 最近的cluster的idx
        # 創建一個全零的目標數组，用於存放结果
        output = np.zeros((I.shape[0], self.num_cluster), dtype=D.dtype)  #

        # 遍歷每一行，將 distance 根據 index 中的列索引填入 output，output[i, j]: "第i個representation"和"第j個cluster中心"的距離
        for i in range(output.shape[0]):
            output[i, I[i]] = D[i]
        output = softmax(-output)  # / (math.sqrt(self.num_cluster) * T1) distance越大，表示越不相似，這邊用來表示intent distribution
        output = torch.tensor(output)
        true_distribution = create_one_hot_matrix(indices.reshape(-1), self.num_cluster)

        ########################################################################################33
        distribution_distance = JSD()(output, true_distribution).sum(-1) * (1 / math.log(2))
        penalty = (1 - distribution_distance) * T2

        # all_distirbution = torch.cat([output, true_distribution], dim=0)
        # all_penalty = JSD()(all_distirbution.unsqueeze(1).repeat(1, all_distirbution.shape[0], 1),
        #                     all_distirbution.unsqueeze(0).repeat(all_distirbution.shape[0], 1, 1)).sum(-1).reshape(
        #     all_distirbution.shape[0], all_distirbution.shape[0]) * (1 / math.log(2))
        ##########################################################################################3
        # penalty = F.cosine_similarity(output, true_distribution, dim=-1) * T2
        # all_distirbution = torch.cat([output, true_distribution], dim=0)
        # all_penalty = 1 - F.cosine_similarity(all_distirbution.unsqueeze(1), all_distirbution.unsqueeze(0), dim=2)

        return penalty  # , all_penalty


class WeightedKMeans(object):
    def __init__(self, num_cluster, seed, hidden_size, gpu_id=0, device="cpu"):
        """
        Args:
            k: number of clusters
        """
        self.seed = seed
        self.num_cluster = num_cluster
        self.max_points_per_centroid = 4096
        self.min_points_per_centroid = 0
        self.gpu_id = 0
        self.device = device
        self.first_batch = True
        self.hidden_size = hidden_size
        self.clus, self.index = self.__init_cluster(self.hidden_size)
        self.centroids = []

    def __init_cluster(
        self, hidden_size, verbose=False, niter=20, nredo=5, max_points_per_centroid=4096, min_points_per_centroid=0
    ):
        print(" cluster train iterations:", niter)
        clus = faiss.Clustering(hidden_size, self.num_cluster)
        clus.verbose = verbose
        clus.niter = niter
        clus.nredo = nredo
        clus.seed = self.seed
        clus.max_points_per_centroid = max_points_per_centroid
        clus.min_points_per_centroid = min_points_per_centroid

        res = faiss.StandardGpuResources()
        res.noTempMemory()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = self.gpu_id
        index = faiss.GpuIndexFlatL2(res, hidden_size, cfg)
        return clus, index

    def train(self, x, weights):
        # train to get centroids with weights
        if x.shape[0] > self.num_cluster:
            self.clus.train(x, self.index, weights)
        # get cluster centroids
        centroids = faiss.vector_to_array(self.clus.centroids).reshape(self.num_cluster, self.hidden_size)
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).to(self.device)
        self.centroids = nn.functional.normalize(centroids, p=2, dim=1)

    def query(self, x):
        D, I = self.index.search(x, 1)  # for each sample, find cluster distance and assignments
        seq2cluster = [int(n[0]) for n in I]
        seq2cluster = torch.LongTensor(seq2cluster).to(self.device)
        return seq2cluster, self.centroids[seq2cluster]

    def get_intensity(self, x, T1=0.1, T2=1.15):
        class JSD(nn.Module):
            def __init__(self):
                super(JSD, self).__init__()
                self.kl = nn.KLDivLoss(reduction='none', log_target=True)

            def forward(self, p: torch.tensor, q: torch.tensor):
                epsilon = 1e-10
                p = np.maximum(p, epsilon)
                q = np.maximum(q, epsilon)
                p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
                m = (0.5 * (p + q)).log()
                return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))

        def softmax(x):
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        def create_one_hot_matrix(ind, num_classes):
            """
            輸入一個tensor ind, shape=(n,)
            可以得到一個output, output[i][ind[i]] = 1 其他地方=0
            :param ind:
            :param num_classes:
            :return:
            """
            eye_matrix = torch.eye(num_classes)  # 创建单位矩阵
            # 使用索引操作将单位矩阵的每行按照 ind 张量的值进行索引
            # 最终得到的 one_hot_matrix 将满足要求
            one_hot_matrix = eye_matrix[ind]

            return one_hot_matrix

        # self.index.add(x)
        D, I = self.index.search(x, self.num_cluster)  # for each sample, find "all" cluster distance and assignments
        _, indices = self.index.search(x, 1)  # 最近的cluster的idx
        # 創建一個全零的目標數组，用於存放结果
        output = np.zeros((I.shape[0], self.num_cluster), dtype=D.dtype)  #

        # 遍歷每一行，將 distance 根據 index 中的列索引填入 output，output[i, j]: "第i個representation"和"第j個cluster中心"的距離
        for i in range(output.shape[0]):
            output[i, I[i]] = D[i]
        output = softmax(-output)  # / (math.sqrt(self.num_cluster) * T1) distance越大，表示越不相似，這邊用來表示intent distribution
        output = torch.tensor(output)
        true_distribution = create_one_hot_matrix(indices.reshape(-1), self.num_cluster)

        ########################################################################################33
        distribution_distance = JSD()(output, true_distribution).sum(-1) * (1 / math.log(2))
        penalty = (1 - distribution_distance) * T2

        # all_distirbution = torch.cat([output, true_distribution], dim=0)
        # all_penalty = JSD()(all_distirbution.unsqueeze(1).repeat(1, all_distirbution.shape[0], 1),
        #                     all_distirbution.unsqueeze(0).repeat(all_distirbution.shape[0], 1, 1)).sum(-1).reshape(
        #     all_distirbution.shape[0], all_distirbution.shape[0]) * (1 / math.log(2))
        ##########################################################################################3
        # penalty = F.cosine_similarity(output, true_distribution, dim=-1) * T2
        # all_distirbution = torch.cat([output, true_distribution], dim=0)
        # all_penalty = 1 - F.cosine_similarity(all_distirbution.unsqueeze(1), all_distirbution.unsqueeze(0), dim=2)

        return penalty  # , all_penalty



class SASRecModel(nn.Module):
    def __init__(self, args):
        super(SASRecModel, self).__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.init_weights)

    # Positional Embedding
    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)

        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # model same as SASRec
    def forward(self, input_ids):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)

        sequence_output = item_encoded_layers[-1]
        return sequence_output

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


# GRU Encoder
class GRUEncoder(nn.Module):
    r"""GRU4Rec is a model that incorporate RNN for recommendation.

    Note:

        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, args):
        super(GRUEncoder, self).__init__()

        # load parameters info
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.args = args
        self.embedding_size = args.hidden_size #64
        self.hidden_size = args.hidden_size*2  #128
        self.num_layers = args.num_hidden_layers-1 #1
        self.dropout_prob =args.hidden_dropout_prob #0.3

        # define layers and loss
        self.emb_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)


    def forward(self, item_seq):
        item_seq_emb = self.item_embeddings(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        # seq_output = self.gather_indexes(gru_output, item_seq_len - 1)
        seq_output=gru_output
        return seq_output




