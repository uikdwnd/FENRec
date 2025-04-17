# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils import recall_at_k, ndcg_k, get_metric
from models import KMeans, WeightedKMeans


class Trainer:
    def __init__(self, model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        self.model = model
        self.dot_max = None

        self.batch_size = self.args.batch_size
        self.sim = self.args.sim

        cluster = KMeans(
            num_cluster=args.intent_num,
            seed=1,
            hidden_size=64,
            gpu_id=args.gpu_id,
            device=torch.device("cuda"),
        )
        self.clusters = [cluster]
        self.clusters_t = [self.clusters]

        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.cluster_dataloader = cluster_dataloader

        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        self.optim = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader, self.cluster_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort=full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort=full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": "{:.4f}".format(HIT_1),
            "NDCG@1": "{:.4f}".format(NDCG_1),
            "HIT@5": "{:.4f}".format(HIT_5),
            "NDCG@5": "{:.4f}".format(NDCG_5),
            "HIT@10": "{:.4f}".format(HIT_10),
            "NDCG@10": "{:.4f}".format(NDCG_10),
            "MRR": "{:.4f}".format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": "{:.4f}".format(recall[0]),
            "NDCG@5": "{:.4f}".format(ndcg[0]),
            "HIT@10": "{:.4f}".format(recall[1]),
            "NDCG@10": "{:.4f}".format(ndcg[1]),
            "HIT@20": "{:.4f}".format(recall[3]),
            "NDCG@20": "{:.4f}".format(ndcg[3]),
        }
        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def save_all_state(self, epoch, file_name, dataloader_name):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.cpu().state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
        }, file_name)
        torch.save(self.train_dataloader, dataloader_name)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def load_all_state(self, file_name, dataloader_name):
        checkpoint = torch.load(file_name)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_dataloader = torch.load(dataloader_name)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=torch.bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    # False Negative Mask
    def mask_correlated_samples_(self, label):
        label = label.view(1, -1)
        label = label.expand((2, label.shape[-1])).reshape(1, -1)  # 重複兩次
        label = label.contiguous().view(-1, 1)
        mask = torch.eq(label, label.t())  # 看label是否相等
        return mask == 0  # True的地方要留下

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot', intent_id=None):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)
        if sim == 'cos':
            sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.t()) / temp

        sim_i_j = torch.diag(sim, batch_size)  # i*j
        sim_j_i = torch.diag(sim, -batch_size)  # j*i

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        if self.args.f_neg:
            mask = self.mask_correlated_samples_(intent_id)
            negative_samples = sim
            negative_samples[mask == 0] = float("-inf")
        else:
            mask = self.mask_correlated_samples(batch_size)
            negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def focal_info_nce(self, z_i, z_j, temp, batch_size, sim='dot', intent_id=None):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size
        m = self.args.focal_m
        z = torch.cat((z_i, z_j), dim=0)
        if sim == 'cos':
            sim1 = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
            sim = sim1 / temp
        elif sim == 'dot':
            sim1 = torch.mm(z, z.t())
            # if self.dot_max == None or sim1.max() > self.dot_max:
            #     dot_max = sim1.max()
            sim = sim1 / temp
            # sim1 = sim1 / torch.norm(z, dim=-1).unsqueeze(0)
            sim1 = torch.tanh(sim1 / self.args.tau)  # torch.sigmoid(sim1 / 3) # / min(dot_max, 1)

        sim_i_j = torch.diag(sim, batch_size) * torch.diag(sim1, batch_size)  # i*j
        sim_j_i = torch.diag(sim, -batch_size) * torch.diag(sim1, -batch_size)  # j*i

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        if self.args.f_neg:
            mask = self.mask_correlated_samples_(intent_id)
            negative_samples = sim * (sim1 + m)
            negative_samples[mask == 0] = float("-inf")
        else:
            mask = self.mask_correlated_samples(batch_size)
            samples = sim * (sim1 + m)
            negative_samples = samples[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def predict_full(self, seq_out):
        test_item_emb = self.model.item_embeddings.weight
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

    def cicl_loss(self, coarse_intents, target_item):
        coarse_intent_1, coarse_intent_2 = coarse_intents[0], coarse_intents[1]
        sem_nce_logits, sem_nce_labels = self.info_nce(coarse_intent_1[:, -1, :], coarse_intent_2[:, -1, :],
                                                       self.args.temperature, coarse_intent_1.shape[0], self.sim,
                                                       target_item[:, -1])
        cicl_loss = nn.CrossEntropyLoss()(sem_nce_logits, sem_nce_labels)
        return cicl_loss

    def ficl_loss(self, sequences, clusters_t):
        output = sequences[0][:, -1, :]
        intent_n = output.view(-1, output.shape[-1])  # [BxH]
        intent_n = intent_n.detach().cpu().numpy()
        intent_id, seq_to_v = clusters_t[0].query(intent_n)

        seq_to_v = seq_to_v.view(seq_to_v.shape[0], -1)  # [BxH]
        a, b = self.info_nce(output.view(output.shape[0], -1), seq_to_v, self.args.temperature, output.shape[0],
                             sim=self.sim, intent_id=intent_id)
        loss_n_0 = nn.CrossEntropyLoss()(a, b)

        output_s = sequences[1][:, -1, :]
        intent_n = output_s.view(-1, output_s.shape[-1])
        intent_n = intent_n.detach().cpu().numpy()
        intent_id, seq_to_v_1 = clusters_t[0].query(intent_n)  # [BxH]
        seq_to_v_1 = seq_to_v_1.view(seq_to_v_1.shape[0], -1)  # [BxH]
        a, b = self.info_nce(output_s.view(output_s.shape[0], -1), seq_to_v_1, self.args.temperature, output_s.shape[0],
                             sim=self.sim, intent_id=intent_id)
        loss_n_1 = nn.CrossEntropyLoss()(a, b)
        ficl_loss = loss_n_0 + loss_n_1

        return ficl_loss


class FENRecTrainer(Trainer):
    def __init__(self, model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args):
        super(FENRecTrainer, self).__init__(
            model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args
        )
        self.warm_up_epoch = 20

    def _softmax(self, x, weight, weight_len):
        """
        :param x: all logits
        :param weight: The weight of each part of logit (list)
        :param weight_len: The length of each part of logit (list)
        :return:
        """

        assert len(weight) == len(weight_len)
        #########Check if the "total length" matches x.shape[1]###############
        total_len = 0
        for l in weight_len:
            total_len += l
        assert x.shape[1] == total_len
        ##################################################
        # To prevent numerical overflow, first apply stabilization to the input.
        exp_x = torch.exp(x - torch.max(x, dim=1, keepdim=True)[0])  # exp_x.shape = x.shape

        ##########Apply weighting to exp_x#######################
        weighted_exp_x_ls = []
        start = 0
        for w, wl in zip(weight, weight_len):
            weighted_exp_x_ls.append(w * exp_x[:, start: start + wl])
            start += wl
        ###################################################
        exp_x = torch.cat(weighted_exp_x_ls, dim=1)
        assert exp_x.shape[1] == x.shape[1]
        return exp_x / torch.sum(exp_x, dim=1, keepdim=True)

    def _cross_entropy_loss(self, outputs, targets, weight, weight_len):
        # Use your own implemented softmax function to convert the output into probabilities.
        probs = self._softmax(outputs, weight, weight_len)

        # Use the gather function to extract the probabilities of the target classes from probs
        batch_size = outputs.size(0)
        target_probs = probs[range(batch_size), targets]

        # Calculate the cross-entropy loss
        loss = -torch.log(target_probs).mean()

        return loss

    def get_noise_negative_mask(self, label):
        label = label.reshape(1, -1)
        label = label.contiguous().view(-1, 1)
        fn_mask = torch.eq(label, label.t())  # Check if the labels are equal. fn_mask == 0 indicates the positions to keep as True.
        diagonal_mask = torch.eye(fn_mask.shape[0], dtype=torch.bool).to(fn_mask.device)  # The diagonal is set to True
        return (fn_mask == 0) | diagonal_mask

    def _get_enduring_hard_negatives(self, neg_seqs, pos_seqs, min_mix_ratio, max_mix_ratio, epoch):
        """
        :param neg_seqs: shape = (2n, d)  -->  (1, 2n, d).repeat(2n, 1, 1)
        :param pos_seqs: shape = (2n, d)  -->  (2n, 1, d)
        :return: shape = (2n, 2n, d) 2n positives, each with 2n negatives.
        """
        warm_up_epochs = self.warm_up_epoch

        max_lda = max_mix_ratio
        min_lda = max(min(self.args.mix_ratio_rate * (epoch + 1 - warm_up_epochs) + min_mix_ratio, max_mix_ratio),
                      min_mix_ratio)

        normalized_neg_seqs = neg_seqs / torch.norm(neg_seqs, dim=-1).unsqueeze(-1)
        normalized_pos_seqs = pos_seqs / torch.norm(pos_seqs, dim=-1).unsqueeze(-1)
        normalized_neg_seqs = normalized_neg_seqs.unsqueeze(0).repeat(normalized_pos_seqs.shape[0], 1, 1)
        normalized_pos_seqs = normalized_pos_seqs.unsqueeze(1)
        lda = (torch.rand(normalized_neg_seqs.shape[:-1]) * (max_lda - min_lda) + min_lda).unsqueeze(-1).to(
            pos_seqs.device)
        hard_negative = normalized_neg_seqs * (1 - lda) + normalized_pos_seqs * lda
        hard_negative = hard_negative / torch.norm(hard_negative, dim=-1).unsqueeze(-1)


        hard_negative = hard_negative * torch.norm(neg_seqs.unsqueeze(0), dim=-1).unsqueeze(-1).repeat(
            hard_negative.shape[0], 1, 1)

        ###########################################################################
        hard_negative = hard_negative.detach()
        return hard_negative

    def enduring_hard_negatives_contrastive_loss(self, z_i, z_j, temp, epoch, batch_size, sim='dot', intent_id=None,
                                     min_mix_ratio=0.2, max_mix_ratio=0.2, z_negative=None):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        m = self.args.focal_m

        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)  # shape = (2n, d)
        hard_negative = self._get_enduring_hard_negatives(z, z, min_mix_ratio, max_mix_ratio,
                                                          epoch)  # shape = (2n, 2n, d)
        # hard_negative = z.unsqueeze(0).repeat(z.shape[0], 1, 1).detach()
        if sim == 'cos':
            sim1 = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
            sim = sim1 / temp
            neg_sim1 = F.cosine_similarity(z.unsqueeze(1), hard_negative, dim=2)
            neg_sim = neg_sim1 / temp
            noise_neg_sim1 = F.cosine_similarity(z.unsqueeze(1), z_negative.unsqueeze(0), dim=2)
            noise_neg_sim = noise_neg_sim1 / temp
        elif sim == 'dot':
            sim1 = torch.mm(z, z.t())
            sim = sim1 / temp

            neg_sim1 = torch.bmm(z.unsqueeze(1), hard_negative.transpose(1, 2)).squeeze(
                1)  # shape (2n, 1, 2n) --> (2n, 2n)
            neg_sim = neg_sim1 / temp

            noise_neg_sim1 = torch.mm(z, z_negative.t())
            noise_neg_sim = noise_neg_sim1 / temp

            sim1 = torch.tanh(sim1 / self.args.tau)  # torch.sigmoid(sim1 / 3)
            neg_sim1 = torch.tanh(neg_sim1 / self.args.tau)  # torch.sigmoid(neg_sim1 / 3)
            noise_neg_sim1 = torch.tanh(noise_neg_sim1 / self.args.tau)

        sim_i_j = torch.diag(sim, batch_size) * torch.diag(sim1, batch_size)  # i*j
        sim_j_i = torch.diag(sim, -batch_size) * torch.diag(sim1, -batch_size)  # j*i

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        if self.args.f_neg:
            mask = self.mask_correlated_samples_(intent_id)
            negative_samples = sim * (sim1 + m)
            hard_negative_samples = neg_sim * (neg_sim1 + m)
            noise_negative_samples = noise_neg_sim * (noise_neg_sim1 + m)

            negative_samples[mask == 0] = float("-inf")
            hard_negative_samples[mask == 0] = float("-inf")
            weight_len = [negative_samples.shape[1], noise_negative_samples.shape[1], hard_negative_samples.shape[1]]
            negative_samples = torch.cat((negative_samples, noise_negative_samples, hard_negative_samples), dim=1)
        else:
            print("ERROR")
            exit(1)
            mask = self.mask_correlated_samples(batch_size)
            negative_samples = sim[mask].reshape(N, -1) * (sim1[mask].reshape(N, -1) + m)
            hard_negative_samples = neg_sim[mask].reshape(N, -1) * (neg_sim1[mask].reshape(N, -1) + m)
            negative_samples = torch.cat((negative_samples, hard_negative_samples), dim=1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        weight_len[0] += positive_samples.shape[1]
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels, weight_len

    def cicl_loss(self, coarse_intents, target_item, epoch, z_negative=None):
        warm_up_epochs = self.warm_up_epoch
        min_mix_ratio = self.args.min_mix_ratio
        max_mix_ratio = self.args.max_mix_ratio
        rate = self.args.mix_portion
        max_weight = self.args.mix_portion

        coarse_intent_1, coarse_intent_2 = coarse_intents[0], coarse_intents[1]
        if epoch >= warm_up_epochs:  # True
            noise_negative_weight = 1.
            sem_nce_logits, sem_nce_labels, weight_len = self.enduring_hard_negatives_contrastive_loss(
                coarse_intent_1[:, -1, :], coarse_intent_2[:, -1, :], self.args.temperature, epoch,
                coarse_intent_1.shape[0], self.sim, target_item[:, -1], min_mix_ratio=min_mix_ratio,
                max_mix_ratio=max_mix_ratio, z_negative=z_negative)
            weight = min(rate * (epoch + 1 - warm_up_epochs), max_weight)  # 0.002#1 / 512.
            cicl_loss = self._cross_entropy_loss(sem_nce_logits, sem_nce_labels, [1, noise_negative_weight, weight], weight_len)
        else:
            sem_nce_logits, sem_nce_labels = self.info_nce(coarse_intent_1[:, -1, :], coarse_intent_2[:, -1, :],
                                                           self.args.temperature, coarse_intent_1.shape[0], self.sim,
                                                           target_item[:, -1])
            cicl_loss = nn.CrossEntropyLoss()(sem_nce_logits, sem_nce_labels)
        return cicl_loss

    def ficl_loss(self, sequences, clusters_t, epoch, z_negative=None):
        warm_up_epochs = self.warm_up_epoch
        min_mix_ratio = self.args.min_mix_ratio
        max_mix_ratio = self.args.max_mix_ratio
        rate = self.args.mix_portion
        max_weight = self.args.mix_portion

        output = sequences[0][:, -1, :]
        intent_n = output.view(-1, output.shape[-1])  # [BxH]
        intent_n = intent_n.detach().cpu().numpy()
        intent_id, seq_to_v = clusters_t[0].query(intent_n)
        seq_to_v = seq_to_v.view(seq_to_v.shape[0], -1)  # [BxH]
        if epoch >= warm_up_epochs:  # False
            noise_negative_weight = 1.
            a, b, weight_len = self.enduring_hard_negatives_contrastive_loss(output.view(output.shape[0], -1), seq_to_v,
                                                                             self.args.temperature, epoch,
                                                                             output.shape[0], sim=self.sim,
                                                                             intent_id=intent_id,
                                                                             min_mix_ratio=min_mix_ratio,
                                                                             max_mix_ratio=max_mix_ratio,
                                                                             z_negative=z_negative)
            weight = min(rate * (epoch + 1 - warm_up_epochs), max_weight)  # 0.002#1 / 512.
            loss_n_0 = self._cross_entropy_loss(a, b, [1, noise_negative_weight, weight], weight_len)
        else:
            a, b = self.info_nce(output.view(output.shape[0], -1), seq_to_v, self.args.temperature, output.shape[0],
                                 sim=self.sim, intent_id=intent_id)
            loss_n_0 = nn.CrossEntropyLoss()(a, b)

        output_s = sequences[1][:, -1, :]
        intent_n = output_s.view(-1, output_s.shape[-1])
        intent_n = intent_n.detach().cpu().numpy()
        intent_id, seq_to_v_1 = clusters_t[0].query(intent_n)  # [BxH]
        seq_to_v_1 = seq_to_v_1.view(seq_to_v_1.shape[0], -1)  # [BxH]
        if epoch >= warm_up_epochs:  # False
            noise_negative_weight = 1.
            a, b, weight_len = self.enduring_hard_negatives_contrastive_loss(output_s.view(output_s.shape[0], -1),
                                                                             seq_to_v_1, self.args.temperature, epoch,
                                                                             output_s.shape[0], sim=self.sim,
                                                                             intent_id=intent_id,
                                                                             min_mix_ratio=min_mix_ratio,
                                                                             max_mix_ratio=max_mix_ratio,
                                                                             z_negative=z_negative)
            weight = min(rate * (epoch + 1 - warm_up_epochs), max_weight)  # 0.002#1 / 512.
            loss_n_1 = self._cross_entropy_loss(a, b, [1, noise_negative_weight, weight], weight_len)
        else:
            a, b = self.info_nce(output_s.view(output_s.shape[0], -1), seq_to_v_1, self.args.temperature,
                                 output_s.shape[0],
                                 sim=self.sim, intent_id=intent_id)
            loss_n_1 = nn.CrossEntropyLoss()(a, b)
        ficl_loss = loss_n_0 + loss_n_1
        return ficl_loss

    def iteration(self, epoch, dataloader, cluster_dataloader=None, full_sort=True, train=True):
        str_code = "train" if train else "test"
        if train:
            if self.args.cl_mode in ['cf', 'f']:
                # ------ intentions clustering ----- #
                print("Preparing Clustering:")
                self.model.eval()
                # save N
                kmeans_training_data = []
                rec_cf_data_iter = tqdm(enumerate(cluster_dataloader), total=len(cluster_dataloader))

                for i, (rec_batch) in rec_cf_data_iter:
                    """
                    rec_batch shape: key_name x batch_size x feature_dim
                    cl_batches shape: 
                        list of n_views x batch_size x feature_dim tensors
                    """
                    # 0. batch_data will be sent into the device(GPU or CPU)
                    rec_batch = tuple(t.to(self.device) for t in rec_batch)
                    _, subsequence, _, _, _ = rec_batch
                    sequence_output_a = self.model(subsequence)  # [BxLxH]
                    sequence_output_b = sequence_output_a[:, -1, :]  # [BxH]
                    kmeans_training_data.append(sequence_output_b.detach().cpu().numpy())

                kmeans_training_data = np.concatenate(kmeans_training_data, axis=0)
                kmeans_training_data_t = [kmeans_training_data]

                for i, clusters in tqdm(enumerate(self.clusters_t), total=len(self.clusters_t)):
                    for j, cluster in enumerate(clusters):
                        cluster.train(kmeans_training_data_t[i])
                        self.clusters_t[i][j] = cluster

                # clean memory
                del kmeans_training_data
                del kmeans_training_data_t
                import gc
                gc.collect()

            # ------ model training -----#
            print("Performing Rec model Training:")
            self.model.train()
            rec_avg_loss = 0.0
            joint_avg_loss = 0.0
            icl_losses = 0.0

            print(f"rec dataset length: {len(dataloader)}")
            rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            for i, (rec_batch) in rec_cf_data_iter:
                """             
                rec_batch shape: key_name x batch_size x feature_dim
                """
                # 0. batch_data will be sent into the device(GPU or CPU)
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                seq_id, subsequence_1, target_pos_1, subsequence_2, _ = rec_batch

                # ---------- prediction task -----------------------#
                intent_output = self.model(subsequence_1)
                logits = self.predict_full(intent_output[:, -1, :])  # [Bx|I|]

                #-----------time-dependent soft labeling---------------#
                rec_label = torch.tensor(
                    dataloader.dataset.label_matrix[seq_id[:, 0].detach().cpu().numpy()].toarray()).to(logits.device)
                rec_loss = nn.CrossEntropyLoss()(logits, rec_label)
                del rec_label

                # ---------- intent representation learning task ---------------#
                coarse_intent_1 = self.model(subsequence_1)
                coarse_intent_2 = self.model(subsequence_2)

                if epoch >= self.warm_up_epoch:
                    ## ----------------noise-based negative-------------------------#
                    z_negative = torch.randn([int(coarse_intent_1.shape[0] * self.args.noise_negative_portion), coarse_intent_1.shape[-1]],
                                             device=coarse_intent_1.device)  # * variation + avg
                    z_negative.requires_grad = True
                    intent_sim = torch.mm(coarse_intent_1[:, -1, :], coarse_intent_2[:, -1, :].t()).detach()
                    noise_negative_mask = self.get_noise_negative_mask(target_pos_1[:, -1])
                    intent_sim[noise_negative_mask == 0] = float("-inf")
                    labels = torch.arange(intent_sim.size(0)).long().to(intent_sim.device)
                    loss_fct = nn.CrossEntropyLoss()
                    for _ in range(4):
                        sim_negative = torch.mm(coarse_intent_1[:, -1, :].detach(), z_negative.t())
                        sim_fused = torch.cat([intent_sim, sim_negative], 1)

                        loss = loss_fct(sim_fused, labels)

                        noise_grad = torch.autograd.grad(loss, z_negative, retain_graph=True)[0]

                        z_negative = z_negative + (noise_grad / torch.norm(noise_grad, dim=-1, keepdim=True)).mul_(1e-3)
                        z_negative = torch.where(torch.isnan(z_negative), torch.zeros_like(z_negative), z_negative)
                else:
                    z_negative = None

                if self.args.cl_mode in ['c', 'cf']:
                    cicl_loss = self.cicl_loss([coarse_intent_1, coarse_intent_2], target_pos_1, epoch, z_negative)
                else:
                    cicl_loss = 0.0
                if self.args.cl_mode in ['f', 'cf']:
                    ficl_loss = self.ficl_loss([coarse_intent_1, coarse_intent_2], self.clusters_t[0], epoch,
                                               z_negative)
                else:
                    ficl_loss = 0.0
                icl_loss = self.args.lambda_0 * cicl_loss + self.args.beta_0 * ficl_loss

                # ---------- multi-task learning --------------------#
                joint_loss = self.args.rec_weight * rec_loss + icl_loss

                self.optim.zero_grad()
                joint_loss.backward()
                self.optim.step()

                rec_avg_loss += rec_loss.item()
                if type(icl_loss) != float:
                    icl_losses += icl_loss.item()
                else:
                    icl_losses += icl_loss
                joint_avg_loss += joint_loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": "{:.4f}".format(rec_avg_loss / len(rec_cf_data_iter)),
                "icl_avg_loss": "{:.4f}".format(icl_losses / len(rec_cf_data_iter)),
                "joint_avg_loss": "{:.4f}".format(joint_avg_loss / len(rec_cf_data_iter)),
            }
            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, "a") as f:
                f.write(str(post_fix) + "\n")

        else:
            rec_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, answers = batch
                    recommend_output = self.model(input_ids)  # [BxLxH]
                    recommend_output = recommend_output[:, -1, :]  # [BxH]
                    # recommendation results
                    rating_pred = self.predict_full(recommend_output)
                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # argpartition T: O(n)  argsort O(nlogn)
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                for i, batch in rec_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)

                return self.get_sample_scores(epoch, pred_list)
