# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import random
import torch
import os
import pickle

from torch.utils.data import Dataset
from tqdm import tqdm

from utils import neg_sample, get_user_seqs
import copy
import dgl
import numpy as np
from os import path
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp



class Generate_tag():
    def __init__(self, data_path, data_name, save_path):
        self.path = data_path
        self.data_name = data_name + "_1"
        self.save_path = save_path

    def generate(self):
        data_f = self.path + "/" + self.data_name + ".txt"
        train_dic = {}
        empty_train_dic = {}
        valid_dic = {}
        test_dic = {}
        with open(data_f, "r") as fr:
            data = fr.readlines()
            for d_ in data:
                items = d_.split(' ')
                tag_train = int(items[-3])
                tag_valid = int(items[-2])
                tag_test = int(items[-1])
                train_temp = list(map(int, items[:-3]))
                valid_temp = list(map(int, items[:-2]))
                test_temp = list(map(int, items[:-1]))
                if len(train_temp) > 1:
                    if tag_train not in train_dic:
                        train_dic.setdefault(tag_train, [])
                    train_dic[tag_train].append(train_temp)
                    if tag_valid not in valid_dic:
                        valid_dic.setdefault(tag_valid, [])
                    valid_dic[tag_valid].append(valid_temp)
                    if tag_test not in test_dic:
                        test_dic.setdefault(tag_test, [])
                    test_dic[tag_test].append(test_temp)
                else:
                    if tag_train not in empty_train_dic:
                        empty_train_dic.setdefault(tag_train, [])
                    empty_train_dic[tag_train].append(train_temp)

        for empty_tag in empty_train_dic.keys():
            if empty_tag not in train_dic:
                train_dic[empty_tag] = empty_train_dic[empty_tag]

        total_dic = {"train": train_dic, "valid": valid_dic, "test": test_dic}
        print("Saving data to ", self.save_path)
        with open(self.save_path + "/" + self.data_name + "_t.pkl", "wb") as fw:
            pickle.dump(total_dic, fw)

    def load_dict(self, data_path):
        if not data_path:
            raise ValueError('invalid path')
        elif not os.path.exists(data_path):
            print("The dict not exist, generating...")
            self.generate()
        with open(data_path, 'rb') as read_file:
            data_dict = pickle.load(read_file)
        return data_dict

    def get_data(self, data_path, mode):
        data = self.load_dict(data_path)
        return data[mode]


class Generate_tag_id():
    def __init__(self, data_path, data_name, save_path):
        self.path = data_path
        self.data_name = data_name + "_1"
        self.save_path = save_path

    def generate(self):
        data_f = self.path + "/" + self.data_name + ".txt"
        train_dic = {}
        empty_train_dic = {}
        valid_dic = {}
        test_dic = {}
        seq2id = {}
        id2seq = {}
        id2tag = {}
        user2seqs = {}
        seq2user = {}
        id_count = 0
        with open(data_f, "r") as fr:
            data = fr.readlines()
            for d_ in data:
                items = d_.split(' ')
                tag_train = int(items[-3])
                tag_valid = int(items[-2])
                tag_test = int(items[-1])
                train_temp = list(map(int, items[:-3]))
                valid_temp = list(map(int, items[:-2]))
                test_temp = list(map(int, items[:-1]))

                seq2id[tuple(train_temp)] = id_count
                id2seq[id_count] = train_temp
                id2tag[id_count] = tag_train
                if int(items[0]) not in user2seqs:
                    user2seqs.setdefault(int(items[0]), [])
                user2seqs[int(items[0])].append(id_count)
                seq2user[id_count] = int(items[0])

                if len(train_temp) > 1:
                    if tag_train not in train_dic:
                        train_dic.setdefault(tag_train, [])
                    train_dic[tag_train].append(id_count)
                    if tag_valid not in valid_dic:
                        valid_dic.setdefault(tag_valid, [])
                    valid_dic[tag_valid].append(valid_temp)
                    if tag_test not in test_dic:
                        test_dic.setdefault(tag_test, [])
                    test_dic[tag_test].append(test_temp)
                else:
                    if tag_train not in empty_train_dic:
                        empty_train_dic.setdefault(tag_train, [])
                    empty_train_dic[tag_train].append(id_count)

                id_count += 1

        for empty_tag in empty_train_dic.keys():
            if empty_tag not in train_dic:
                train_dic[empty_tag] = empty_train_dic[empty_tag]

        total_dic = {"train": train_dic, "valid": valid_dic, "test": test_dic,
                     "id2seq": id2seq, "seq2id": seq2id, "id2tag": id2tag, "user2seqs": user2seqs, "seq2user": seq2user}
        print("Saving data to ", self.save_path)
        with open(self.save_path + "/" + self.data_name + "_t_id.pkl", "wb") as fw:
            pickle.dump(total_dic, fw)

    def load_dict(self, data_path):
        if not data_path:
            raise ValueError('invalid path')
        elif not os.path.exists(data_path):
            print("The dict not exist, generating...")
            self.generate()
        with open(data_path, 'rb') as read_file:
            data_dict = pickle.load(read_file)
        return data_dict

    def get_data(self, data_path, mode):
        data = self.load_dict(data_path)
        return data[mode]


class Generate_uum():
    def __init__(self, data_path, data_name, save_path):
        self.path = data_path
        self.data_name = data_name  # ori_file
        self.save_path = save_path

    def generate(self):
        data_f = self.path + "/" + self.data_name + ".txt"

        item_set = set()
        user2items = {}
        item2users = {}
        with open(data_f, "r") as fr:
            data = fr.readlines()
            for d_ in data:
                user, items = d_.strip().split(" ", 1)
                items = items.split(" ")[:-2]  # remove valid & test
                user = int(user)
                items = [int(item) for item in items]
                for i in items:
                    if i not in item2users:
                        item2users.setdefault(i, [])
                    item2users[i].append(user)
                user2items[user] = items
                item_set = item_set | set(items)

        total_dic = {"item2users": item2users, "user2fullseq": user2items}
        print("Saving data to ", self.save_path)
        with open(self.save_path + "/" + self.data_name + "_u.pkl", "wb") as fw:
            pickle.dump(total_dic, fw)

    def load_dict(self, data_path):
        if not data_path:
            raise ValueError('invalid path')
        elif not os.path.exists(data_path):
            print("The dict not exist, generating...")
            self.generate()
        with open(data_path, 'rb') as read_file:
            data_dict = pickle.load(read_file)
        return data_dict

    def get_data(self, data_path, mode):
        data = self.load_dict(data_path)
        return data[mode]


class RecWithContrastiveLearningDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type="train", similarity_model_type="offline"):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

        # create target item sets
        self.sem_tag = Generate_tag(self.args.data_dir, self.args.data_name, self.args.data_dir)
        self.train_tag = self.sem_tag.get_data(self.args.data_dir + "/" + self.args.data_name + "_1_t.pkl", "train")
        self.true_user_id, _, _, _, _, _ = get_user_seqs(args.train_data_file, self.args.train_user_seq_path)

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        # make a deep copy to avoid original sequence be modified
        copied_input_ids = copy.deepcopy(input_ids)
        pad_len = self.max_len - len(copied_input_ids)
        copied_input_ids = [0] * pad_len + copied_input_ids
        copied_input_ids = copied_input_ids[-self.max_len:]

        if type(target_pos) == tuple:
            pad_len_1 = self.max_len - len(target_pos[1])
            target_pos_1 = [0] * pad_len + target_pos[0]
            target_pos_2 = [0] * pad_len_1 + target_pos[1]
            target_pos_1 = target_pos_1[-self.max_len:]
            target_pos_2 = target_pos_2[-self.max_len:]
            assert len(target_pos_1) == self.max_len
            assert len(target_pos_2) == self.max_len
        else:
            target_pos = [0] * pad_len + target_pos
            target_pos = target_pos[-self.max_len:]
            assert len(target_pos) == self.max_len

        assert len(copied_input_ids) == self.max_len
        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]
            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(copied_input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            if type(target_pos) == tuple:  # training
                cur_rec_tensors = (
                    torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                    torch.tensor(copied_input_ids, dtype=torch.long),
                    torch.tensor(target_pos_1, dtype=torch.long),
                    torch.tensor(target_pos_2, dtype=torch.long),
                    torch.tensor(answer, dtype=torch.long),
                )

            else:  # testing
                cur_rec_tensors = (
                    torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                    torch.tensor(copied_input_ids, dtype=torch.long),
                    torch.tensor(target_pos, dtype=torch.long),
                    torch.tensor(answer, dtype=torch.long),
                )
        return cur_rec_tensors

    def _add_noise_interactions(self, items):
        copied_sequence = copy.deepcopy(items)
        insert_nums = max(int(self.args.noise_ratio * len(copied_sequence)), 0)
        if insert_nums == 0:
            return copied_sequence
        insert_idx = random.choices([i for i in range(len(copied_sequence))], k=insert_nums)
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):
            if index in insert_idx:
                item_id = random.randint(1, self.args.item_size - 2)
                while item_id in copied_sequence:
                    item_id = random.randint(1, self.args.item_size - 2)
                inserted_sequence += [item_id]
            inserted_sequence += [item]
        return inserted_sequence

    def __getitem__(self, index):
        user_id = index
        t_user_id = self.true_user_id[index]
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]
        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            temp = self.train_tag[items[-3]]
            flag = False
            for t_ in temp:
                if t_[1:] == items[:-3]:
                    continue
                else:
                    target_pos_ = t_[1:]
                    flag = True
            if not flag:
                target_pos_ = random.choice(temp)[1:]
            seq_label_signal = items[-2]  # no use
            answer = [0]  # no use
        elif self.data_type == "valid":
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]
        else:
            items_with_noise = self._add_noise_interactions(items)
            input_ids = items_with_noise[:-1]
            target_pos = items_with_noise[1:]
            answer = [items_with_noise[-1]]
        if self.data_type == "train":
            target_pos = (target_pos, target_pos_)
            cur_rec_tensors = self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)
            return (cur_rec_tensors)
        elif self.data_type == "valid":
            cur_rec_tensors = self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)
            return cur_rec_tensors
        else:
            cur_rec_tensors = self._data_sample_rec_task(user_id, items_with_noise, input_ids, target_pos, answer)
            return cur_rec_tensors

    def __len__(self):
        """
        consider n_view of a single sequence as one sample
        """
        return len(self.user_seq)


class RecSoftWithContrastiveLearningDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type="train", similarity_model_type="offline"):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

        # create target item sets
        self.sem_tag = Generate_tag_id(self.args.data_dir, self.args.data_name, self.args.data_dir)
        self.uum_dict = Generate_uum(self.args.data_dir, self.args.data_name, self.args.data_dir)
        self.train_tag = self.sem_tag.get_data(self.args.data_dir + "/" + self.args.data_name + "_1_t_id.pkl", "train")
        self.seq2id = self.sem_tag.get_data(self.args.data_dir + "/" + self.args.data_name + "_1_t_id.pkl", "seq2id")
        self.id2tag = self.sem_tag.get_data(self.args.data_dir + "/" + self.args.data_name + "_1_t_id.pkl", "id2tag")
        self.id2seq = self.sem_tag.get_data(self.args.data_dir + "/" + self.args.data_name + "_1_t_id.pkl", "id2seq")
        self.user2seqs = self.sem_tag.get_data(self.args.data_dir + "/" + self.args.data_name + "_1_t_id.pkl",
                                               "user2seqs")
        self.seq2user = self.sem_tag.get_data(self.args.data_dir + "/" + self.args.data_name + "_1_t_id.pkl",
                                              "seq2user")
        self.item2users = self.uum_dict.get_data(self.args.data_dir + "/" + self.args.data_name + "_u.pkl",
                                                 "item2users")  # user2fullseq
        self.user2fullseq = self.uum_dict.get_data(self.args.data_dir + "/" + self.args.data_name + "_u.pkl",
                                                   "user2fullseq")
        self.true_user_id, _, _, _, _, _ = get_user_seqs(args.train_data_file, self.args.train_user_seq_path)
        print("gamma", self.args.soft_label_gamma)
        self._get_soft_label(gamma=self.args.soft_label_gamma)

    def _get_soft_label(self, d=2, gamma=0.2):
        rows = []
        cols = []
        data = []
        for sid, seq in tqdm(self.id2seq.items()):
            uid = self.seq2user[sid]
            target = self.id2tag[sid]
            user_seq = self.user2fullseq[uid]
            # user_seq = seq[1:]
            gt_id = user_seq.index(target)
            # neighbor_gt_items = user_seq[max(gt_id - d, 0): gt_id + d + 1]
            neighbor_gt_items = user_seq[gt_id: gt_id + d + 1]
            gt_id = neighbor_gt_items.index(target)
            indices = np.arange(len(neighbor_gt_items))
            weights = gamma ** np.abs(indices - gt_id)
            weights = weights / weights.sum(-1)

            rows.extend([sid] * len(neighbor_gt_items))
            cols.extend(neighbor_gt_items)
            data.extend(weights.tolist())
        self.label_matrix = sp.csr_matrix((data, (rows, cols)), shape=(max(rows) + 1, max(cols) + 2))

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        # make a deep copy to avoid original sequence be modified
        copied_input_ids = copy.deepcopy(input_ids)
        pad_len = self.max_len - len(copied_input_ids)
        copied_input_ids = [0] * pad_len + copied_input_ids
        copied_input_ids = copied_input_ids[-self.max_len:]

        if type(target_pos) == tuple:
            pad_len_1 = self.max_len - len(target_pos[1])
            target_pos_1 = [0] * pad_len + target_pos[0]
            target_pos_2 = [0] * pad_len_1 + target_pos[1]
            target_pos_1 = target_pos_1[-self.max_len:]
            target_pos_2 = target_pos_2[-self.max_len:]
            assert len(target_pos_1) == self.max_len
            assert len(target_pos_2) == self.max_len
        else:
            target_pos = [0] * pad_len + target_pos
            target_pos = target_pos[-self.max_len:]
            assert len(target_pos) == self.max_len

        assert len(copied_input_ids) == self.max_len
        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]
            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(copied_input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            if type(target_pos) == tuple:  # training
                cur_rec_tensors = (
                    torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                    torch.tensor(copied_input_ids, dtype=torch.long),
                    torch.tensor(target_pos_1, dtype=torch.long),
                    torch.tensor(target_pos_2, dtype=torch.long),
                    torch.tensor(answer, dtype=torch.long),
                )

            else:  # testing
                cur_rec_tensors = (
                    torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                    torch.tensor(copied_input_ids, dtype=torch.long),
                    torch.tensor(target_pos, dtype=torch.long),
                    torch.tensor(answer, dtype=torch.long),
                )
        return cur_rec_tensors

    def _add_noise_interactions(self, items):
        copied_sequence = copy.deepcopy(items)
        insert_nums = max(int(self.args.noise_ratio * len(copied_sequence)), 0)
        if insert_nums == 0:
            return copied_sequence
        insert_idx = random.choices([i for i in range(len(copied_sequence))], k=insert_nums)
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):
            if index in insert_idx:
                item_id = random.randint(1, self.args.item_size - 2)
                while item_id in copied_sequence:
                    item_id = random.randint(1, self.args.item_size - 2)
                inserted_sequence += [item_id]
            inserted_sequence += [item]
        return inserted_sequence

    def __getitem__(self, index):
        user_id = index
        t_user_id = self.true_user_id[index]
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]
        if self.data_type == "train":
            input_ids = items[:-3]
            seq_id = self.seq2id[tuple([t_user_id] + input_ids)]
            target_pos = items[1:-2]
            temp = self.train_tag[items[-3]]
            flag = False
            for t__ in temp:
                t_ = self.id2seq[t__]
                if t_[1:] == items[:-3]:
                    continue
                else:
                    target_pos_ = t_[1:]
                    target_pos_id_ = t__
                    flag = True
            if not flag:
                idx_ = random.choice(range(len(temp)))
                target_pos_id_ = temp[idx_]
                target_pos_ = self.id2seq[target_pos_id_][1:]
            # for t_ in temp:
            #     if t_[1:] == items[:-3]:
            #         continue
            #     else:
            #         pos_user_id = t_[0]
            #         target_pos_ = t_[1:]
            #         flag = True
            # if not flag:
            #     rand_choice = random.choice(temp)
            #     pos_user_id = rand_choice[0]
            #     target_pos_ = rand_choice[1:]

            seq_label_signal = items[-2]  # no use
            answer = [0]  # no use
        elif self.data_type == "valid":
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]
        else:
            items_with_noise = self._add_noise_interactions(items)
            input_ids = items_with_noise[:-1]
            target_pos = items_with_noise[1:]
            answer = [items_with_noise[-1]]
        if self.data_type == "train":
            # t_user_id = (t_user_id, pos_user_id)t_user_id
            seq_ids = (seq_id, target_pos_id_)
            target_pos = (target_pos, target_pos_)
            cur_rec_tensors = self._data_sample_rec_task(seq_ids, items, input_ids, target_pos, answer)
            return (cur_rec_tensors)
        elif self.data_type == "valid":
            cur_rec_tensors = self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)
            return cur_rec_tensors
        else:
            cur_rec_tensors = self._data_sample_rec_task(user_id, items_with_noise, input_ids, target_pos, answer)
            return cur_rec_tensors

    def __len__(self):
        """
        consider n_view of a single sequence as one sample
        """
        return len(self.user_seq)


# Dynamic Segmentation operations
def DS_default(i_file, o_file):
    """
    :param i_file: original data
    :param o_file: output data
    :return:
    """
    with open(i_file, "r+") as fr:
        data = fr.readlines()
    aug_d = {}
    for d_ in data:
        u_i, item = d_.split(' ', 1)
        item = item.split(' ')
        item[-1] = str(eval(item[-1]))
        aug_d.setdefault(u_i, [])
        start = 0
        j = 3
        if len(item) > 53:
            while start < len(item) - 52:
                j = start + 4
                while j < len(item):
                    if start < 1 and j - start < 53:
                        aug_d[u_i].append(item[start:j])
                        j += 1
                    else:
                        aug_d[u_i].append(item[start:start + 53])
                        break
                start += 1
        else:
            while j < len(item):
                aug_d[u_i].append(item[start:j + 1])
                j += 1
    with open(o_file, "w+") as fw:
        for u_i in aug_d:
            for i_ in aug_d[u_i]:
                fw.write(u_i + " " + ' '.join(i_) + "\n")


# Dynamic Segmentation operations
def DS(i_file, o_file, max_len):
    """
    :param i_file: original data
    :param o_file: output data
    :max_len: the max length of the sequence
    :return:
    """
    with open(i_file, "r+") as fr:
        data = fr.readlines()
    aug_d = {}
    # training, validation, and testing
    max_save_len = max_len + 3
    # save
    max_keep_len = max_len + 2
    for d_ in data:
        u_i, item = d_.split(' ', 1)
        item = item.split(' ')
        item[-1] = str(eval(item[-1]))
        aug_d.setdefault(u_i, [])
        start = 0
        j = 3
        if len(item) > max_save_len:
            # training, validation, and testing
            while start < len(item) - max_keep_len:
                j = start + 4
                while j < len(item):
                    if start < 1 and j - start < max_save_len:
                        aug_d[u_i].append(item[start:j])
                        j += 1
                    else:
                        aug_d[u_i].append(item[start:start + max_save_len])
                        break
                start += 1
        else:
            while j < len(item):
                aug_d[u_i].append(item[start:j + 1])
                j += 1
    with open(o_file, "w+") as fw:
        for u_i in aug_d:
            for i_ in aug_d[u_i]:
                fw.write(u_i + " " + ' '.join(i_) + "\n")


class SASRecDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type="train"):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        # make a deep copy to avoid original sequence be modified
        copied_input_ids = copy.deepcopy(input_ids)
        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
            )

        return cur_rec_tensors

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]
        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0]  # no use

        elif self.data_type == "valid":
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]

        return self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)

    def __len__(self):
        return len(self.user_seq)


if __name__ == "__main__":
    # dynamic segmentation
    DS("../data/Beauty.txt", "../data/Beauty_1.txt", 10)
    # DS_default("../data/Beauty.txt", "../data/Beauty_1.txt")
    # generate target item
    g = Generate_tag("../data", "Beauty", "../data")
    # generate the dictionary
    data = g.get_data("../data/Beauty_1_t.pkl", "train")
    i = 0
    # Only one sequence in the data dictionary in the training phase has the target item ID
    for d_ in data:
        if len(data[d_]) < 2:
            i += 1
            print("less is : ", data[d_], d_)
    print(i)
