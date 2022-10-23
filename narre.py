"""
WWW'18 Neural Attentional Rating Regression with Review-level Explanations
"""

import os
from abc import ABC
import sys

sys.path.append("..")

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from load_data import load_data_for_review_based_rating_prediction
import numpy as np
from util import change_optimizer_device, change_triplet_data_type, get_logger, \
    get_optimizer


class Args:
    batch_size = 512
    epochs = 100

    n_filters = 32
    kernel_size = 3
    word_dim = 100
    fc_dim = 32
    dropout = 0.5
    optim = 'Adam'

    learning_rate = 1e-3
    weight_decay = 1e-3

    device = 'cuda:0'

    dataset_name = 'Digital_Music_5'
    dataset_path = '/home/d1/shuaijie/data/Digital_Music_5/Digital_Music_5.json'
    user_n_reviews = 8
    item_n_reviews = 8
    n_tokens = 128
    dropout = 0.5
    optim = 'Adam'

    model_short_name = 'NARRE'
    model_name = 'NARRE_word_dim_{}'.format(word_dim)

    model_save_path = '../checkpoint/{}/NARRE/{}.pkl'.format(dataset_name, model_name)


logger = get_logger('NARRE', None)


class ReviewDataset(Dataset):

    def __init__(self, user, item, rating, user_doc, item_doc, word2id, args, user2item_id, item2user_id):
        self.user = user.astype(np.int64) + 1
        self.item = item.astype(np.int64) + 1
        self.rating = rating.astype(np.float32)
        self.user_n_reviews = args.user_n_reviews
        self.item_n_reviews = args.item_n_reviews
        self.n_tokens = args.n_tokens
        self.word2id = word2id

        # forbid second calculation when create valid and test dataset
        if type(user_doc.values().__iter__().__next__()) is not np.ndarray:
            self.user_doc, self.user2item_id = self.process_doc(user_doc, self.user_n_reviews, 'item_id')
        else:
            self.user_doc = user_doc
            self.user2item_id = user2item_id

        if type(item_doc.values().__iter__().__next__()) is not np.ndarray:
            self.item_doc, self.item2user_id = self.process_doc(item_doc, self.item_n_reviews, 'user_id')
        else:
            self.item_doc = item_doc
            self.item2user_id = item2user_id

        pass

    def process_doc(self, doc, n_reviews, user_or_item_id):
        doc_review = {}
        id_list = {}
        for k, v in tqdm(doc.items(), desc='process doc'):
            review_info = v[: n_reviews]
            review_text = [x['review_text'] for x in review_info]

            review_text = [x.split()[:self.n_tokens] for x in review_text]
            review_text = [' '.join(x) for x in review_text]
            review_text = [self.parse_word_to_idx(x) for x in review_text]
            review_text = [self.pad_sentence(x) for x in review_text]
            review_text = np.array(review_text)

            review_ids = [x[user_or_item_id] for x in review_info]
            review_ids = np.array(review_ids, dtype=np.int64)

            pad_length = n_reviews - review_text.shape[0]
            if pad_length > 0:
                review_text = np.pad(review_text, ((0, pad_length), (0, 0)), 'constant', constant_values=0)
                review_ids = np.pad(review_ids, (0, pad_length), 'constant', constant_values=0)

            doc_review[k+1] = review_text
            id_list[k+1] = review_ids
        return doc_review, id_list

    def parse_word_to_idx(self, sentence):
        idx = np.array([self.word2id[x] for x in sentence.split()], dtype=np.int64)
        return idx

    def __getitem__(self, idx):
        user_id = self.user[idx]
        item_id = self.item[idx]
        rating = self.rating[idx]

        user_doc = self.user_doc[user_id]
        item_doc = self.item_doc[item_id]

        user2item_id = self.user2item_id[user_id]
        item2user_id = self.item2user_id[item_id]

        return user_id, item_id, rating, user_doc, item_doc, user2item_id, item2user_id

    def pad_sentence(self, sentence):
        if sentence.shape[0] < self.n_tokens:
            pad_length = self.n_tokens - sentence.shape[0]
            sentence = np.pad(sentence, (0, pad_length), 'constant', constant_values=0)
        return sentence

    def __len__(self):
        return self.user.shape[0]


class Bias(nn.Module, ABC):

    def __init__(self, user_size, item_size, rating_mean):
        super(Bias, self).__init__()
        self.user_bias = nn.Embedding(user_size, 1)
        self.item_bias = nn.Embedding(item_size, 1)
        self.user_bias.weight.data.normal_(0, 0.1)
        self.item_bias.weight.data.normal_(0, 0.1)
        self.global_bias = nn.Parameter(torch.FloatTensor([rating_mean]),
                                        requires_grad=False)
        # self.global_bias.requires_grad = False

    def forward(self, u_id, i_id):
        u_b = self.user_bias(u_id).view(-1)
        i_b = self.item_bias(i_id).view(-1)
        return u_b + i_b + self.global_bias


class Net(nn.Module, ABC):

    def __init__(self, narre, bias):
        super(Net, self).__init__()
        self.narre = narre
        self.bias = bias

    def forward(self, user_id, item_id, user_doc, item_doc, user2item_id, item2user_id):
        narre_r = self.narre(user_id, item_id, user_doc, item_doc, user2item_id, item2user_id)
        bias_r = self.bias(user_id, item_id)
        return narre_r + bias_r


class NARRE(nn.Module):

    def __init__(self, n_filters, kernel_size, word_embedding, fc_dim, drop_out, user_size, item_size, rating_mean, args):
        super(NARRE, self).__init__()
        self.args = args

        _, word_dim = word_embedding.shape
        self.word_embed = nn.Embedding.from_pretrained(torch.from_numpy(word_embedding))
        self.word_embed.weight.requires_grad = True

        self.user_cnn = nn.Conv2d(1, n_filters, (kernel_size, word_dim))
        self.item_cnn = nn.Conv2d(1, n_filters, (kernel_size, word_dim))

        self.user_review_linear = nn.Linear(n_filters, fc_dim)
        self.item_review_linear = nn.Linear(n_filters, fc_dim)

        self.user_embed = nn.Embedding(user_size, args.fc_dim)
        self.item_embed = nn.Embedding(item_size, args.fc_dim)

        self.user_importance_embed = nn.Embedding(user_size, args.fc_dim)
        self.item_importance_embed = nn.Embedding(item_size, args.fc_dim)

        self.user_embed_linear = nn.Linear(fc_dim, fc_dim)
        self.item_embed_linear = nn.Linear(fc_dim, fc_dim)

        self.user_attention_layer = nn.Linear(fc_dim, 1)
        self.item_attention_layer = nn.Linear(fc_dim, 1)

        # see eq(7) in paper
        self.user_b1 = nn.Parameter(torch.Tensor(fc_dim), requires_grad=True)
        self.item_b1 = nn.Parameter(torch.Tensor(fc_dim), requires_grad=True)

        self.user_drop_out = nn.Dropout(drop_out)
        self.item_drop_out = nn.Dropout(drop_out)

        self.predictor = nn.Linear(fc_dim, 1, bias=False)
        self.rating_mean = nn.Parameter(torch.FloatTensor([rating_mean]), requires_grad=False)

        self.init_param()

    def init_param(self):
        nn.init.xavier_normal_(self.user_cnn.weight)
        nn.init.xavier_normal_(self.item_cnn.weight)
        nn.init.uniform_(self.user_cnn.bias, -0.1, 0.1)
        nn.init.uniform_(self.item_cnn.bias, -0.1, 0.1)

        nn.init.uniform_(self.user_embed_linear.weight, -0.1, 0.1)
        nn.init.constant_(self.user_embed_linear.bias, 0.1)
        nn.init.uniform_(self.item_embed_linear.weight, -0.1, 0.1)
        nn.init.constant_(self.item_embed_linear.bias, 0.1)

        nn.init.uniform_(self.user_review_linear.weight, -0.1, 0.1)
        nn.init.constant_(self.user_review_linear.bias, 0.1)
        nn.init.uniform_(self.item_review_linear.weight, -0.1, 0.1)
        nn.init.constant_(self.item_review_linear.bias, 0.1)
        
        nn.init.uniform_(self.user_attention_layer.weight, 0, 0.1)
        nn.init.uniform_(self.item_attention_layer.weight, 0, 0.1)

        self.user_embed.weight.data.normal_(0, 0.1)
        self.item_embed.weight.data.normal_(0, 0.1)

        self.user_importance_embed.weight.data.normal_(0, 0.1)
        self.item_importance_embed.weight.data.normal_(0, 0.1)

        nn.init.normal_(self.user_b1, 0, 0.01)
        nn.init.normal_(self.item_b1, 0, 0.01)

    def forward(self, user_id, item_id, user_doc, item_doc, user2item_id, item2user_id):
        """
        doc: batch_size * n_reviews * n_tokens * word_dim
        """

        user_embed = self.user_embed(user_id)
        item_embed = self.item_embed(item_id)

        u_import = self.user_importance_embed(item2user_id)
        i_import = self.item_importance_embed(user2item_id)

        user_doc = self.word_embed(user_doc)
        item_doc = self.word_embed(item_doc)

        b_s, u_n_r, n_t, w_d = user_doc.shape
        user_doc = user_doc.reshape(b_s * u_n_r, 1, n_t, w_d)
        b_s, i_n_r, n_t, w_d = item_doc.shape
        item_doc = item_doc.reshape(b_s * i_n_r, 1, n_t, w_d)

        u_fea = func.relu(self.user_cnn(user_doc)).squeeze(3)  # (batch_size * n_reviews) * n_filters * n_tokens
        i_fea = func.relu(self.item_cnn(item_doc)).squeeze(3)

        u_fea = u_fea.max(dim=2).values  # batch_size * n_filters
        i_fea = i_fea.max(dim=2).values
        u_fea = u_fea.reshape(b_s, u_n_r, -1)  # batch_size * n_reviews * n_filters
        i_fea = i_fea.reshape(b_s, i_n_r, -1)

        u_att_score = self.user_attention_layer(
            func.relu(
                self.user_review_linear(u_fea) + self.item_embed_linear(i_import) + self.user_b1
            )
        )  # batch_size * n_reviews * 1
        
        u_att_score = self.softmax_with_pad(u_att_score, user2item_id)

        u_fea = u_fea * u_att_score
        u_fea = u_fea.sum(dim=1)
        u_fea = self.user_drop_out(u_fea)
        u_fea = user_embed + u_fea

        i_att_score = self.item_attention_layer(
            func.relu(
                self.item_review_linear(i_fea) + self.user_embed_linear(u_import) + self.item_b1
            )
        )
        
        i_att_score = self.softmax_with_pad(i_att_score, item2user_id)

        i_fea = i_fea * i_att_score
        i_fea = i_fea.sum(dim=1)
        i_fea = self.item_drop_out(i_fea)
        i_fea = item_embed + i_fea

        r = self.predictor(u_fea * i_fea)
        r = r.view(-1)
        return r
    
    def softmax_with_pad(self, attn, pad):
        """
        :param attn: batch_size * length * 1
        :param dim: batch_size * length
        """
        pad = pad > 0
        pad = pad.unsqueeze(dim=2)
        attn = torch.exp(attn)
        attn = attn * pad
        s = attn.sum(dim=1, keepdims=True) + 1e-10
        attn = attn / s
        return attn

    def change_tensor_device(self, *tensors):
        result = [x.to(self.args.device) for x in tensors]
        return result

    def train_one_epoch(self, data_loader, optim):
        self.train()

        batch_rating_loss = []

        pbar = tqdm(data_loader)
        for inputs in pbar:
            user_id, item_id, rating_target, user_doc, item_doc, user2item_id, item2user_id \
                = self.change_tensor_device(*inputs)

            rating = self.forward(user_id, item_id, user_doc, item_doc, user2item_id, item2user_id)
            train_loss = self.train_loss(rating, rating_target)
            optim.zero_grad()
            train_loss.backward()
            optim.step()

            record_loss = self.record_loss(rating, rating_target)
            batch_rating_loss.extend(record_loss.cpu().detach())

            pbar.set_description('train_loss:{:<4.3f}'.format(train_loss))

        batch_rating_loss = torch.Tensor(batch_rating_loss)
        batch_rating_loss = batch_rating_loss.mean().sqrt()

        return batch_rating_loss

    def valid_one_data_loader(self, data_loader):
        self.eval()
        total_loss = []
        for inputs in data_loader:
            user_id, item_id, rating_target, user_doc, item_doc, user2item_id, item2user_id \
                = self.change_tensor_device(*inputs)

            rating = self.forward(user_id, item_id, user_doc, item_doc, user2item_id, item2user_id)
            loss = self.record_loss(rating, rating_target)
            total_loss.extend(loss.cpu().detach())

        total_loss = torch.Tensor(total_loss)
        total_loss = total_loss.mean().sqrt()

        return total_loss


def top_review_length(doc: dict, top=0.8):
    docs = list(doc.values())
    docs = [[x['review_text'] for x in doc] for doc in docs]
    docs = sum(docs, [])
    sentence_length = [len(x.split()) for x in docs]
    sentence_length.sort()
    length = sentence_length[int(len(sentence_length) * top)]
    length = 128 if length > 128 else length
    return length


def top_n_reviews(doc, top=0.8):
    review_count = [len(x) for x in list(doc.values())]
    review_count.sort()
    n_reviews = review_count[int(len(review_count) * top)]
    return n_reviews


class Trainer:

    def __init__(self, args):
        self.args = args
        logger.info(args.dataset_name)
        logger.info(args.model_name)

        data = load_data_for_review_based_rating_prediction(args.dataset_path)
        train_data = data['train_triplet']
        valid_data = data['valid_triplet']
        test_data = data['test_triplet']
        user_doc = data['user_doc']
        item_doc = data['item_doc']

        data['dataset_info']['user_size'] += 1
        data['dataset_info']['item_size'] += 1

        user_size = data['dataset_info']['user_size']
        item_size = data['dataset_info']['item_size']
        word2id = data['word2id']
        embeddings = data['embeddings']

        rating_mean = train_data[:, 2].mean()

        # args.n_tokens = top_review_length(user_doc)
        logger.info('Review length: {}'.format(args.n_tokens))

        # args.user_n_reviews = top_n_reviews(user_doc)
        # args.item_n_reviews = top_n_reviews(item_doc)

        logger.info('Num of user reviews: {}, num of item reviews:{}'.format(
            args.user_n_reviews, args.item_n_reviews))

        train_dataset = ReviewDataset(*change_triplet_data_type(train_data),
                                      user_doc, item_doc, word2id, args, None, None)
        valid_dataset = ReviewDataset(*change_triplet_data_type(valid_data),
                                      train_dataset.user_doc,
                                      train_dataset.item_doc,
                                      word2id,
                                      args,
                                      train_dataset.user2item_id,
                                      train_dataset.item2user_id)
        test_dataset = ReviewDataset(*change_triplet_data_type(test_data),
                                     train_dataset.user_doc,
                                     train_dataset.item_doc,
                                     word2id,
                                     args,
                                     train_dataset.user2item_id,
                                     train_dataset.item2user_id)

        self.train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                            shuffle=True, pin_memory=False,
                                            num_workers=4, drop_last=True)
        self.valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                            pin_memory=False)
        self.test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                           pin_memory=False)

        narre = NARRE(args.n_filters, args.kernel_size, embeddings, args.fc_dim,
                      args.dropout, user_size, item_size,
                      rating_mean, args)
        bias = Bias(user_size, item_size, rating_mean)
        self.net = Net(narre, bias)

        self.train_loss = torch.nn.MSELoss(reduction='mean')
        self.record_loss = torch.nn.MSELoss(reduction='none')

        self.optimizer = get_optimizer(args.optim)(
            [
                {'params': self.net.narre.parameters(), 'lr': args.learning_rate, 'weight_decay': args.weight_decay},
                {'params': self.net.bias.parameters(), 'lr': args.learning_rate},
            ])

    def train(self):

        epoch_start = 0
        min_loss = float('inf')
        checkpoint = None

        self.net.to(self.args.device)
        over_fit_count = 0
        for e in range(epoch_start, epoch_start + self.args.epochs):
            train_loss = self.train_one_epoch()

            valid_loss = self.valid_one_data_loader(self.valid_data_loader)
            test_loss = self.valid_one_data_loader(self.test_data_loader)

            logger.info(
                f'Epoch:{e:>2d}, '
                f'train_rmse:{train_loss:<5.4f}, '
                f'valid_rmse:{valid_loss:<5.4f}, '
                f'test_rmse:{test_loss:<5.4f}')
            # wandb.log({'epoch': e,
            #            'training_rmse': train_loss,
            #            'valid_rmse': valid_loss,
            #            'test_rmse': test_loss})

            # model.check_importance(valid_data_loader)

            if min_loss > valid_loss:
                min_loss = valid_loss
                # checkpoint = {'epoch': e,
                #               'model_state_dict': self.net.state_dict(),
                #               'optimizer_state_dict': self.optimizer.state_dict(),
                #               'train_loss': train_loss,
                #               'valid_loss': min_loss,
                #               'test_loss': test_loss}
                checkpoint = self.net.state_dict()

                torch.save(checkpoint, self.args.model_save_path)
                logger.info('Model saved.')
                over_fit_count = 0

                # wandb.run.summary['best_epoch'] = checkpoint['epoch']
                # wandb.run.summary['best_valid_loss'] = checkpoint['valid_loss']
                # wandb.run.summary['best_test_loss'] = checkpoint['test_loss']

            else:
                over_fit_count += 1
                if over_fit_count >= 5:
                    break

    def change_tensor_device(self, *tensors):
        result = [x.to(self.args.device) for x in tensors]
        return result

    def train_one_epoch(self):
        self.net.train()

        batch_rating_loss = []

        pbar = tqdm(self.train_data_loader)

        for inputs in pbar:
            user_id, item_id, rating_target, user_doc, item_doc, user2item_id, item2user_id \
                = self.change_tensor_device(*inputs)

            rating = self.net.forward(user_id, item_id, user_doc, item_doc, user2item_id, item2user_id)
            train_loss = self.train_loss(rating, rating_target)
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()

            # record_loss = self.record_loss(rating, rating_target)
            batch_rating_loss.append(train_loss.cpu().detach())

            pbar.set_description('train_loss:{:<4.3f}'.format(train_loss))

        batch_rating_loss = torch.Tensor(batch_rating_loss).mean().sqrt()
        # batch_rating_loss = batch_rating_loss.mean().sqrt()

        return batch_rating_loss

    def valid_one_data_loader(self, data_loader):
        self.net.eval()
        total_loss = []
        for inputs in data_loader:
            user_id, item_id, rating_target, user_doc, item_doc, user2item_id, item2user_id \
                = self.change_tensor_device(*inputs)

            rating = self.net(user_id, item_id, user_doc, item_doc, user2item_id, item2user_id)
            loss = self.record_loss(rating, rating_target)
            total_loss.extend(loss.cpu().detach())

        total_loss = torch.Tensor(total_loss)
        total_loss = total_loss.mean().sqrt()

        return total_loss

if __name__ == '__main__':
    # train(Args)
    trainer = Trainer(Args)
    trainer.train()

