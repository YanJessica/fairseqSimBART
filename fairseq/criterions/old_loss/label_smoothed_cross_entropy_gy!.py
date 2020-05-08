'''
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np

from fairseq import utils
#maggie adding encoders for printing texts
from fairseq.data import encoders
#maggie adding ended

from . import FairseqCriterion, register_criterion

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_dim = 50264
sim_mul = 20

# with torch.autograd.set_detect_anomaly(True)

def init_embeddings():
    path = '/home/ubuntu/project/semsim/fairseq-semsim/pretrained/bart.large.cnn/'
    model = torch.load(path+'model.pt')
    embed_tokens = model['model']['encoder.embed_tokens.weight']
    embed_positions = model['model']['encoder.embed_positions.weight']
    return embed_tokens, embed_positions
#

embed_tokens, embed_positions = init_embeddings()
embed_tokens = embed_tokens.to(device)[-vocab_dim:, :]   # TODO 50265 * 1024
embed_tokens.requires_grad = False

class MyLSTM(nn.Module):
    def __init__(self):
        super(MyLSTM, self).__init__()
        # vocab_dim = 50264
        self.emb_dim = 1024
        # self.embed_tokens, self.embed_positions = init_embeddings()
        # self.embed_tokens = self.embed_tokens.to(device)[-vocab_dim:, :]   # TODO 50265 * 1024
        # self.embed_tokens.requires_grad = False
        # self.embed_positions = self.embed_positions.to(device)
        # self.embeddings = nn.Embedding(embed_tokens.shape[0], embed_tokens.shape[1], _weight=embed_tokens).to(device)
        # self.embeddings = BertModel.from_pretrained('bert-base-uncased').embeddings
        # self.emb = nn.Embedding(num_embeddings=vocab_dim, embedding_dim=emb_dim)
        # input of lstm: (seq_len, batch, input_size)
        self.lstm = nn.LSTM(input_size=self.emb_dim, hidden_size=32, num_layers=2, bidirectional=True,
                            batch_first=True)
        # pretrained_weight = torch.rand(vocab_dim, emb_dim)
        # self.emb.weight.data.copy_(pretrained_weight)  # torch.from_numpy(pretrained_weight)
        for layer_p in self.lstm._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.normal_(self.lstm.__getattr__(p), 0.0, 0.02)
    def forward(self, X):
        assert len(X.shape) == 2  # batch_size, seq_len
        # out_emb = self.embeddings(X.long())  # batch_size, seq_len, emb_dim
        out_emb = torch.matmul(X, embed_tokens.float())
        out_emb = out_emb.reshape(1, out_emb.shape[0], out_emb.shape[1])
        output, (h_n, c_n) = self.lstm(out_emb.float())  # output: n, 1, 64
        return output[:, 0, :]  # (n, 64)
#

def soft_argmax(voxels):
    """
    Arguments: voxel patch in shape (batch_size, channel, H, W, depth)
    Return: 3D coordinates in shape (batch_size, channel, 3)
    """
    assert voxels.dim()==2
    voxels = voxels.reshape(1, voxels.shape[0], voxels.shape[1], 1, 1)
    # alpha is here to make the largest element really big, so it
    # would become very close to 1 after softmax
    alpha = 1000.0
    N,C,H,W,D = voxels.shape
    soft_max = nn.functional.softmax(voxels.view(N,C,-1)*alpha,dim=2)
    soft_max = soft_max.view(voxels.shape)
    indices_kernel = torch.arange(start=0,end=H*W*D).unsqueeze(0).to(device)
    indices_kernel = indices_kernel.view((H,W,D))
    conv = soft_max*indices_kernel
    indices = conv.sum(2).sum(2).sum(2)
    z = indices%D
    y = (indices/D).floor()%W
    x = (((indices/D).floor())/W).floor()%H
    coords = torch.stack([x,y,z],dim=2)
    return coords[0, :, 0].round()
#


def get_l_output_option1(lprobs):  # small change with nll loss
    l_output = lprobs
    l_output_max = l_output.max(-1, keepdim=True)[0]
    l_output_mask = l_output.ge(l_output_max).float()
    l_output = l_output * l_output_mask
    l_output_mul = 1/l_output.detach().cpu().numpy()
    l_output_mul[np.isinf(l_output_mul)] = 0
    l_output = l_output * torch.tensor(l_output_mul).to(device)
    # print('l_output', l_output)  # <MulBackward0>
    return l_output
#


def get_l_output_option2(lprobs):  # nll loss change
    l_output = lprobs
    l_output_max = l_output.max(-1, keepdim=True)[0]
    l_output_mask = l_output.ge(l_output_max).float()
    l_output = l_output * l_output_mask
    # print('l_output', l_output)  # <MulBackward0>
    return l_output
#


def get_l_output_option3(lprobs):  # small change with nll loss
    l_output = lprobs
    # Option 1 
    l_output_max = l_output.max(-1, keepdim=True)[0]
    l_output_mask = l_output.ge(l_output_max).float()
    l_output = l_output * l_output_mask
    l_output[l_output != 0] = 1
    # print('l_output', l_output)  # <IndexPutBackward>
    return l_output
#

def get_l_output_option4(lprobs):  # cannot gradient, don't use
    l_output = lprobs
    for i in range(len(lprobs)):
        l_output[i][l_output[i] != l_output[i].max()] = 0
        l_output[i][l_output[i] == l_output[i].max()] = 1
    # print('l_output', l_output)  
    return l_output
#

def get_l_output_option5(lprobs):  # too slow, don't use
    for i in range(len(lprobs)):
        has_max = False
        i_max = l_output[i].max()
        for j in range(len(lprobs[0])):
            if has_max:
                l_output[i, j] = 0
            elif l_output[i, j] == i_max:
                l_output[i, j] = 1
                has_max = True
            else:
                l_output[i, j] = 0   
    # print('l_output', l_output)  
    return l_output
#

def compute_similarity_lstm(lprobs, target, lstm):
    # print('lprobs', lprobs)   # grad_fn=<ViewBackward>
    l_output = get_l_output_option1(lprobs)   # options can be used: 1 or 2
    l_output = l_output.to(device)
    l_target = torch.nn.functional.one_hot(target.reshape(-1).to(device), num_classes=vocab_dim).float()
    output_lstm = lstm(l_output)
    target_lstm = lstm(l_target)
    # print('output_lstm', output_lstm) #grad_fn=<SliceBackward>
    # print('target_lstm', target_lstm) #grad_fn=<SliceBackward>
    diff = torch.norm(output_lstm-target_lstm, dim=1)
    sim = torch.exp(-diff)
    score = torch.mean(torch.pow(sim, 2))
    # print('score', score)  # tensor(0.9810, device='cuda:0', grad_fn=<MeanBackward0>)
    return score
#

def compute_similarity_cosine(lprobs, target):
    # print('lprobs', lprobs)   # grad_fn=<ViewBackward>
    l_output = get_l_output_option1(lprobs)   # options can be used: 1 or 2
    l_output = l_output.to(device)
    l_target = torch.nn.functional.one_hot(target.reshape(-1).to(device), num_classes=vocab_dim).float()
    
    out_emb_output = torch.matmul(l_output, embed_tokens.float())
    out_emb_output = out_emb_output.reshape(1, out_emb_output.shape[0], out_emb_output.shape[1])
    out_emb_target = torch.matmul(l_target, embed_tokens.float())
    out_emb_target = out_emb_target.reshape(1, out_emb_target.shape[0], out_emb_target.shape[1])
    
    torch.nn.functional.cosine_similarity(l_target, x2, dim=1, eps=1e-8)
    # print('output_lstm', output_lstm) #grad_fn=<SliceBackward>
    # print('target_lstm', target_lstm) #grad_fn=<SliceBackward>
    diff = torch.norm(output_lstm-target_lstm, dim=1)
    sim = torch.exp(-diff)
    score = torch.mean(torch.pow(sim, 2))
    # print('score', score)  # tensor(0.9810, device='cuda:0', grad_fn=<MeanBackward0>)
    return score
#

"""
lprobs = torch.tensor([[0.2000, 0.4000, 0.4000],
                        [0.4000, 0.5000, 0.1000],
                        [0.2000, 0.0000, 0.8000],
                        [0.7000, 0.1000, 0.2000]], requires_grad=True).to(device)
target = torch.tensor([[1],
                       [0],
                       [2],
                       [0]]).to(device)

lstm = MyLSTM()
lstm.eval()
score = compute_similarity(lprobs, target, lstm)

# x = torch.rand(10, 20)
"""

def label_smoothed_nll_loss(lprobs, target, epsilon, lstm, ignore_index=None, reduce=True):
    with torch.autograd.set_detect_anomaly(True):
        #for idx in range(10):
        #    print('=' * 20)
        # print('label_smoothed_nll_loss, target.shape', target.shape)  # (n, 1)
        # print('label_smoothed_nll_loss, lprobs.shape', lprobs.shape)  # (n, vocab) (n, 50264)
        # print('lprobs', lprobs)  # device='cuda:0', grad_fn=<ViewBackward>
        # print('target', target)  # device='cuda:0'
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)
        nll_loss = -lprobs.gather(dim=-1, index=target)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        # print('1-nll_loss.shape', nll_loss.shape)       # (n, 1)
        # print('1-smooth_loss.shape', smooth_loss.shape) # (n, 1)
        if ignore_index is not None:
            # print('ignore_index is not None')  # here!
            non_pad_mask = target.ne(ignore_index)
            nll_loss = nll_loss[non_pad_mask]
            smooth_loss = smooth_loss[non_pad_mask]
        else:
            # print('ignore_index is None')
            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = smooth_loss.squeeze(-1)
        if reduce:
            # print('reduce is not None')   # here!
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = epsilon / lprobs.size(-1)
        loss_orig = (1. - epsilon) * nll_loss + eps_i * smooth_loss
        score = compute_similarity_lstm(lprobs, target, lstm)
        # score = compute_similarity_cosine(lprobs, target)
        # print(score) # tensor(0.9810, device='cuda:0', grad_fn=<MeanBackward0>)
        loss = loss_orig - sim_mul * score
        # score.backward()
        print('loss_orig', loss_orig)  # tensor(689.3746, device='cuda:0', grad_fn=<AddBackward0>)
        print('score', score)          # tensor(0.9810, device='cuda:0', grad_fn=<MeanBackward0>)
        print('return loss', loss)     # grad_fn=<SubBackward0>
        # print('nll_loss', nll_loss)        # tensor(646.0529, device='cuda:0', grad_fn=<SumBackward0>)
        # print('smooth_loss', smooth_loss)  # tensor(32706696., device='cuda:0', grad_fn=<SumBackward0>)
        #for idx in range(10):
        #    print('=' * 20)
    return loss, nll_loss


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.lstm = MyLSTM()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm.to(device)
        self.lstm.eval()
        
        #maggie adding more for printing texts:
        args.bpe='gpt2'
        self.bpe = encoders.build_bpe(args)
        self.task = task
        #maggie adding ended

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        
        #maggie adding for print text
        sentence_tok = torch.argmax(utils.log_softmax(net_output[0], dim=-1),-1).flatten() # maxpool
        sentence_txt = self.bpe.decode(self.task.target_dictionary.string(sentence_tok)) 
        ignore_index = self.padding_idx
        if ignore_index is not None:
            non_pad_mask = target.ne(ignore_index)
            target_ig=target[non_pad_mask]

        target_txt = self.bpe.decode(self.task.target_dictionary.string(target_ig))
        print("=="*10)
        print('sentence: ', sentence_txt, len(sentence_tok))
        print('target: ', target_txt, len(target_ig))
        #maggie adding ended
        
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, self.lstm, ignore_index=self.padding_idx, reduce=reduce,
        )
        l_output = soft_argmax(lprobs).reshape(1, -1)
        # l_output = torch.tensor([[1., 1., 2., 0.]], grad_fn=<ViewBackward>)
        l_target = target.reshape(1, -1)
        return loss, nll_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
'''