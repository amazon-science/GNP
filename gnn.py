import sys
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter


def make_one_hot(labels, C):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        (N, ), where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.
    Returns : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C, where C is class number. One-hot encoded.
    '''
    labels = labels.unsqueeze(1)
    one_hot = torch.FloatTensor(labels.size(0), C).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    target = Variable(target)
    return target


class GATConvE(MessagePassing):
    """
    Args:
        emb_dim (int): dimensionality of GNN hidden states
        n_ntype (int): number of node types (e.g. 4)
        n_etype (int): number of edge relation types (e.g. 38)
    """

    # def __init__(self, args, emb_dim, n_ntype, n_etype, edge_encoder, head_count=4, aggr="add"):
    def __init__(self, args, emb_dim, n_ntype, n_etype, head_count=4, aggr="add"):
        super(GATConvE, self).__init__(aggr=aggr)
        self.args = args

        assert emb_dim % 2 == 0
        self.emb_dim = emb_dim

        self.n_ntype = n_ntype
        self.n_etype = n_etype
        # self.edge_encoder = edge_encoder
        self.gnn_edge_dim = args.gnn_edge_dim
        self.edge_encoder = torch.nn.Sequential(torch.nn.Linear(self.n_etype + 1 + self.n_ntype * 2, self.gnn_edge_dim), torch.nn.BatchNorm1d(self.gnn_edge_dim), torch.nn.ReLU(), torch.nn.Linear(self.gnn_edge_dim, self.gnn_edge_dim))

        # For attention
        self.head_count = head_count
        assert emb_dim % head_count == 0
        self.dim_per_head = emb_dim // head_count
        # self.linear_key = nn.Linear(2*emb_dim, head_count * self.dim_per_head)
        # self.linear_msg = nn.Linear(2*emb_dim, head_count * self.dim_per_head)
        self.linear_key = nn.Linear(emb_dim + self.gnn_edge_dim, head_count * self.dim_per_head)
        self.linear_msg = nn.Linear(emb_dim + self.gnn_edge_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(emb_dim, head_count * self.dim_per_head)

        self._alpha = None

        # For final MLP
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))

    def forward(self, x, edge_index, edge_type, node_type, return_attention_weights=False):
        """
        x: [N, emb_dim]
        edge_index: [2, E]
        edge_type [E,] -> edge_attr: [E, 39] / self_edge_attr: [N, 39]
        node_type [N,] -> headtail_attr [E, 8(=4+4)] / self_headtail_attr: [N, 8]
        node_feature_extra [N, dim]
        """

        # Prepare edge feature
        edge_vec = make_one_hot(edge_type, self.n_etype + 1)  # [E, 39]
        self_edge_vec = torch.zeros(x.size(0), self.n_etype + 1).to(edge_vec.device)
        self_edge_vec[:, self.n_etype] = 1

        head_type = node_type[edge_index[0]]  # [E,] #head=src
        tail_type = node_type[edge_index[1]]  # [E,] #tail=tgt
        head_vec = make_one_hot(head_type, self.n_ntype)  # [E,4]
        tail_vec = make_one_hot(tail_type, self.n_ntype)  # [E,4]
        headtail_vec = torch.cat([head_vec, tail_vec], dim=1)  # [E,8]
        self_head_vec = make_one_hot(node_type, self.n_ntype)  # [N,4]
        self_headtail_vec = torch.cat([self_head_vec, self_head_vec], dim=1)  # [N,8]

        edge_vec = torch.cat([edge_vec, self_edge_vec], dim=0)  # [E+N, ?]
        headtail_vec = torch.cat([headtail_vec, self_headtail_vec], dim=0)  # [E+N, ?]
        edge_embeddings = self.edge_encoder(torch.cat([edge_vec, headtail_vec], dim=1))  # [E+N, emb_dim]

        # Add self loops to edge_index
        loop_index = torch.arange(0, x.size(0), dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_index], dim=1)  # [2, E+N]

        # origin
        # x = torch.cat([x, node_feature_extra], dim=1)
        x = (x, x)
        aggr_out = self.propagate(edge_index, x=x, edge_attr=edge_embeddings)  # [N, emb_dim]
        out = self.mlp(aggr_out)

        alpha = self._alpha
        self._alpha = None

        if return_attention_weights:
            assert alpha is not None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, edge_index, x_i, x_j, edge_attr):  # i: tgt, j:src
        assert len(edge_attr.size()) == 2
        # assert edge_attr.size(1) == self.emb_dim
        # assert x_i.size(1) == x_j.size(1) == 2*self.emb_dim
        assert x_i.size(1) == x_j.size(1) == self.emb_dim
        assert x_i.size(0) == x_j.size(0) == edge_attr.size(0) == edge_index.size(1)

        key = self.linear_key(torch.cat([x_i, edge_attr], dim=1)).view(-1, self.head_count, self.dim_per_head)  # [E, heads, _dim]
        msg = self.linear_msg(torch.cat([x_j, edge_attr], dim=1)).view(-1, self.head_count, self.dim_per_head)  # [E, heads, _dim]
        query = self.linear_query(x_j).view(-1, self.head_count, self.dim_per_head)  # [E, heads, _dim]

        if self.args.fp16 and self.training and self.args.upcast:
            with torch.cuda.amp.autocast(enabled=False):
                query = query.float() / math.sqrt(self.dim_per_head)
                scores = (query * key.float()).sum(dim=2)  # [E, heads]
        else:
            query = query / math.sqrt(self.dim_per_head)
            scores = (query * key).sum(dim=2)  # [E, heads]

        src_node_index = edge_index[0]  # [E,]
        alpha = softmax(scores, src_node_index)  # [E, heads] #group by src side node
        self._alpha = alpha

        # adjust by outgoing degree of src
        E = edge_index.size(1)  # n_edges
        N = int(src_node_index.max()) + 1  # n_nodes
        ones = torch.full((E,), 1.0, dtype=torch.float).to(edge_index.device)
        src_node_edge_count = scatter(ones, src_node_index, dim=0, dim_size=N, reduce='sum')[src_node_index]  # [E,]
        assert len(src_node_edge_count.size()) == 1 and len(src_node_edge_count) == E
        alpha = alpha * src_node_edge_count.unsqueeze(1)  # [E, heads]

        out = msg * alpha.view(-1, self.head_count, 1)  # [E, heads, _dim]
        return out.view(-1, self.head_count * self.dim_per_head)  # [E, emb_dim]


class OriginConceptEmbedding(nn.Module):
    def __init__(self, concept_num, concept_in_dim, concept_out_dim,
                 pretrained_concept_emb=None, freeze_ent_emb=True, scale=1.0, init_range=0.02):
        super().__init__()
        self.scale = scale

        self.emb = nn.Embedding(concept_num + 2, concept_in_dim)
        if pretrained_concept_emb is not None:
            self.emb.weight.data.fill_(0)
            self.emb.weight.data[:concept_num].copy_(pretrained_concept_emb)
        else:
            self.emb.weight.data.normal_(mean=0.0, std=init_range)
        if freeze_ent_emb:
            self.freeze_net(self.emb)

        if concept_in_dim != concept_out_dim:
            self.cpt_transform = nn.Linear(concept_in_dim, concept_out_dim)
            self.activation = GELU()

    def forward(self, index):
        """
        index: size (bz, a)
        contextualized_emb: size (bz, b, emb_size) (optional)
        """
        # if contextualized_emb is not None:
        #     assert index.size(0) == contextualized_emb.size(0)
        #     if hasattr(self, 'cpt_transform'):
        #         contextualized_emb = self.activation(self.cpt_transform(contextualized_emb * self.scale))
        #     else:
        #         contextualized_emb = contextualized_emb * self.scale
        #     emb_dim = contextualized_emb.size(-1)
        #     return contextualized_emb.gather(1, index.unsqueeze(-1).expand(-1, -1, emb_dim))
        # else:

        if hasattr(self, 'cpt_transform'):
            return self.activation(self.cpt_transform(self.emb(index) * self.scale))
        else:
            return self.emb(index) * self.scale

    def freeze_net(self, module):
        for p in module.parameters():
            p.requires_grad = False


def gelu(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


class MultiheadAttPoolLayer(nn.Module):

    def __init__(self, n_head, d_q_original, d_k_original, dropout=0.1):
        super().__init__()
        assert d_k_original % n_head == 0  # make sure the outpute dimension equals to d_k_origin
        self.n_head = n_head
        self.d_k = d_k_original // n_head
        self.d_v = d_k_original // n_head

        self.w_qs = nn.Linear(d_q_original, n_head * self.d_k)
        self.w_ks = nn.Linear(d_k_original, n_head * self.d_k)
        self.w_vs = nn.Linear(d_k_original, n_head * self.d_v)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q_original + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_v)))

        self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(self.d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask=None):
        """
        q: tensor of shape (b, d_q_original)
        k: tensor of shape (b, l, d_k_original)
        mask: tensor of shape (b, l) (optional, default None)
        returns: tensor of shape (b, n*d_v)
        """
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v

        bs, _ = q.size()
        bs, len_k, _ = k.size()

        qs = self.w_qs(q).view(bs, n_head, d_k)  # (b, n, dk)
        ks = self.w_ks(k).view(bs, len_k, n_head, d_k)  # (b, l, n, dk)
        vs = self.w_vs(k).view(bs, len_k, n_head, d_v)  # (b, l, n, dv)

        qs = qs.permute(1, 0, 2).contiguous().view(n_head * bs, d_k)
        ks = ks.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_k)
        vs = vs.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1)
        output, attn = self.attention(qs, ks, vs, mask=mask)

        output = output.view(n_head, bs, d_v)
        output = output.permute(1, 0, 2).contiguous().view(bs, n_head * d_v)  # (b, n*dv)
        output = self.dropout(output)
        return output, attn


class MatrixVectorScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, mask=None):
        """
        q: tensor of shape (n*b, d_k)
        k: tensor of shape (n*b, l, d_k)
        v: tensor of shape (n*b, l, d_v)

        returns: tensor of shape (n*b, d_v), tensor of shape(n*b, l)
        """
        # V0
        # attn = ((q.float().unsqueeze(1) / self.temperature) * k.float()).sum(2)  # (n*b, l)

        # V1
        # attn = (q.float().unsqueeze(1) * (k.float() / self.temperature)).sum(2)  # (n*b, l)

        # V2
        # attn = (q.float().unsqueeze(1) * k.float()).sum(2)  # (n*b, l)
        # attn = attn / self.temperature

        # V3: seems to work the best (CSQA, OBQA)
        Qmax = torch.abs(q).max().detach().item()
        Kmax = torch.abs(k).max().detach().item()
        if Qmax > Kmax:
            attn = ((q.float().unsqueeze(1) / self.temperature) * k.float()).sum(2)  # (n*b, l)
        else:
            attn = (q.float().unsqueeze(1) * (k.float() / self.temperature)).sum(2)  # (n*b, l)

        # V4
        # Qmax = torch.abs(q).max().detach().item()
        # Kmax = torch.abs(k).max().detach().item()
        # if Qmax < 0.5 and Kmax < 0.5:
        #     attn = (q.float().unsqueeze(1) * k.float()).sum(2) / self.temperature # (n*b, l)
        # else:
        #     if Qmax > Kmax:
        #         attn = ((q.float().unsqueeze(1) / self.temperature) * k.float()).sum(2)  # (n*b, l)
        #     else:
        #         attn = (q.float().unsqueeze(1) * (k.float() / self.temperature)).sum(2)  # (n*b, l)

        # attn = attn.to(dtype=v.dtype)
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = (attn.unsqueeze(2) * v).sum(1)
        return output, attn


class Decoder(nn.Module):
    def __init__(self, args, num_rels, h_dim):
        super().__init__()
        self.args = args
        self.num_relations = num_rels
        self.embedding_dim = h_dim
        # nn.init.xavier_uniform_(self.w_relation,
        #                         gain=nn.init.calculate_gain('relu'))

        self.negative_adversarial_sampling = args.link_negative_adversarial_sampling
        self.adversarial_temperature = args.link_negative_adversarial_sampling_temperature
        self.reg_param = args.link_regularizer_weight

    def forward(self, embs, sample, mode='single'):
        """
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        """

        if mode == 'single':
            batch_size, negative_sample_size = sample[0].shape[0], 1

            head = embs[sample[0]].unsqueeze(1)  # [n_triple, 1, dim]
            relation = self.w_relation[sample[1]].unsqueeze(1)  # [n_triple, 1, dim]
            tail = embs[sample[2]].unsqueeze(1)  # [n_triple, 1, dim]

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.shape

            head = embs[head_part]  # [n_triple, n_neg, dim]
            relation = self.w_relation[tail_part[1]].unsqueeze(1)  # [n_triple, 1, dim]
            tail = embs[tail_part[2]].unsqueeze(1)  # [n_triple, 1, dim]

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.shape

            head = embs[head_part[0]].unsqueeze(1)
            relation = self.w_relation[head_part[1]].unsqueeze(1)

            tail = embs[tail_part]

        else:
            raise ValueError('mode %s not supported' % mode)

        score = self.score(head, relation, tail, mode)  # [n_triple, 1 or n_neg]

        return score

    def score(self, h, r, t, mode):
        raise NotImplementedError

    def reg_loss(self):
        return torch.mean(self.w_relation.pow(2))
        # return torch.tensor(0)

    def loss(self, scores):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        positive_score, negative_score = scores
        if self.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * self.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)  # [n_triple,]

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)  # [n_triple,]

        assert positive_score.dim() == 1
        if len(positive_score) == 0:
            positive_sample_loss = negative_sample_loss = 0.
        else:
            positive_sample_loss = - positive_score.mean()  # scalar
            negative_sample_loss = - negative_score.mean()  # scalar

        loss = (positive_sample_loss + negative_sample_loss) / 2 + self.reg_param * self.reg_loss()

        return loss, positive_sample_loss, negative_sample_loss


class TransEDecoder(Decoder):
    """TransE score function
    Paper link: https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data
    """

    def __init__(self, args, num_rels, h_dim, dist_func='l2'):
        super().__init__(args, num_rels, h_dim)

        self.gamma = self.args.link_gamma
        if dist_func == 'l1':
            dist_ord = 1
        else:  # default use l2
            dist_ord = 2
        self.dist_ord = dist_ord

        print(f"Initializing w_relation for TransEDecoder... (gamma={self.gamma})", file=sys.stderr)
        self.epsilon = 2.0
        self.register_parameter('w_relation', nn.Parameter(torch.Tensor(self.num_relations, self.embedding_dim)))
        self.embedding_range = (self.gamma + self.epsilon) / self.embedding_dim
        with torch.no_grad():
            self.w_relation.uniform_(-self.embedding_range, self.embedding_range)

    def score(self, head, relation, tail, mode):
        """
        Input head/tail has stdev 1 for each element. Scale to stdev 1/sqrt(12) * (b-a) = a/sqrt(3).
        Reference: https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/wikikg2/model.py
        """
        head = head * self.embedding_range / math.sqrt(3.0)
        tail = tail * self.embedding_range / math.sqrt(3.0)

        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma - torch.norm(score, p=self.dist_ord, dim=2)
        return score

    def __repr__(self):
        return '{}(embedding_size={}, num_relations={}, gamma={}, dist_ord={})'.format(self.__class__.__name__,
                                                                                       self.embedding_dim,
                                                                                       self.num_relations,
                                                                                       self.gamma,
                                                                                       self.dist_ord)


class DistMultDecoder(Decoder):
    """DistMult score function
        Paper link: https://arxiv.org/abs/1412.6575
    """

    def __init__(self, args, num_rels, h_dim):
        super().__init__(args, num_rels, h_dim)

        print("Initializing w_relation for DistMultDecoder...", file=sys.stderr)
        self.register_parameter('w_relation', nn.Parameter(torch.Tensor(self.num_relations, self.embedding_dim)))
        self.embedding_range = math.sqrt(1.0 / self.embedding_dim)
        with torch.no_grad():
            self.w_relation.uniform_(-self.embedding_range, self.embedding_range)

    def score(self, head, relation, tail, mode):
        if mode == 'head-batch':
            if self.args.scaled_distmult:
                tail = tail / math.sqrt(self.embedding_dim)
            score = head * (relation * tail)
        else:
            if self.args.scaled_distmult:
                head = head / math.sqrt(self.embedding_dim)
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def __repr__(self):
        return '{}(embedding_size={}, num_relations={})'.format(self.__class__.__name__,
                                                                self.embedding_dim,
                                                                self.num_relations)


class RotatEDecoder(Decoder):
    """RotatE score function
    Paper link: https://arxiv.org/pdf/1902.10197.pdf
    """

    def __init__(self, args, num_rels, h_dim):
        super().__init__(args, num_rels, h_dim)

        self.gamma = self.args.link_gamma

        print(f"Initializing w_relation for RotatEDecoder... (gamma={self.gamma})", file=sys.stderr)
        self.epsilon = 2.0
        self.register_parameter('w_relation', nn.Parameter(torch.Tensor(self.num_relations, self.embedding_dim // 2)))
        self.embedding_range = (self.gamma + self.epsilon) / self.embedding_dim
        with torch.no_grad():
            self.w_relation.uniform_(-self.embedding_range, self.embedding_range)

    def score(self, head, relation, tail, mode):
        """
        Input head/tail has stdev 1 for each element. Scale to stdev 1/sqrt(12) * (b-a) = a/sqrt(3).
        Reference: https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/wikikg2/model.py
        """
        head = head * self.embedding_range / math.sqrt(3.0)
        tail = tail * self.embedding_range / math.sqrt(3.0)

        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma - score.sum(dim=2)
        return score

    def __repr__(self):
        return '{}(embedding_size={}, num_relations={}, gamma={}, dist_ord={})'.format(self.__class__.__name__,
                                                                                       self.embedding_dim,
                                                                                       self.num_relations,
                                                                                       self.gamma)
