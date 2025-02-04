import argparse
import logging
import random
import shutil
import time
import json


import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import transformers
try:
    from transformers import (
        ConstantLRSchedule, WarmupLinearSchedule, WarmupConstantSchedule)
except:
    from transformers import get_constant_schedule, get_constant_schedule_with_warmup,  get_linear_schedule_with_warmup
import wandb

from utils import data_utils
from utils import utils

import numpy as np
import os
import sys
import subprocess

# llm
# from transformers import T5Tokenizer, T5Model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from sklearn import metrics

# peft
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType, LoraConfig

# gnn
import gnn
import math
from torch_geometric.nn import GATConv, GIN
from moe import MoE

try:
    import pickle
except:
    pass

logger = logging.getLogger(__name__)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.random.manual_seed(seed)
    torch.backends.cudnn.determinstic = True


def load_data(args, devices, kg):
    one_process_at_a_time = args.data_loader_one_process_at_a_time

    if args.local_rank != -1 and one_process_at_a_time:
        for p_rank in range(args.world_size):
            if args.local_rank != p_rank:  # Barrier
                torch.distributed.barrier()
            dataset = data_utils.DataLoader(args, args.train_statements, args.train_adj,
                                                   args.dev_statements, args.dev_adj,
                                                   args.test_statements, args.test_adj,
                                                   batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                                                   device=devices,
                                                   model_name=args.encoder,
                                                   max_node_num=args.max_node_num, max_seq_length=args.max_seq_len,
                                                   is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
                                                   subsample=args.subsample, n_train=args.n_train, debug=args.debug, cxt_node_connects_all=args.cxt_node_connects_all, kg=kg)
            if args.local_rank == p_rank:  # End of barrier
                torch.distributed.barrier()
    else:
        dataset = data_utils.DataLoader(args, args.train_statements, args.train_adj,
                                               args.dev_statements, args.dev_adj,
                                               args.test_statements, args.test_adj,
                                               batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                                               device=devices,
                                               model_name=args.encoder,
                                               max_node_num=args.max_node_num, max_seq_length=args.max_seq_len,
                                               is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
                                               subsample=args.subsample, n_train=args.n_train, debug=args.debug, cxt_node_connects_all=args.cxt_node_connects_all, kg=kg)

    return dataset


class GAT(torch.nn.Module):
    def __init__(self, args, in_channels, hidden_channels, out_channels, n_ntype, n_etype, heads, gnn_layers=3):
        super().__init__()
        self.gnn_layers = gnn_layers
        if args.use_relational_gnn:
            assert gnn_layers >= 2
            self.middle_conv_list = nn.ModuleList([gnn.GATConvE(args, hidden_channels, n_ntype, n_etype) for _ in range(gnn_layers-1)])
            self.end_conv = gnn.GATConvE(args, hidden_channels, n_ntype, n_etype)
        else:
            self.start_conv = GATConv(in_channels, hidden_channels, heads, dropout=0.6)

            if gnn_layers >= 3:
                self.middle_conv_list = nn.ModuleList([GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=0.6) for i in range(gnn_layers-2)])

            self.end_conv = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index, edge_type, node_type):
        if args.use_relational_gnn:
            x = F.dropout(x, p=0.6, training=self.training)
            for middle_conv in self.middle_conv_list:
                x = middle_conv(x, edge_index, edge_type, node_type)
                x = F.elu(x)
                x = F.dropout(x, p=0.6, training=self.training)
            x = self.end_conv(x, edge_index, edge_type, node_type)

        else:
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.start_conv(x, edge_index)

            # middle layers
            if self.gnn_layers >= 3:
                for middle_conv in self.middle_conv_list:
                    x = F.elu(x)
                    x = F.dropout(x, p=0.6, training=self.training)
                    x = middle_conv(x, edge_index)

            # output layer
            x = F.elu(x)
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.end_conv(x, edge_index)

        return x


class ShallowQFormer(nn.Module):
    def __init__(self, dim_input, n_query, n_layers=1):  # , dim_output
        super(ShallowQFormer, self).__init__()
        self.n_query = n_query
        self.queries = nn.Parameter(torch.randn(n_query, dim_input), requires_grad=True)
        self.model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=dim_input, dim_feedforward=dim_input * 4, nhead=8, batch_first=True),  # , dropout=0.2
            num_layers=n_layers
        )

    def forward(self, input_feats, memory_mask):
        # input_feats: shape B x n_patch x dim_input
        memory_mask = memory_mask.repeat(8, 1, self.n_query)
        memory_mask = torch.transpose(memory_mask, 2, 1)
        print('tgt: ', self.queries.unsqueeze(0).expand(len(input_feats), *self.queries.shape).shape)
        x = self.model(tgt=self.queries.unsqueeze(0).expand(len(input_feats), *self.queries.shape), memory=input_feats, memory_mask=memory_mask)
        x = F.dropout(x, p=0.2, training=self.training)

        return x


class CrossModalityPooler(nn.Module):
    def __init__(self, args, dim_input, n_query, n_layers=1):  # , dim_output
        super(CrossModalityPooler, self).__init__()
        self.n_query = n_query
        self.dim_input = dim_input
        self.queries = nn.Parameter(torch.randn(n_query, dim_input), requires_grad=True)
        self.model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=dim_input, dim_feedforward=dim_input * 4, nhead=8, batch_first=True),  # , dropout=0.2
            num_layers=n_layers
        )

    def forward(self, input_feats, memory_mask, valid_text_emb):
        # text input
        valid_text_emb = valid_text_emb.unsqueeze(1)

        if args.token_dim < self.dim_input:
            raise ValueError("Need a mapping to increase the dimension of valid_text_emb to match self.dim_input")
        valid_text_emb = valid_text_emb.view(valid_text_emb.size(0), -1, self.dim_input)
        cur_diff_between_dim_input_and_token_dim = args.token_dim / self.dim_input

        valid_text_emb = valid_text_emb.repeat(args.num_choice, self.n_query, 1)

        # memory mask
        memory_mask = memory_mask.repeat(8, 1, int(self.n_query*cur_diff_between_dim_input_and_token_dim))
        memory_mask = torch.transpose(memory_mask, 2, 1)

        x = self.model(tgt=valid_text_emb, memory=input_feats, memory_mask=memory_mask)

        return x


class CrossModalityMHA(nn.Module):
    def __init__(self, args, dim_input, n_layers=1):  # , dim_output
        super(CrossModalityMHA, self).__init__()
        self.dim_input = dim_input
        self.model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=dim_input, dim_feedforward=dim_input * 4, nhead=8, batch_first=True),
            num_layers=n_layers
        )
        self.linear_text_emb = nn.Sequential(nn.Linear(args.token_dim, dim_input), gnn.GELU(), nn.Linear(dim_input, dim_input))

    def forward(self, input_gnn_feats, input_gnn_mask, text_emb, text_mask):
        text_emb = text_emb.to(torch.float32)
        text_mask = text_mask.to(torch.float32)
        text_emb = text_emb.repeat(args.num_choice, 1, 1)
        text_mask = text_mask.repeat(args.num_choice, 1)

        text_emb = self.linear_text_emb(text_emb)

        x = self.model(tgt=input_gnn_feats, tgt_key_padding_mask=input_gnn_mask, memory=text_emb, memory_key_padding_mask=text_mask)

        return x


class CrossModalityMHA_for_text(nn.Module):
    def __init__(self, args, dim_input, n_layers=1):
        super(CrossModalityMHA_for_text, self).__init__()
        self.dim_input = dim_input
        self.model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=args.token_dim, dim_feedforward=args.token_dim * 4, nhead=8, batch_first=True),
            num_layers=n_layers
        )
        self.linear_gnn_emb = nn.Sequential(nn.Linear(dim_input, args.token_dim), gnn.GELU(), nn.Linear(args.token_dim, args.token_dim))

    def forward(self, input_gnn_feats, input_gnn_mask, text_emb, text_mask=None):
        text_emb = text_emb.to(torch.float32)
        if text_mask != None:
            text_mask = text_mask.to(torch.float32)
        input_gnn_feats = input_gnn_feats.view(text_emb.size(0), -1, input_gnn_feats.size(2))
        input_gnn_mask = input_gnn_mask.view(text_emb.size(0), -1)

        input_gnn_feats = self.linear_gnn_emb(input_gnn_feats)

        if text_mask == None:
            x = self.model(tgt=text_emb, memory=input_gnn_feats, memory_key_padding_mask=input_gnn_mask)
        else:
            x = self.model(tgt=text_emb, tgt_key_padding_mask=text_mask, memory=input_gnn_feats, memory_key_padding_mask=input_gnn_mask)

        return x


class MyGNN(nn.Module):
    def __init__(self, args, cp_emb):
        super(MyGNN, self).__init__()

        # init gat
        self.gnn_layers = args.gnn_layers
        self.hidden_size = args.gnn_dim
        self.n_ntype = args.n_ntype
        self.n_etype = args.n_etype
        args.gnn_edge_dim = int(args.gnn_dim / 4)

        self.gat = GAT(args, self.hidden_size, self.hidden_size, self.hidden_size, self.n_ntype, self.n_etype, heads=4, gnn_layers=self.gnn_layers)

        # init concept embedding layer
        self.concept_num = args.concept_num
        self.concept_in_dim = args.concept_in_dim
        self.freeze_ent_emb = args.freeze_ent_emb
        self.concept_emb = gnn.OriginConceptEmbedding(concept_num=self.concept_num, concept_out_dim=self.hidden_size, concept_in_dim=self.concept_in_dim, pretrained_concept_emb=cp_emb, freeze_ent_emb=self.freeze_ent_emb)

        self.activation = gnn.GELU()

        self.p_dropout_gnn = args.p_dropout_gnn

        if args.cross_modality_layers > 0:
            self.cross_modality_MHA = CrossModalityMHA(args, self.hidden_size, n_layers=args.cross_modality_layers)

        # init projector
        if args.moe_experts:
            args.num_activated_experts = args.moe_experts
            assert args.num_activated_experts <= args.moe_experts
            self.projector = MoE(input_size=self.hidden_size, hidden_size=self.hidden_size, output_size=args.token_dim, num_experts=args.moe_experts, k=args.num_activated_experts, noisy_gating=True)
        else:
            self.projector = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), gnn.GELU(), nn.Linear(self.hidden_size, args.token_dim))

        # link prediction task
        if args.link_task:
            if args.link_decoder == 'DistMult':
                self.link_pred = gnn.DistMultDecoder(args, num_rels=self.n_etype, h_dim=self.hidden_size)
            elif args.link_decoder == 'TransE':
                self.link_pred = gnn.TransEDecoder(args, num_rels=self.n_etype, h_dim=self.hidden_size)
            elif args.link_decoder == 'RotatE':
                self.link_pred = gnn.RotatEDecoder(args, num_rels=self.n_etype, h_dim=self.hidden_size)
            else:
                raise NotImplementedError
            if args.link_proj_headtail:
                self.link_pred_proj = nn.Linear(self.hidden_size, self.hidden_size)
            if args.link_normalize_headtail == 3:
                self.link_pred_emb_LayerNorm = nn.LayerNorm(self.hidden_size)

    def batch_graph(self, edge_index_init, edge_type_init, pos_triples_init, neg_nodes_init, n_nodes):
        """
        edge_index_init:  list of (n_examples, ). each entry is torch.tensor(2, E?)    ==> [2, total_E]
        edge_type_init:   list of (n_examples, ). each entry is torch.tensor(E?, )     ==> [total_E, ]
        pos_triples_init: list of (n_examples, ). each entry is [h,r,t] where h/r/t: torch.tensor(n_triple?, ) ==> [3, `total_n_triple`]
        neg_nodes_init:   list of (n_examples, ). each entry is torch.tensor(n_triple?, n_neg) ==> [`total_n_triple`, n_neg]
        """
        n_examples = len(edge_index_init)
        edge_index = [edge_index_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
        edge_index = torch.cat(edge_index, dim=1)  # [2, total_E]
        edge_type = torch.cat(edge_type_init, dim=0)  # [total_E, ]

        pos_triples = [[], [], []]
        for _i_ in range(n_examples):
            h = pos_triples_init[_i_][0] + _i_ * n_nodes  # tensor[n_triple?,]
            r = pos_triples_init[_i_][1]  # tensor[n_triple?,]
            t = pos_triples_init[_i_][2] + _i_ * n_nodes  # tensor[n_triple?,]
            pos_triples[0].append(h)
            pos_triples[1].append(r)
            pos_triples[2].append(t)
        pos_triples = torch.stack([torch.cat(item) for item in pos_triples])  # [3, `total_n_triple`] where `total_n_triple` is sum of n_triple within batch
        assert pos_triples.size(0) == 3

        neg_nodes = [neg_nodes_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
        neg_nodes = torch.cat(neg_nodes)  # [`total_n_triple`, n_neg]
        assert neg_nodes.dim() == 2
        assert pos_triples.size(1) == neg_nodes.size(0)
        return edge_index, edge_type, pos_triples, neg_nodes

    def calc_link_loss(self, pos_triples, neg_nodes, gnn_output):
        # pos_triples: [3, `total_n_triple`],  neg_nodes: [`total_n_triple`, n_neg]
        pos_samples = pos_triples  # [3, `total_n_triple`]

        _n_neg = neg_nodes.size(1)
        head_negative_sample = neg_nodes[:, :_n_neg//2]  # [`total_n_triple`, n_neg//2]
        tail_negative_sample = neg_nodes[:, _n_neg//2:_n_neg//2*2]  # [`total_n_triple`, n_neg//2]

        _bs, _, gnn_dim = gnn_output.size()
        embs = gnn_output.view(-1, gnn_dim)  # [`total_n_nodes`, gnn_dim]

        if args.link_proj_headtail:
            embs = self.link_pred_proj(embs)
        if args.link_normalize_headtail == 1:
            embs = embs / torch.norm(embs, p=2, dim=1, keepdim=True).detach()
        elif args.link_normalize_headtail == 2:
            embs = torch.tanh(embs)
        elif args.link_normalize_headtail == 3:
            embs = self.link_pred_emb_LayerNorm(embs)

        positive_score = self.link_pred(embs, pos_samples)  # [`total_n_triple`, 1]
        head_neg_scores = self.link_pred(embs, (pos_samples, head_negative_sample), mode='head-batch')
        tail_neg_scores = self.link_pred(embs, (pos_samples, tail_negative_sample), mode='tail-batch')
        negative_score = torch.cat([head_neg_scores, tail_neg_scores], dim=-1)  # [`total_n_triple`, total_n_neg]
        scores = (positive_score, negative_score)

        link_loss, pos_link_loss, neg_link_loss = self.link_pred.loss(scores)
        return link_loss

    def forward(self, graph_inputs, text_to_GNN_inputs):
        edge_index, edge_type, pos_triples, neg_nodes = [sum(x, []) for x in graph_inputs[:4]]
        concept_ids, node_type_ids, node_scores, adj_lengths = [x.reshape(x.size(0) * x.size(1), *x.size()[2:]) for x in graph_inputs[4:]]

        edge_index, edge_type, pos_triples, neg_nodes = self.batch_graph(edge_index, edge_type, pos_triples, neg_nodes, concept_ids.size(1))

        # GNN inputs
        concept_ids[concept_ids == 0] = self.concept_num + 2

        gnn_input = self.concept_emb(concept_ids - 1).to(node_type_ids.device)
        gnn_input[:, 0] = 0

        X = gnn_input
        _X = X.view(-1, X.size(2)).contiguous()  # [`total_n_nodes`, d_node] where `total_n_nodes` = b_size * n_node
        _node_type = node_type_ids.view(-1).contiguous()  # [`total_n_nodes`, ]

        # regular GAT
        _X = self.gat(_X, edge_index, edge_type, _node_type)

        X = _X.view(node_type_ids.size(0), -1, node_type_ids.size(1), self.hidden_size)  # [4*batch_size, final num_virtual_tokens/4 (only last layer or multi-layer output), n_node, dim]
        num_tokens = int(args.num_choice * X.size(1))
        X = X.squeeze()  # squeeze if we only have 4 gnn tokens

        gnn_output = X

        # obtain node_mask (1 means masked out)
        node_mask = torch.arange(node_type_ids.size(1), device=node_type_ids.device) >= adj_lengths.unsqueeze(1)  # [4bs, 200 (nodes)]
        if num_tokens > args.num_choice:
            # include node_mask for 3 layers concat output
            valid_node_mask = (~node_mask).float().unsqueeze(2).unsqueeze(1)
            memory_mask = node_mask.unsqueeze(2).unsqueeze(1)
        else:
            valid_node_mask = (~node_mask).float().unsqueeze(2)
            memory_mask = node_mask.unsqueeze(2)

        gnn_output = gnn_output * valid_node_mask  # [4bs, 200, hidden_dim]

        # link prediction
        if args.link_task:
            link_loss = self.calc_link_loss(pos_triples, neg_nodes, gnn_output)
        else:
            link_loss = 0.0

        # Cross Modality MHA: input GNN features, memory is LLM tokens
        if args.cross_modality_layers > 0:
            inputs_embeds, attention_mask = text_to_GNN_inputs
            attention_mask_for_padding = ~attention_mask.to(torch.bool)
            graph_vecs = self.cross_modality_MHA(gnn_output, node_mask, inputs_embeds, attention_mask_for_padding)
        else:
            graph_vecs = gnn_output

        # average pooling
        graph_vecs = graph_vecs * valid_node_mask
        graph_vecs = graph_vecs.sum(1) / ((graph_vecs != 0).sum(1) + 1e-10)
        graph_vecs = graph_vecs.unsqueeze(1)
        graph_vecs = graph_vecs.view(-1, int(graph_vecs.size(1))*args.num_choice, graph_vecs.size(2))

        # projector
        if args.no_projector:
            try:
                graph_vecs = graph_vecs.view(graph_vecs.size(0), -1, args.token_dim)
            # for odd token numbers
            except:
                if args.dataset == 'bioasq':
                    temp_solution = torch.zeros(graph_vecs.size(0), 2, graph_vecs.size(2)).to(graph_vecs.device)
                else:
                    temp_solution = torch.zeros(graph_vecs.size(0), 1, graph_vecs.size(2)).to(graph_vecs.device)
                graph_vecs = torch.cat([graph_vecs, temp_solution], dim=1)
                graph_vecs = graph_vecs.view(graph_vecs.size(0), -1, args.token_dim)

            moe_loss = 0.0
        else:
            if args.moe_experts:
                temp_bs = graph_vecs.size(0)
                graph_vecs = graph_vecs.view(-1, graph_vecs.size(-1))
                graph_vecs, moe_loss = self.projector(graph_vecs)
                graph_vecs = graph_vecs.view(temp_bs, -1, graph_vecs.size(-1))
            else:
                graph_vecs = self.projector(graph_vecs)
                moe_loss = 0.0

        if args.use_cross_modality_for_text:
            return graph_vecs, gnn_output, node_mask, attention_mask_for_padding

        return graph_vecs, link_loss, moe_loss


class MyModel(nn.Module):
    def __init__(self, args, cp_emb):
        super(MyModel, self).__init__()

        num_device = torch.cuda.device_count()
        if num_device == 1:
            my_max_memory = {0: "12GB"}
        elif num_device == 2:
            my_max_memory = {0: "10GB", 1: "12GB"}
        elif num_device == 3:
            my_max_memory = {0: "0GB", 1: "12GB", 2: "12GB"}
        elif num_device == 4:
            my_max_memory = {0: "0GB", 1: "12GB", 2: "12GB", 3: "12GB"}
        else:
            my_max_memory = {}
            my_max_memory[0] = "0GB"
            for i in range(1, 8):
                my_max_memory[i] = "12GB"

        if args.prompt == False:
            self.llm = AutoModelForSeq2SeqLM.from_pretrained(args.encoder, torch_dtype=torch.bfloat16, device_map='auto')
            self.llm.gradient_checkpointing_enable()
        elif args.prompt == 'regular':
            self.llm = AutoModelForSeq2SeqLM.from_pretrained(args.encoder, torch_dtype=torch.bfloat16, device_map='auto')
            self.llm.gradient_checkpointing_enable()
        else:
            self.llm = AutoModelForSeq2SeqLM.from_pretrained(args.encoder, torch_dtype=torch.bfloat16, device_map='auto', max_memory=my_max_memory)
            self.llm.gradient_checkpointing_enable()
        print(self.llm.hf_device_map)

        # LoRA
        if args.lora:
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q", "v"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM
            )
            self.llm = get_peft_model(self.llm, peft_config)
            print('using lora !!!!')

        # word_embeddings
        for named_param, value in list(self.llm.named_parameters()):
            if named_param == 'shared.weight' or named_param == 'base_model.model.shared.weight':  # t5
                self.word_embeddings = self.llm.get_submodule(named_param.replace(".weight", ""))
                args.token_dim = self.word_embeddings.weight.shape[1]
                break
        try:
            print('self.word_embeddings: ', self.word_embeddings)
        except:
            raise ValueError("self.word_embeddings cannot be found")

        # regular prompt
        if args.prompt == 'regular':
            self.num_virtual_tokens = args.num_virtual_tokens
            if args.use_wandb:
                wandb.run.summary["num_virtual_tokens"] = self.num_virtual_tokens
            self.token_dim = self.word_embeddings.weight.shape[1]

            # manual prompt word initialization
            init_text = 'Choose the best option to answer the question.'
            init_tokenizer = AutoTokenizer.from_pretrained(args.encoder)
            init_token_ids = init_tokenizer(init_text)["input_ids"]

            # Trim or iterate until num_text_tokens matches total_virtual_tokens
            num_text_tokens = len(init_token_ids)
            if num_text_tokens > self.num_virtual_tokens:
                init_token_ids = init_token_ids[:self.num_virtual_tokens]
            elif num_text_tokens < self.num_virtual_tokens:
                num_reps = math.ceil(self.num_virtual_tokens / num_text_tokens)
                init_token_ids = init_token_ids * num_reps
            init_token_ids = init_token_ids[:self.num_virtual_tokens]

            self.prompt = nn.Parameter(self.word_embeddings(torch.LongTensor(init_token_ids)).detach().clone().to(torch.float32))

        # GNN prompt
        elif args.prompt == 'gnn':
            self.gnn_model = MyGNN(args, cp_emb).to('cuda:0')  # self.llm.device
            print('self.gnn_model: ', self.gnn_model)
            # how many tokens we get from gnn output
            self.record_gnn_num_virtual_tokens = True

            if args.use_cross_modality_for_text:
                self.cross_modality_MHA_for_text = CrossModalityMHA_for_text(args, args.gnn_dim, n_layers=args.cross_modality_for_text_layers)
                self.cross_modality_MHA_for_text.to(self.llm.device)

            if args.dataset_level_prompt:
                # prompt
                self.num_virtual_tokens = args.num_virtual_tokens
                self.token_dim = self.word_embeddings.weight.shape[1]

                # manual prompt word initialization
                init_text = 'Choose the best option to answer the question based on the context. Context:'
                init_tokenizer = AutoTokenizer.from_pretrained(args.encoder)
                init_token_ids = init_tokenizer(init_text)["input_ids"]

                # Trim or iterate until num_text_tokens matches total_virtual_tokens
                num_text_tokens = len(init_token_ids)
                if num_text_tokens > self.num_virtual_tokens:
                    init_token_ids = init_token_ids[:self.num_virtual_tokens]
                elif num_text_tokens < self.num_virtual_tokens:
                    num_reps = math.ceil(self.num_virtual_tokens / num_text_tokens)
                    init_token_ids = init_token_ids * num_reps
                init_token_ids = init_token_ids[:self.num_virtual_tokens]

                self.prompt = nn.Parameter(self.word_embeddings(torch.LongTensor(init_token_ids)).detach().clone().to(torch.float32))

    def generate(self, input_ids, attention_mask, graph_inputs):
        with torch.no_grad():
            inputs_embeds = self.word_embeddings(input_ids)

            if args.prompt == 'regular':
                input_prompt = self.prompt.repeat(inputs_embeds.size(0), 1, 1)
                inputs_embeds = torch.cat((input_prompt, inputs_embeds), dim=1)

                prefix_attention_mask = torch.ones(inputs_embeds.size(0), self.num_virtual_tokens).to(attention_mask.device)
                attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

            elif args.prompt == 'gnn':
                # prepare for cross-modality attention
                text_to_GNN_inputs = inputs_embeds.to('cuda:0'), attention_mask.to('cuda:0')

                # attention from gnn to llm
                if args.use_cross_modality_for_text:
                    # gnn to llm attention: attention on an additional token
                    input_prompt, gnn_output, node_mask, attention_mask_for_padding = self.gnn_model(graph_inputs, text_to_GNN_inputs)
                    input_prompt = input_prompt.to(inputs_embeds.device)
                    gnn_output = gnn_output.to(inputs_embeds.device)
                    node_mask = node_mask.to(inputs_embeds.device)
                    attention_mask_for_padding = attention_mask_for_padding.to(inputs_embeds.device)

                    valid_text_emb = inputs_embeds * attention_mask.unsqueeze(2).to(inputs_embeds.device)
                    valid_text_emb = valid_text_emb.sum(1) / ((valid_text_emb != 0).sum(1) + 1e-10)
                    valid_text_emb = valid_text_emb.unsqueeze(1)
                    special_token_gnn_to_llm_embeds = self.cross_modality_MHA_for_text(gnn_output, node_mask, valid_text_emb)
                    inputs_embeds = torch.cat((input_prompt, special_token_gnn_to_llm_embeds, inputs_embeds), dim=1)

                    prefix_attention_mask = torch.ones(inputs_embeds.size(0), input_prompt.size(1)+special_token_gnn_to_llm_embeds.size(1)).to(attention_mask.device)

                # NO attention from gnn to llm
                else:
                    gnn_input_prompt, _, _ = self.gnn_model(graph_inputs, text_to_GNN_inputs)
                    gnn_input_prompt = gnn_input_prompt.to(inputs_embeds.device)
                    gnn_prefix_attention_mask = torch.ones(inputs_embeds.size(0), gnn_input_prompt.size(1)).to(attention_mask.device)

                    if args.dataset_level_prompt:
                        dataset_level_input_prompt = self.prompt.repeat(inputs_embeds.size(0), 1, 1)
                        dataset_level_prefix_attention_mask = torch.ones(dataset_level_input_prompt.size(0), self.num_virtual_tokens).to(attention_mask.device)
                        inputs_embeds = torch.cat((dataset_level_input_prompt, gnn_input_prompt, inputs_embeds), dim=1)
                        attention_mask = torch.cat((dataset_level_prefix_attention_mask, gnn_prefix_attention_mask, attention_mask), dim=1)
                    else:
                        inputs_embeds = torch.cat((gnn_input_prompt, inputs_embeds), dim=1)
                        attention_mask = torch.cat((gnn_prefix_attention_mask, attention_mask), dim=1)

            return self.llm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=30)

    def forward(self, input_ids, attention_mask, labels, graph_inputs):
        inputs_embeds = self.word_embeddings(input_ids)

        if args.prompt == 'regular':
            input_prompt = self.prompt.repeat(inputs_embeds.size(0), 1, 1)
            inputs_embeds = torch.cat((input_prompt, inputs_embeds), dim=1)

            prefix_attention_mask = torch.ones(inputs_embeds.size(0), self.num_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        elif args.prompt == 'gnn':
            # prepare for cross-modality attention
            text_to_GNN_inputs = inputs_embeds.to('cuda:0'), attention_mask.to('cuda:0')

            # attention from gnn to llm
            if args.use_cross_modality_for_text:
                # gnn to llm attention: attention on an additional token
                input_prompt, gnn_output, node_mask, attention_mask_for_padding = self.gnn_model(graph_inputs, text_to_GNN_inputs)
                input_prompt = input_prompt.to(inputs_embeds.device)
                gnn_output = gnn_output.to(inputs_embeds.device)
                node_mask = node_mask.to(inputs_embeds.device)
                attention_mask_for_padding = attention_mask_for_padding.to(inputs_embeds.device)

                valid_text_emb = inputs_embeds * attention_mask.unsqueeze(2).to(inputs_embeds.device)
                valid_text_emb = valid_text_emb.sum(1) / ((valid_text_emb != 0).sum(1) + 1e-10)
                valid_text_emb = valid_text_emb.unsqueeze(1)
                special_token_gnn_to_llm_embeds = self.cross_modality_MHA_for_text(gnn_output, node_mask, valid_text_emb)
                inputs_embeds = torch.cat((input_prompt, special_token_gnn_to_llm_embeds, inputs_embeds), dim=1)

                prefix_attention_mask = torch.ones(inputs_embeds.size(0), input_prompt.size(1)+special_token_gnn_to_llm_embeds.size(1)).to(attention_mask.device)

                cur_num_virtual_tokens = input_prompt.size(1)

            # NO attention from gnn to llm
            else:
                gnn_input_prompt, link_loss, moe_loss = self.gnn_model(graph_inputs, text_to_GNN_inputs)
                gnn_input_prompt = gnn_input_prompt.to(inputs_embeds.device)
                gnn_prefix_attention_mask = torch.ones(inputs_embeds.size(0), gnn_input_prompt.size(1)).to(attention_mask.device)

                if args.dataset_level_prompt:
                    dataset_level_input_prompt = self.prompt.repeat(inputs_embeds.size(0), 1, 1)
                    dataset_level_prefix_attention_mask = torch.ones(dataset_level_input_prompt.size(0), self.num_virtual_tokens).to(attention_mask.device)
                    inputs_embeds = torch.cat((dataset_level_input_prompt, gnn_input_prompt, inputs_embeds), dim=1)
                    attention_mask = torch.cat((dataset_level_prefix_attention_mask, gnn_prefix_attention_mask, attention_mask), dim=1)
                    cur_num_virtual_tokens = dataset_level_input_prompt.size(1) + gnn_input_prompt.size(1)
                else:
                    inputs_embeds = torch.cat((gnn_input_prompt, inputs_embeds), dim=1)
                    attention_mask = torch.cat((gnn_prefix_attention_mask, attention_mask), dim=1)
                    cur_num_virtual_tokens = gnn_input_prompt.size(1)

            # record num_virtual_tokens
            if self.record_gnn_num_virtual_tokens and args.use_wandb:
                wandb.run.summary["num_virtual_tokens"] = int(cur_num_virtual_tokens)
                self.record_gnn_num_virtual_tokens = False
            return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels), link_loss, moe_loss

        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )


def construct_model(args, kg, dataset):
    print('constructing model ...')

    # Load pretrained concept embeddings
    cp_emb = [np.load(path) for path in args.ent_emb_paths]
    cp_emb = np.concatenate(cp_emb, 1)
    cp_emb = torch.tensor(cp_emb, dtype=torch.float)

    concept_num, concept_in_dim = cp_emb.size(0), cp_emb.size(1)
    print('| num_concepts: {}, concept_in_dim: {} |'.format(concept_num, concept_in_dim))
    args.concept_num = concept_num
    if args.random_ent_emb:
        cp_emb = None
        args.freeze_ent_emb = False
        args.concept_in_dim = args.gnn_dim
    else:
        args.concept_in_dim = concept_in_dim

    if args.dataset == 'csqa':
        args.num_choice = 5
    elif args.dataset == 'obqa':
        args.num_choice = 4
    elif args.dataset == 'riddle':
        args.num_choice = 5
    elif args.dataset == 'medqa':
        args.num_choice = 4
    elif args.dataset == 'arc':
        args.num_choice = 4
    elif args.dataset == 'piqa':
        args.num_choice = 2
    elif args.dataset == 'pubmedqa':
        args.num_choice = 3
    elif args.dataset == 'bioasq':
        args.num_choice = 2
    elif args.dataset == 'cosmosqa':
        args.num_choice = 4
    else:
        raise ValueError("Please indicate how many answer choices this dataset has.")

    # Build model
    if kg == "cpnet":
        n_ntype = 4
        n_etype = 38
        # assert n_etype == dataset.final_num_relation *2
    elif kg == "ddb":
        n_ntype = 4
        n_etype = 34
        # assert n_etype == dataset.final_num_relation *2
    elif kg == "umls":
        n_ntype = 4
        n_etype = dataset.final_num_relation * 2
        print('final_num_relation', dataset.final_num_relation,
              'len(id2relation)', len(dataset.id2relation))
        print('final_num_relation', dataset.final_num_relation,
              'len(id2relation)', len(dataset.id2relation), file=sys.stderr)
    else:
        raise ValueError("Invalid KG.")
    if args.cxt_node_connects_all:
        n_etype += 2
    print('n_ntype', n_ntype, 'n_etype', n_etype)
    print('n_ntype', n_ntype, 'n_etype', n_etype, file=sys.stderr)

    args.n_ntype = n_ntype
    args.n_etype = n_etype

    model = MyModel(args, cp_emb)

    # freeze llm for prompt tuning
    if args.prompt == 'regular':
        if not args.lora:
            # Freeze all the parameters in the model
            for param in model.parameters():
                param.requires_grad = False
        # Unfreeze the prompt Parameter, we want to keep this as trainable
        model.prompt.requires_grad = True
    elif args.prompt == 'gnn':
        if not args.lora:
            for param in model.llm.parameters():
                param.requires_grad = False
    elif args.prompt == False:
        print('!!!'*20)
        print('fine-tuning the full LLM')
        print()
    else:
        raise ValueError('Not Implemented. What is the args.prompt?')

    model.print_trainable_parameters()

    return model


def sep_params(model, loaded_roberta_keys):
    """Separate the parameters into loaded and not loaded."""
    loaded_params = dict()
    not_loaded_params = dict()
    params_to_freeze = []
    small_lr_params = dict()
    large_lr_params = dict()
    for n, p in model.named_parameters():
        if n in loaded_roberta_keys:
            loaded_params[n] = p
            params_to_freeze.append(p)
            small_lr_params[n] = p
        else:
            not_loaded_params[n] = p
            large_lr_params[n] = p

    return loaded_params, not_loaded_params, params_to_freeze, small_lr_params, large_lr_params


def count_parameters(loaded_params, not_loaded_params):
    num_params = sum(p.numel()
                     for p in not_loaded_params.values() if p.requires_grad)
    num_fixed_params = sum(p.numel()
                           for p in not_loaded_params.values() if not p.requires_grad)
    num_loaded_params = sum(p.numel() for p in loaded_params.values())
    print('num_trainable_params (out of not_loaded_params):', num_params)
    print('num_fixed_params (out of not_loaded_params):', num_fixed_params)
    print('num_loaded_params:', num_loaded_params)
    print('num_total_params:', num_params +
          num_fixed_params + num_loaded_params)


def calc_acc_only(outputs, labels):
    if outputs is None:
        return 0
    else:
        return metrics.accuracy_score(labels, outputs)


def add_manual_prompts_to_flattern_graph_str_list(input):
    output = []
    which_option_list = ['first', 'second', 'third', 'fourth', 'fifth']
    output.append('Based on the following concepts of each option, choose the best option to answer the question.')
    for i, concept_of_a_choice_str in enumerate(input):
        output.append('Concept of the ' + which_option_list[i] + ' option: ' + concept_of_a_choice_str + '\n')
    return output


def merge_encoded_encoded_flattern_graph_str_list(input):
    output = []
    for i, concept_of_a_choice_list in enumerate(input):
        output.extend(concept_of_a_choice_list)
    return torch.LongTensor(output)


def calc_eval_accuracy(args, eval_set, model, cur_save_model_wrong_prediction=False, epoch_id=None):
    """Eval on the dev or test set - calculate loss and accuracy"""
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)

    # gpt
    if 'bloomz' in args.encoder:
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

    with torch.no_grad():
        all_outputs = []
        all_labels = []
        all_qids = []
        for qids, labels, batch_lm_inputs, batch_input_masks, batch_graph_inputs in tqdm(eval_set, desc="Eval batch"):
            # baseline: flattern
            if args.baseline_flattern:
                if args.case_study:
                    all_triplets_list, node_type_in_triplets = batch_graph_inputs[-1]
                    all_triplets_list = list(set(all_triplets_list))
                    print('len(all_triplets_list): ', len(all_triplets_list))
                    print('len(node_type_in_triplets): ', len(node_type_in_triplets))

                else:
                    flattern_graph_str_list = batch_graph_inputs[-1]
                    batch_graph_inputs = batch_graph_inputs[:-1]
                    flattern_graph_str_list = add_manual_prompts_to_flattern_graph_str_list(flattern_graph_str_list)
                    encoded_flattern_graph_list = tokenizer(flattern_graph_str_list, padding=False, truncation=True, return_token_type_ids=True, return_special_tokens_mask=True).input_ids  # no padding
                    encoded_flattern_graph_input = merge_encoded_encoded_flattern_graph_str_list(encoded_flattern_graph_list)
                    encoded_flattern_graph_input = encoded_flattern_graph_input.unsqueeze(0)
                    encoded_flattern_graph_input = encoded_flattern_graph_input.to(batch_lm_inputs.device)

            # prepare batch_lm_inputs
            assert batch_lm_inputs.size(-1) == args.max_seq_len
            batch_lm_inputs = batch_lm_inputs.view(-1, args.max_seq_len)  # 1*100
            batch_input_masks = batch_input_masks.view(-1, args.max_seq_len)  # 1*100

            # baseline: flattern. add gnn input to llm input
            if args.baseline_flattern:
                if not args.case_study:
                    batch_lm_inputs = torch.cat([encoded_flattern_graph_input, batch_lm_inputs], 1)  # first gnn prompt, then QA pair
                    flattern_graph_input_mask = torch.ones(encoded_flattern_graph_input.size()).to(batch_input_masks.device)
                    batch_input_masks = torch.cat([flattern_graph_input_mask, batch_input_masks], 1)  # first gnn prompt, then QA pair

            # validate batch_lm_inputs
            if args.debug2:
                decoded_inputs = tokenizer.batch_decode(batch_lm_inputs, skip_special_tokens=True)
                decoded_inputs = decoded_inputs[0]
                print('decoded_inputs: ', decoded_inputs)

            # validate the labels
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            all_labels.extend(decoded_labels)
            if args.debug2:
                print('decoded_labels:', decoded_labels)

            outputs = model.generate(input_ids=batch_lm_inputs, attention_mask=batch_input_masks, graph_inputs=batch_graph_inputs)
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_outputs.extend(decoded_outputs)

            if args.debug2:
                print('decoded_outputs:', decoded_outputs)

            all_qids.extend(qids)

            if args.case_study:
                if decoded_outputs != decoded_labels:
                    with open('case_study/'+qids[0], 'wb') as f:
                        pickle.dump([decoded_inputs, decoded_labels[0], decoded_outputs[0], all_triplets_list, node_type_in_triplets], f)

    all_labels = [i.lower() for i in all_labels]
    all_outputs = [i.lower() for i in all_outputs]

    if cur_save_model_wrong_prediction:
        all_labels_wrong_prediction = []
        all_outputs_wrong_prediction = []
        all_qids_wrong_prediction = []
        for idx in range(len(all_labels)):
            cur_label = all_labels[idx]
            cur_output = all_outputs[idx]
            cur_qid = all_qids[idx]
            if cur_label != cur_output:
                all_labels_wrong_prediction.append(cur_label)
                all_outputs_wrong_prediction.append(cur_output)
                all_qids_wrong_prediction.append(cur_qid)
        with open('case_study/gnn_wrong_pred/'+'epoch_'+str(epoch_id), 'wb') as f:
            pickle.dump([all_labels_wrong_prediction, all_outputs_wrong_prediction, all_qids_wrong_prediction], f)
            print('save successfully.')

    total_loss_avg = 0
    end_loss_avg = 0
    out_accuracy = calc_acc_only(all_outputs, all_labels)
    return total_loss_avg, end_loss_avg, out_accuracy


def train(args, resume, has_test_split, devices, kg):
    print("args: {}".format(args))

    if resume:
        args.save_dir = os.path.dirname(args.resume_checkpoint)
    if not args.debug:
        if args.local_rank in [-1, 0]:
            log_path = os.path.join(args.save_dir, 'log.csv')
            utils.check_path(log_path)

            if not resume:
                with open(log_path, 'w') as fout:
                    fout.write(
                        'epoch,step,dev_acc,test_acc,best_dev_acc,final_test_acc,best_dev_epoch\n')

            config_path = os.path.join(args.save_dir, 'config.json')
            utils.export_config(args, config_path)

    model_path = os.path.join(args.save_dir, 'model.pt')

    # load datasets
    dataset = load_data(args, devices, kg)
    dev_dataloader = dataset.dev()
    if has_test_split:
        test_dataloader = dataset.test()

    train_eval_dataloader = dataset.train_eval()

    model = construct_model(args, kg, dataset)
    INHERIT_BERT = os.environ.get('INHERIT_BERT', 0)
    model.llm.resize_token_embeddings(len(dataset.tokenizer))

    def _rename_key(key):
        return "lmgnn." + key

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Optionally loading from a checkpoint
    if resume:
        print("loading from checkpoint: {}".format(args.resume_checkpoint))
        checkpoint = torch.load(args.resume_checkpoint, map_location='cpu')
        last_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        model.load_state_dict(checkpoint["model"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_dev_epoch = checkpoint["best_dev_epoch"]
        best_dev_acc = checkpoint["best_dev_acc"]
        final_test_acc = checkpoint["final_test_acc"]
        print(
            f"resume from global_step {global_step}, last_epoch {last_epoch}")
    else:
        last_epoch = -1
        global_step = 0
        best_dev_epoch = best_dev_acc = final_test_acc = 0

    # Create a scheduler
    if args.lr_schedule == 'fixed':
        try:
            scheduler = ConstantLRSchedule(optimizer)
        except:
            scheduler = get_constant_schedule(optimizer)
    elif args.lr_schedule == 'warmup_constant':
        try:
            scheduler = WarmupConstantSchedule(
                optimizer, warmup_steps=args.warmup_steps, last_epoch=last_epoch)
        except:
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps, last_epoch=last_epoch)
    elif args.lr_schedule == 'warmup_linear':
        max_steps = int(
            args.n_epochs * (dataset.train_size() / args.batch_size))
        try:
            scheduler = WarmupLinearSchedule(
                optimizer, warmup_steps=args.warmup_steps, t_total=max_steps, last_epoch=last_epoch)
        except:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=max_steps, last_epoch=last_epoch)
    if resume:
        scheduler.load_state_dict(checkpoint["scheduler"])
        print("loaded scheduler", checkpoint["scheduler"])

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[
                args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Construct the loss function
    if args.loss == 'margin_rank':
        loss_func = nn.MarginRankingLoss(margin=0.1, reduction='mean')
    elif args.loss == 'cross_entropy':
        loss_func = nn.CrossEntropyLoss(reduction='mean')
    else:
        raise ValueError("Invalid value for args.loss.")

    #############################################################
    #   Training
    #############################################################

    print()
    print('-' * 71)
    if args.fp16:
        print('Using fp16 training')
        print(f'Upcast {args.upcast}')
        scaler = torch.cuda.amp.GradScaler()

    total_loss_acm = 0.0
    link_loss_acm = pos_link_loss_acm = neg_link_loss_acm = 0.0
    moe_loss_acm = 0.0
    n_samples_acm = n_corrects_acm = 0
    total_time = 0
    model.train()

    tokenizer = AutoTokenizer.from_pretrained(args.encoder)

    for epoch_id in trange(0, args.n_epochs, desc="Epoch", disable=args.local_rank not in [-1, 0]):
        if last_epoch + 1 > epoch_id:
            time.sleep(1)
            continue
        model.train()

        for qids, labels, batch_lm_inputs, batch_input_masks, batch_graph_inputs in tqdm(dataset.train(steps=args.redef_epoch_steps, local_rank=args.local_rank), desc="Batch", disable=args.local_rank not in [-1, 0]):

            start_time = time.time()
            optimizer.zero_grad()
            bs = labels.size(0)
            a_list = list(range(0, bs, args.mini_batch_size))
            for _idx_, a in enumerate(a_list):
                is_last = (_idx_ == len(a_list) - 1)
                b = min(a + args.mini_batch_size, bs)

                # prepare batch_lm_inputs
                batch_lm_inputs = batch_lm_inputs.view(-1, args.max_seq_len)  # 1*100
                batch_input_masks = batch_input_masks.view(-1, args.max_seq_len)  # 1*100

                input_ids = batch_lm_inputs

                labels[labels == tokenizer.pad_token_id] = -100

                if args.prompt == 'gnn':
                    outputs, link_loss, moe_loss = model(input_ids=input_ids, attention_mask=batch_input_masks, labels=labels, graph_inputs=batch_graph_inputs)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=batch_input_masks, labels=labels, graph_inputs=batch_graph_inputs)

                loss = outputs.loss

                if args.prompt == 'gnn' and args.link_task:
                    link_loss = link_loss.to(loss.device)
                    loss += args.link_task * link_loss
                if args.prompt == 'gnn' and args.moe_loss_weight:
                    moe_loss = moe_loss.to(loss.device)
                    loss += args.moe_loss_weight * moe_loss

                total_loss_acm += float(loss)
                if args.prompt == 'gnn' and args.link_task:
                    link_loss_acm += float(link_loss)
                if args.prompt == 'gnn' and args.moe_loss_weight:
                    moe_loss_acm += float(moe_loss)

                loss = loss / bs
                if (args.local_rank != -1) and (not is_last):
                    with model.no_sync():
                        if args.fp16:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                else:
                    if args.fp16:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                n_samples_acm += (b - a)

            if args.max_grad_norm > 0:
                if args.fp16:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)
                else:
                    nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)

            if args.fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()

            total_time += (time.time() - start_time)

            if (global_step + 1) % args.log_interval == 0:
                ms_per_batch = 1000 * total_time / args.log_interval
                if args.local_rank in [-1, 0]:
                    print('| step {:5} |  lr: {:9.7f} | total loss {:7.4f} | ms/batch {:7.2f} |'.format(
                        global_step, scheduler.get_last_lr()[0], total_loss_acm / n_samples_acm, ms_per_batch))
                    wandb.log({"lr": scheduler.get_last_lr()[0],
                               "train_loss": total_loss_acm / n_samples_acm,
                               "train_link_loss": link_loss_acm / n_samples_acm,
                               "moe_loss_acm": moe_loss_acm / n_samples_acm,
                               }, step=global_step)

                total_loss_acm = end_loss_acm = mlm_loss_acm = 0.0
                link_loss_acm = pos_link_loss_acm = neg_link_loss_acm = 0.0
                moe_loss_acm = 0.0
                n_samples_acm = n_corrects_acm = 0
                total_time = 0
            global_step += 1  # Number of batches processed up to now

            # break

        # Save checkpoints and evaluate after every epoch
        if args.local_rank in [-1, 0]:
            model.eval()

            _, _, train_acc = calc_eval_accuracy(args, train_eval_dataloader, model)
            print('train_acc', train_acc)

            # calculate dev_acc
            _, _, dev_acc = calc_eval_accuracy(args, dev_dataloader, model)
            print('dev_acc', dev_acc)

            # calculate test_acc
            if has_test_split:
                _, _, test_acc = calc_eval_accuracy(args, test_dataloader, model, cur_save_model_wrong_prediction=args.save_model_wrong_prediction, epoch_id=epoch_id)
                print('test_acc', test_acc)
            else:
                test_acc = 0

            print('-' * 71)
            print('| epoch {:3} | step {:5} | dev_acc {:7.4f} | test_acc {:7.4f} |'.format(
                epoch_id, global_step, dev_acc, test_acc))
            print('-' * 71)

            if dev_acc >= best_dev_acc:
                best_dev_acc = dev_acc
                final_test_acc = test_acc
                best_dev_epoch = epoch_id
            if not args.debug:
                with open(log_path, 'a') as fout:
                    fout.write('{:3},{:5},{:7.4f},{:7.4f},{:7.4f},{:7.4f},{:3}\n'.format(
                        epoch_id, global_step, dev_acc, test_acc, best_dev_acc, final_test_acc, best_dev_epoch))

            if args.use_wandb:
                wandb.log({"epoch_id": epoch_id, "train_acc": train_acc, "dev_acc": dev_acc, "best_dev_acc": best_dev_acc, "best_dev_epoch": best_dev_epoch}, step=global_step)
            if has_test_split:
                if args.use_wandb:
                    wandb.log({"test_acc": test_acc, "final_test_acc": final_test_acc}, step=global_step)
                if args.use_codalab:
                    with open("stats.json", 'w') as fout:
                        json.dump({'epoch': epoch_id, 'step': global_step,
                                  'dev_acc': dev_acc, 'test_acc': test_acc}, fout, indent=2)

            # Save the model checkpoint
            if (args.save_model == 2) or ((args.save_model == 1) and (best_dev_epoch == epoch_id)):
                if args.local_rank != -1:
                    model_state_dict = model.module.state_dict()
                else:
                    model_state_dict = model.state_dict()
                try:
                    del model_state_dict["lmgnn.concept_emb.emb.weight"]
                except:
                    pass
                checkpoint = {"model": model_state_dict, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(
                ), "epoch": epoch_id, "global_step": global_step, "best_dev_epoch": best_dev_epoch, "best_dev_acc": best_dev_acc, "final_test_acc": final_test_acc, "config": args}
                print('Saving model to {}.{}'.format(model_path, epoch_id))
                torch.save(checkpoint, model_path + ".{}".format(epoch_id))

        model.train()
        start_time = time.time()
        if epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
            if args.local_rank in [-1, 0]:
                break

        if args.debug:
            break


def evaluate(args, has_test_split, devices, kg, only_test_set=False):
    train_statements = args.train_statements
    dev_statements = args.dev_statements
    test_statements = args.test_statements
    train_adj = args.train_adj
    dev_adj = args.dev_adj
    test_adj = args.test_adj
    debug = args.debug
    inhouse = args.inhouse

    args.train_statements = train_statements
    args.dev_statements = dev_statements
    args.test_statements = test_statements
    args.train_adj = train_adj
    args.dev_adj = dev_adj
    args.test_adj = test_adj
    args.inhouse = inhouse

    dataset = load_data(args, devices, kg)
    dev_dataloader = dataset.dev()
    if has_test_split:
        test_dataloader = dataset.test()
    model = construct_model(args, kg, dataset)

    model.eval()

    print('inhouse?', args.inhouse)
    print('args.train_statements', args.train_statements)
    print('args.dev_statements', args.dev_statements)
    print('args.test_statements', args.test_statements)
    print('args.train_adj', args.train_adj)
    print('args.dev_adj', args.dev_adj)
    print('args.test_adj', args.test_adj)

    model.eval()

    if not only_test_set:
        # Evaluation on the dev set
        _, _, dev_acc = calc_eval_accuracy(args, dev_dataloader, model)
        print('dev_acc: ', dev_acc)

    # Evaluation on the test set
    if has_test_split:
        _, _, test_acc = calc_eval_accuracy(args, test_dataloader, model)
    else:
        test_acc = 0
    print('test_acc: ', test_acc)

    print('-' * 71)
    if not only_test_set:
        print('dev_acc {:7.4f}, test_acc {:7.4f}'.format(dev_acc, test_acc))
    else:
        print('test_acc {:7.4f}'.format(test_acc))
    print('-' * 71)


def get_devices(args):
    """Get the devices to put the data and the model based on whether to use GPUs and, if so, how many of them are available."""
    if args.local_rank == -1 or not args.cuda:
        if torch.cuda.device_count() >= 2 and args.cuda:
            device0 = torch.device("cuda:0")
            # device1 = torch.device("cuda:1")
            device1 = torch.device("cuda:0")
            print("device0: {}, device1: {}".format(device0, device1))
        elif torch.cuda.device_count() == 1 and args.cuda:
            device0 = torch.device("cuda:0")
            device1 = torch.device("cuda:0")
        else:
            device0 = torch.device("cpu")
            device1 = torch.device("cpu")
    else:
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device0 = torch.device("cuda", args.local_rank)
        device1 = device0
        torch.distributed.init_process_group(backend="nccl")

    args.world_size = world_size = torch.distributed.get_world_size(
    ) if args.local_rank != -1 else 1
    print("Process rank: %s, device: %s, distributed training: %s, world_size: %s" %
          (args.local_rank,
           device0,
           bool(args.local_rank != -1),
           world_size), file=sys.stderr)

    return device0, device1


def main(args):
    print('start ...')
    print('args.batch_size: ', args.batch_size)
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(name)s:%(funcName)s():%(lineno)d] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.WARNING)

    has_test_split = True
    devices = get_devices(args)

    if not args.use_wandb:
        wandb_mode = "disabled"
    elif args.debug:
        wandb_mode = "offline"
    else:
        wandb_mode = "online"

    # We can optionally resume training from a checkpoint. If doing so, also set the `resume_id` so that you resume your previous wandb run instead of creating a new one.
    resume = args.resume_checkpoint not in [None, "None"]

    args.hf_version = transformers.__version__

    if args.local_rank in [-1, 0]:
        wandb_id = args.resume_id if resume and (
            args.resume_id not in [None, "None"]) else wandb.util.generate_id()
        args.wandb_id = wandb_id

        wandb.init(project="GNP", entity="meettyj", config=args, name=args.run_name,
                   resume="allow", id=wandb_id, settings=wandb.Settings(start_method="fork"), mode=wandb_mode)
        print("pid:", os.getpid())
        print("conda env:", os.environ.get('CONDA_DEFAULT_ENV'))
        print("screen: %s" % subprocess.check_output(
            'echo $STY', shell=True).decode('utf'))
        print("gpu: %s" % subprocess.check_output(
            'echo $CUDA_VISIBLE_DEVICES', shell=True).decode('utf'))
        utils.print_cuda_info()
        print("wandb id: ", wandb_id)
        wandb.run.log_code('.')

    kg = args.kg
    if args.dataset == "medqa_usmle":
        kg = "ddb"
    elif args.dataset in ["medqa", "pubmedqa", "bioasq"]:
        kg = "umls"
    print("KG used:", kg)
    print("KG used:", kg, file=sys.stderr)

    if args.mode == 'train':
        train(args, resume, has_test_split, devices, kg)
    elif "eval" in args.mode:
        assert args.world_size == 1, "DDP is only implemented for training"
        evaluate(args, has_test_split, devices, kg, only_test_set=True)
    else:
        raise ValueError('Invalid mode')


def add_data_arguments(parser):
    DATASET_SETTING = {
        'csqa': 'inhouse',
        'obqa': 'official',
    }
    DATASET_NO_TEST = []

    # dataset specific
    parser.add_argument('-ds', '--dataset', default='csqa', help='dataset name')
    parser.add_argument('--data_dir', default='data', type=str, help='Path to the data directory')
    parser.add_argument('-ih', '--inhouse', type=utils.bool_flag, nargs='?', const=True, help='run in-house setting')
    parser.add_argument('--inhouse_train_qids', default='data/{dataset}/inhouse_split_qids.txt', help='qids of the in-house training set')
    # statements
    parser.add_argument('--train_statements', default='{data_dir}/{dataset}/statement/train.statement.jsonl')
    parser.add_argument('--dev_statements', default='{data_dir}/{dataset}/statement/dev.statement.jsonl')
    parser.add_argument('--test_statements', default='{data_dir}/{dataset}/statement/test.statement.jsonl')
    # preprocessing options
    parser.add_argument('-sl', '--max_seq_len', default=100, type=int)
    parser.add_argument('--graph_max_seq_len', default=200, type=int, help='max sequence length for the graph of each answer choice')
    # set dataset defaults
    args, _ = parser.parse_known_args()
    parser.set_defaults(inhouse=(DATASET_SETTING.get(args.dataset, "IH") == 'inhouse'),
                        inhouse_train_qids=args.inhouse_train_qids.format(dataset=args.dataset))
    data_splits = ('train', 'dev') if args.dataset in DATASET_NO_TEST else ('train', 'dev', 'test')
    for split in data_splits:
        for attribute in ('statements',):
            attr_name = f'{split}_{attribute}'
            parser.set_defaults(**{attr_name: getattr(args, attr_name).format(dataset=args.dataset, data_dir=args.data_dir)})
    if 'test' not in data_splits:
        parser.set_defaults(test_statements=None)


def add_encoder_arguments(parser):
    parser.add_argument('-enc', '--encoder', default='bert-large-uncased', help='encoder type')
    parser.add_argument('--encoder_load_path', default='', help='custom encoder to load')
    parser.add_argument('--encoder_layer', default=-1, type=int, help='encoder layer ID to use as features (used only by non-LSTM encoders)')
    parser.add_argument('-lr', '--learning_rate', default=2e-5, type=float, help='learning rate')
    args, _ = parser.parse_known_args()


def add_optimization_arguments(parser):
    parser.add_argument('--loss', default='cross_entropy', choices=['margin_rank', 'cross_entropy'], help='model type')
    parser.add_argument('--optim', default='radam', choices=['sgd', 'adam', 'adamw', 'radam'], help='learning rate scheduler')
    parser.add_argument('--lr_schedule', default='fixed', choices=['fixed', 'warmup_linear', 'warmup_constant'], help='learning rate scheduler')
    parser.add_argument('-bs', '--batch_size', default=32, type=int)
    parser.add_argument('--warmup_steps', type=float, default=150)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max grad norm (0 to disable)')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='l2 weight decay strength')
    parser.add_argument('--n_epochs', default=100, type=int, help='total number of training epochs to perform.')
    parser.add_argument('-me', '--max_epochs_before_stop', default=10, type=int, help='stop training if dev does not increase for N epochs')


def add_additional_arguments(parser):
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--cuda', default=True, type=utils.bool_flag, nargs='?', const=True, help='use GPU')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--debug', default=False, type=utils.bool_flag, nargs='?', const=True, help='run in debug mode')
    args, _ = parser.parse_known_args()
    if args.debug:
        parser.set_defaults(batch_size=1, log_interval=1, eval_interval=5)


def get_parser():
    """A helper function that handles the arguments that all models share"""
    parser = argparse.ArgumentParser(add_help=False)
    add_data_arguments(parser)
    add_encoder_arguments(parser)
    add_optimization_arguments(parser)
    add_additional_arguments(parser)
    return parser


if __name__ == '__main__':
    __spec__ = None
    parser = get_parser()
    args, _ = parser.parse_known_args()

    # General
    parser.add_argument('--mode', default='train', choices=['train', 'eval'], help='run training or evaluation')
    parser.add_argument('--use_codalab', default=0, type=int, help='using codalab or not')
    parser.add_argument('--save_dir', default=f'./saved_models/', help='model output directory')
    parser.add_argument('--save_model', default=2, type=float, help="0: do not save model checkpoints. 1: save if best dev. 2: save always")
    # parser.add_argument('--load_model_path', default=None, help="The model checkpoint to load in the evaluation mode.")
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
    parser.add_argument("--run_name", required=True, type=str, help="The name of this experiment run.")
    parser.add_argument("--resume_checkpoint", default=None, type=str,
                        help="The checkpoint to resume training from.")
    parser.add_argument('--use_wandb', default=False, type=utils.bool_flag, help="Whether to use wandb or not.")
    parser.add_argument("--resume_id", default=None, type=str, help="The wandb run id to resume if `resume_checkpoint` is not None or 'None'.")
    parser.add_argument("--dump_graph_cache", default=True, type=utils.bool_flag)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--world_size", type=int, default=1, help="For distributed training: world_size")
    parser.add_argument("--data_loader_one_process_at_a_time", default=False, type=utils.bool_flag)

    # Task
    parser.add_argument('--end_task', type=float, default=1.0, help='Task weight for the end task (MCQA)')
    parser.add_argument('--mlm_task', type=float, default=0.0, help='Task weight for the MLM task')
    parser.add_argument('--link_task', type=float, default=0.0, help='Task weight for the LinkPred task')

    # Adjunct loss
    parser.add_argument('--span_mask', type=utils.bool_flag, default=False, help='')
    parser.add_argument('--link_drop_max_count', type=int, default=100, help='To specify #target positive triples for LinkPred')
    parser.add_argument('--link_drop_probability', type=float, default=0.2, help='To specify #target positive triples for LinkPred')
    parser.add_argument('--link_drop_probability_in_which_keep', type=float, default=0.2, help='Within target positive triples, how much to keep in the input graph?')
    parser.add_argument('--link_negative_sample_size', type=int, default=64, help='')
    parser.add_argument('--link_negative_adversarial_sampling', type=utils.bool_flag, default=True, help='')
    parser.add_argument('--link_negative_adversarial_sampling_temperature', type=float, default=1, help='')
    parser.add_argument('--link_regularizer_weight', type=float, default=0.01, help='')
    parser.add_argument('--link_normalize_headtail', type=int, default=0, help='')
    parser.add_argument('--link_proj_headtail', type=utils.bool_flag, default=False, help='')
    parser.add_argument('--scaled_distmult', type=utils.bool_flag, default=False, help='')
    parser.add_argument('--link_gamma', type=float, default=12, help='')
    parser.add_argument('--link_decoder', type=str, default="DistMult", help='')

    # Data
    parser.add_argument('--kg', default='cpnet', help="What KG to use.")
    parser.add_argument('--max_num_relation', default=-1, type=int, help="max number of KG relation types to keep.")
    parser.add_argument('--kg_only_use_qa_nodes', default=False, type=utils.bool_flag, help="")

    parser.add_argument('--train_adj', default=f'{args.data_dir}/{args.dataset}/graph/train.graph.adj.pk', help="The path to the retrieved KG subgraphs of the training set.")
    parser.add_argument('--dev_adj', default=f'{args.data_dir}/{args.dataset}/graph/dev.graph.adj.pk', help="The path to the retrieved KG subgraphs of the dev set.")
    parser.add_argument('--test_adj', default=f'{args.data_dir}/{args.dataset}/graph/test.graph.adj.pk', help="The path to the retrieved KG subgraphs of the test set.")
    parser.add_argument('--max_node_num', default=200, type=int, help="Max number of nodes / the threshold used to prune nodes.")
    parser.add_argument('--subsample', default=1.0, type=float, help="The ratio to subsample the training set.")
    parser.add_argument('--n_train', default=-1, type=int, help="Number of training examples to use. Setting it to -1 means using the `subsample` argument to determine the training set size instead; otherwise it will override the `subsample` argument.")

    # Model architecture
    parser.add_argument('--gnn_layers', default=3, type=int, help='The number of GNN layers')
    parser.add_argument('--n_attention_head', default=2, type=int, help='number of attention heads of the final graph nodes\' pooling')
    parser.add_argument('--gnn_dim', default=100, type=int, help='dimension of the GNN layers')
    parser.add_argument('--fc_dim', default=200, type=int, help='number of FC hidden units (except for the MInt operators)')
    parser.add_argument('--fc_layer_num', default=0, type=int, help='number of hidden layers of the final MLP')
    parser.add_argument('--freeze_ent_emb', default=True, type=utils.bool_flag, help='Whether to freeze the entity embedding layer.')
    parser.add_argument('--info_exchange', default=True, choices=[True, False, "every-other-layer"], type=utils.bool_str_flag, help="Whether we have the MInt operator in every Fusion layer or every other Fusion layer or not at all.")
    parser.add_argument('--ie_layer_num', default=1, type=int, help='number of hidden layers in the MInt operator')
    parser.add_argument("--sep_ie_layers", default=False, type=utils.bool_flag, help="Whether to share parameters across the MInt ops across differernt Fusion layers or not. Setting it to `False` means sharing.")
    parser.add_argument('--random_ent_emb', default=False, type=utils.bool_flag, nargs='?', const=True, help='Whether to use randomly initialized learnable entity embeddings or not.')
    parser.add_argument("--cxt_node_connects_all", default=False, type=utils.bool_flag, help="Whether to connect the interaction node to all the retrieved KG nodes or only the linked nodes.")
    parser.add_argument('--no_node_score', default=True, type=utils.bool_flag, help='Don\'t use node score.')

    # Regularization
    parser.add_argument('--p_dropout_gnn', type=float, default=0.2, help='dropout for GNN layers')

    # Optimization
    parser.add_argument('-mbs', '--mini_batch_size', default=1, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=1, type=int)
    parser.add_argument('--refreeze_epoch', default=10000, type=int)
    parser.add_argument('--init_range', default=0.02, type=float, help='stddev when initializing with normal distribution')
    parser.add_argument('--fp16', default=False, type=utils.bool_flag, help='use fp16 training. this requires torch>=1.6.0')
    parser.add_argument('--upcast', default=False, type=utils.bool_flag, help='Upcast attention computation during fp16 training')
    parser.add_argument('--redef_epoch_steps', default=-1, type=int)

    # added
    parser.add_argument("--load_llm_cache", default=True, type=utils.bool_flag)
    parser.add_argument("--load_graph_cache", default=True, type=utils.bool_flag)

    # prompt
    parser.add_argument('--prompt', default=False, choices=[False, "regular", "gnn"], type=utils.bool_str_flag, help="What prompt we use")
    parser.add_argument('--lora', default=False, type=utils.bool_flag, help='Use LoRA or not')
    parser.add_argument('--num_virtual_tokens', default=0, type=int, help='The number of prompt tokens')

    # cross modality
    parser.add_argument('--cross_modality_layers', default=0, type=int, help='The number of cross modality component layers')
    parser.add_argument('--use_cross_modality_for_text', default=False, type=utils.bool_flag, help='whether to use cross modality component for text or not')
    parser.add_argument('--cross_modality_for_text_layers', default=1, type=int, help='The number of cross modality component layers')

    # dataset-level prompt
    parser.add_argument('--dataset_level_prompt', default=False, type=utils.bool_flag, help='whether to use dataset-level prompt with GNN prompt or not')

    parser.add_argument('--debug2', default=False, type=utils.bool_flag, nargs='?', const=True, help='run in debug mode')

    # baseline
    parser.add_argument('--baseline_flattern', default=False, choices=[False, "rs", "bfs", "triplet_a", "triplet_q", "triplet_qa", "every_triplet"], type=utils.bool_str_flag, help="get results for baseline flattern")
    parser.add_argument('--baseline_flattern_similarity', default=True, type=utils.bool_flag, help='Use similarity to sort triplets.')

    # prompt design
    parser.add_argument('--prompt_design', default=False, choices=[False, "prompt_A", "prompt_B", "prompt_C"], type=utils.bool_str_flag, help="whether and which prompt design we use.")

    # relational information
    parser.add_argument('--use_relational_gnn', default=False, type=utils.bool_flag, help='Use customized relational gnn or not.')

    # mixture-of-experts
    parser.add_argument('--moe_experts', default=0, type=int, help='The number of moe experts')
    parser.add_argument('--moe_loss_weight', type=float, default=0.0, help='loss weight for MOE')

    # no projector
    parser.add_argument('--no_projector', default=False, type=utils.bool_flag, help='shall we use projector or not: ablation study')

    # case study for visualization
    parser.add_argument('--case_study', default=False, type=utils.bool_flag, help='save all the triplets for case study and graph visualization.')
    parser.add_argument('--save_model_wrong_prediction', default=False, type=utils.bool_flag, help='save model wrong prediction for case study.')

    args = parser.parse_args()
    args.fp16 = args.fp16 and (torch.__version__ >= '1.6.0')
    if args.local_rank != -1:
        assert not args.dump_graph_cache

    if args.case_study:
        args.baseline_flattern = 'every_triplet'
        args.debug2 = True

    if args.baseline_flattern:
        args.mode = 'eval'
        args.batch_size = 1

    if args.prompt_design:
        args.mode = 'eval'
        args.load_llm_cache = False
        args.load_graph_cache = False
        args.dump_graph_cache = False

    if args.mode == 'eval':
        args.prompt = False
    args.mini_batch_size = args.batch_size
    args.eval_batch_size = args.batch_size

    # sagemaker
    if 'SM_HPS' in os.environ:
        sagemaker_hps = json.loads(os.environ['SM_HPS'])
        if 'on_sagemaker' in sagemaker_hps:
            args.on_sagemaker = sagemaker_hps['on_sagemaker']
        else:
            args.on_sagemaker = False
        print('on_sagemaker: ', args.on_sagemaker)
        print('-'*300)

        if args.on_sagemaker:
            args.data_dir = os.environ["SM_INPUT_DIR"] + '/data/training/data'

            args.train_adj = args.data_dir + f'/{args.dataset}/graph/train.graph.adj.pk'
            args.dev_adj = args.data_dir + f'/{args.dataset}/graph/dev.graph.adj.pk'
            args.test_adj = args.data_dir + f'/{args.dataset}/graph/test.graph.adj.pk'
            args.inhouse_train_qids = args.data_dir + f'/{args.dataset}/inhouse_split_qids.txt'
            args.train_statements = args.data_dir + f'/{args.dataset}/statement/train.statement.jsonl'
            args.dev_statements = args.data_dir + f'/{args.dataset}/statement/dev.statement.jsonl'
            args.test_statements = args.data_dir + f'/{args.dataset}/statement/test.statement.jsonl'

            args.load_llm_cache = False
            args.load_graph_cache = False

            args.use_wandb = True
            args.mode = 'train'
            if args.baseline_flattern:
                args.mode = 'eval'
                args.batch_size = 1
                args.prompt = False
                args.mini_batch_size = args.batch_size
                args.eval_batch_size = args.batch_size

    # set path for some data
    if args.dataset not in ['medqa', 'pubmedqa', 'bioasq']:
        args.ent_emb_paths = [args.data_dir + '/cpnet/tzw.ent.npy']
        args.kg_vocab_path = args.data_dir + '/cpnet/concept.txt'
    else:
        args.ent_emb_paths = [args.data_dir + '/umls/ent_emb_blbertL.npy']
        args.kg_vocab_path = args.data_dir + '/umls/concepts.txt'

    # do not drop link if not link prediction task
    if not args.link_task:
        args.link_drop_max_count = 0
        args.link_drop_probability = 0
        args.link_drop_probability_in_which_keep = 0

    # set random seed
    set_random_seed(args.seed)

    main(args)
