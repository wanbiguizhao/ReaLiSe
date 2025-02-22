from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import math
import os
import sys

import opencc
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.modeling_bert import *
from .helper import pho_convertor, pho2_convertor
from copy import deepcopy
from PIL import ImageFont
import numpy as np

from .char_cnn import CharResNet, CharResNet1

def _is_chinese_char(cp):
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True
    return False

class SpellBert(BertPreTrainedModel):
    def __init__(self, config):
        super(SpellBert, self).__init__(config)

        self.vocab_size = config.vocab_size
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)

        self.init_weights()

    def tie_cls_weight(self):
        self.classifier.weight = self.bert.embeddings.word_embeddings.weight

    @staticmethod
    def build_batch(batch, tokenizer):
        return batch

    def forward(self, batch):
        input_ids = batch['src_idx']
        attention_mask = batch['masks']
        loss_mask = batch['loss_masks']
        label_ids = batch['tgt_idx'] if 'tgt_idx' in batch else None

        outputs = self.bert(input_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if label_ids is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            active_loss = loss_mask.view(-1) == 1
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = label_ids.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            outputs = (loss,) + outputs

        return outputs 

class SpellBertPho1(BertPreTrainedModel):
    def __init__(self, config):
        super(SpellBertPho1, self).__init__(config)

        self.vocab_size = config.vocab_size
        self.bert = BertModel(config)

        self.pho_embeddings = nn.Embedding(pho_convertor.get_pho_size(), config.hidden_size, padding_idx=0)
        pho_config = deepcopy(config)
        pho_config.num_hidden_layers = 4
        self.pho_model = BertModel(pho_config)

        self.integrate = nn.Linear(2*config.hidden_size, config.hidden_size)
        out_config = deepcopy(config)
        out_config.num_hidden_layers = 2
        self.output_block = BertModel(out_config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)

        self.init_weights()

    def tie_cls_weight(self):
        self.classifier.weight = self.bert.embeddings.word_embeddings.weight

    @staticmethod
    def build_batch(batch, tokenizer):
        input_shape = batch['src_idx'].size()
        pho_idx_1, pho_idx_2, pho_idx_3 = [], [], []
        for i in range(input_shape[0]):
            pho1 = [0] * input_shape[1]
            pho2 = [0] * input_shape[1]
            pho3 = [0] * input_shape[1]
            token_ids = batch['src_idx'][i][1:batch['lengths'][i]+1].numpy().tolist()
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            
            for p, (v1, v2, v3) in zip(range(1, len(tokens)+1), pho_convertor.convert(tokens)):
                pho1[p] = v1
                pho2[p] = v2
                pho3[p] = v3
            pho_idx_1.append(pho1)
            pho_idx_2.append(pho2)
            pho_idx_3.append(pho3)
            
        batch['pho_idx_1'] = torch.tensor(pho_idx_1, dtype=torch.long)
        batch['pho_idx_2'] = torch.tensor(pho_idx_2, dtype=torch.long)
        batch['pho_idx_3'] = torch.tensor(pho_idx_3, dtype=torch.long)
        return batch

    def forward(self, batch):
        input_ids = batch['src_idx']
        attention_mask = batch['masks']
        loss_mask = batch['loss_masks']
        pho_idx_1 = batch['pho_idx_1']
        pho_idx_2 = batch['pho_idx_2']
        pho_idx_3 = batch['pho_idx_3']
        label_ids = batch['tgt_idx'] if 'tgt_idx' in batch else None

        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)[0]
        
        pho_embeddings = self.pho_embeddings(pho_idx_1)
        pho_embeddings += self.pho_embeddings(pho_idx_2)
        pho_embeddings += self.pho_embeddings(pho_idx_3)
        pho_outputs = self.pho_model(inputs_embeds=pho_embeddings, attention_mask=attention_mask)[0]

        concated_outputs = torch.cat((bert_outputs, pho_outputs), dim=-1)
        concated_outputs = self.integrate(concated_outputs)

        outputs = self.output_block(inputs_embeds=concated_outputs,
                    position_ids=torch.zeros(input_ids.size(), dtype=torch.long, device=input_ids.device),
                    attention_mask=attention_mask)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if label_ids is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            active_loss = loss_mask.view(-1) == 1
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = label_ids.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            outputs = (loss,) + outputs
        return outputs 

class SpellBertPho2(BertPreTrainedModel):
    def __init__(self, config):
        super(SpellBertPho2, self).__init__(config)

        self.vocab_size = config.vocab_size
        self.bert = BertModel(config)

        self.pho_embeddings = nn.Embedding(pho2_convertor.get_pho_size(), config.hidden_size, padding_idx=0)
        self.pho_gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )
        pho_config = deepcopy(config)
        pho_config.num_hidden_layers = 4
        self.pho_model = BertModel(pho_config)

        self.integrate = nn.Linear(2*config.hidden_size, config.hidden_size)
        out_config = deepcopy(config)
        out_config.num_hidden_layers = 2
        self.output_block = BertModel(out_config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)

        self.init_weights()

    def tie_cls_weight(self):
        self.classifier.weight = self.bert.embeddings.word_embeddings.weight

    @staticmethod
    def build_batch(batch, tokenizer):
        src_idx = batch['src_idx'].flatten().tolist()
        chars = tokenizer.convert_ids_to_tokens(src_idx)
        pho_idx, pho_lens = pho2_convertor.convert(chars)
        batch['pho_idx'] = pho_idx
        batch['pho_lens'] = pho_lens
        return batch

    def forward(self, batch):
        input_ids = batch['src_idx']
        attention_mask = batch['masks']
        loss_mask = batch['loss_masks']
        pho_idx = batch['pho_idx']
        pho_lens = batch['pho_lens']
        label_ids = batch['tgt_idx'] if 'tgt_idx' in batch else None

        input_shape = input_ids.size()

        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)[0]
        
        pho_embeddings = self.pho_embeddings(pho_idx)
        pho_embeddings = torch.nn.utils.rnn.pack_padded_sequence(
            input=pho_embeddings,
            lengths=pho_lens,
            batch_first=True,
            enforce_sorted=False,
        )
        _, pho_hiddens = self.pho_gru(pho_embeddings)
        pho_hiddens = pho_hiddens.squeeze(0).reshape(input_shape[0], input_shape[1], -1).contiguous()
        pho_outputs = self.pho_model(inputs_embeds=pho_hiddens, attention_mask=attention_mask)[0]

        concated_outputs = torch.cat((bert_outputs, pho_outputs), dim=-1)
        concated_outputs = self.integrate(concated_outputs)

        outputs = self.output_block(inputs_embeds=concated_outputs,
                    position_ids=torch.zeros(input_ids.size(), dtype=torch.long, device=input_ids.device),
                    attention_mask=attention_mask)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if label_ids is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            active_loss = loss_mask.view(-1) == 1
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = label_ids.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            outputs = (loss,) + outputs
        return outputs 

class SpellBertPho1Res(BertPreTrainedModel):
    def __init__(self, config):
        super(SpellBertPho1Res, self).__init__(config)

        self.vocab_size = config.vocab_size
        self.bert = BertModel(config)

        self.char_images = nn.Embedding(config.vocab_size, 1024)
        self.char_images.weight.requires_grad = False

        self.pho_embeddings = nn.Embedding(pho_convertor.get_pho_size(), config.hidden_size, padding_idx=0)
        self.resnet = CharResNet()
        pho_res_config = deepcopy(config)
        pho_res_config.num_hidden_layers = 4
        self.pho_res_model = BertModel(pho_res_config)

        self.integrate = nn.Linear(2*config.hidden_size, config.hidden_size)
        out_config = deepcopy(config)
        out_config.num_hidden_layers = 2
        self.output_block = BertModel(out_config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)

        self.init_weights()

    def tie_cls_weight(self):
        self.classifier.weight = self.bert.embeddings.word_embeddings.weight

    def build_glyce_embed(self, vocab_dir, font_path, font_size=32):
        # 类似于初始化，把图片转换成向量。
        vocab_path = os.path.join(vocab_dir, 'vocab.txt')
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = [s.strip() for s in f]

        font = ImageFont.truetype(font_path, size=font_size)# 加载字体，font_size指定画出来字体的大小。

        char_images = []
        for char in vocab:
            if len(char) != 1 or (not _is_chinese_char(ord(char))):
                char_images.append(np.zeros((font_size, font_size)).astype(np.float32))# 这里直接填充零了。
                continue
            image = font.getmask(char)#得到一张图片
            image = np.asarray(image).astype(np.float32).reshape(image.size[::-1])  # Must be [::-1]# 这块应该是格式问题，CHW，还是HWC的样式

            # Crop
            image = image[:font_size, :font_size]# 做图片的裁剪

            # Pad
            if image.size != (font_size, font_size):
                back_image = np.zeros((font_size, font_size)).astype(np.float32)
                offset0 = (font_size - image.shape[0]) // 2
                offset1 = (font_size - image.shape[1]) // 2
                back_image[offset0:offset0 + image.shape[0], offset1:offset1 + image.shape[1]] = image
                image = back_image

            char_images.append(image)
        char_images = np.array(char_images)
        char_images = (char_images - np.mean(char_images)) / np.std(char_images)# 做了图片的归一化操作。
        char_images = torch.from_numpy(char_images).reshape(char_images.shape[0], -1)
        assert char_images.shape == (21128, 1024)# 21128 是bert标准的vocabulary size，1024是32*32，一张图片的大小
        self.char_images.weight.data.copy_(char_images)# 把生成图片的每一个像素都记录下来了。

    @staticmethod
    def build_batch(batch, tokenizer):
        input_shape = batch['src_idx'].size()
        pho_idx_1, pho_idx_2, pho_idx_3 = [], [], []
        for i in range(input_shape[0]):
            pho1 = [0] * input_shape[1]
            pho2 = [0] * input_shape[1]
            pho3 = [0] * input_shape[1]
            token_ids = batch['src_idx'][i][1:batch['lengths'][i]+1].numpy().tolist()
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            
            for p, (v1, v2, v3) in zip(range(1, len(tokens)+1), pho_convertor.convert(tokens)):
                pho1[p] = v1
                pho2[p] = v2
                pho3[p] = v3
            pho_idx_1.append(pho1)
            pho_idx_2.append(pho2)
            pho_idx_3.append(pho3)
            
        batch['pho_idx_1'] = torch.tensor(pho_idx_1, dtype=torch.long)
        batch['pho_idx_2'] = torch.tensor(pho_idx_2, dtype=torch.long)
        batch['pho_idx_3'] = torch.tensor(pho_idx_3, dtype=torch.long)
        return batch

    def forward(self, batch):
        input_ids = batch['src_idx']
        attention_mask = batch['masks']
        loss_mask = batch['loss_masks']
        pho_idx_1 = batch['pho_idx_1']
        pho_idx_2 = batch['pho_idx_2']
        pho_idx_3 = batch['pho_idx_3']
        label_ids = batch['tgt_idx'] if 'tgt_idx' in batch else None

        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)[0]
        
        pho_embeddings = self.pho_embeddings(pho_idx_1)
        pho_embeddings += self.pho_embeddings(pho_idx_2)
        pho_embeddings += self.pho_embeddings(pho_idx_3)
        
        src_idxs = input_ids.view(-1)
        images = self.char_images(src_idxs).reshape(src_idxs.shape[0], 1, 32, 32).contiguous()
        res_embeddings = self.resnet(images)
        res_embeddings = res_embeddings.reshape(input_ids.shape[0], input_ids.shape[1], -1).contiguous()
        pho_res_embeddings = pho_embeddings + res_embeddings
        pho_res_outputs = self.pho_res_model(inputs_embeds=pho_res_embeddings, attention_mask=attention_mask)[0]

        concated_outputs = torch.cat((bert_outputs, pho_res_outputs), dim=-1)
        concated_outputs = self.integrate(concated_outputs)

        outputs = self.output_block(inputs_embeds=concated_outputs,
                    position_ids=torch.zeros(input_ids.size(), dtype=torch.long, device=input_ids.device),
                    attention_mask=attention_mask)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if label_ids is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            active_loss = loss_mask.view(-1) == 1
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = label_ids.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            outputs = (loss,) + outputs
        return outputs

class SpellBertPho2Res(BertPreTrainedModel):
    def __init__(self, config):
        super(SpellBertPho2Res, self).__init__(config)

        self.vocab_size = config.vocab_size
        self.bert = BertModel(config)

        self.char_images = nn.Embedding(config.vocab_size, 1024)
        self.char_images.weight.requires_grad = False

        self.pho_embeddings = nn.Embedding(pho2_convertor.get_pho_size(), config.hidden_size, padding_idx=0)
        self.pho_gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )
        self.resnet = CharResNet()
        pho_res_config = deepcopy(config)
        pho_res_config.num_hidden_layers = 4
        self.pho_res_model = BertModel(pho_res_config)

        self.integrate = nn.Linear(2*config.hidden_size, config.hidden_size)
        out_config = deepcopy(config)
        out_config.num_hidden_layers = 2
        self.output_block = BertModel(out_config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)

        self.init_weights()

    def tie_cls_weight(self):
        self.classifier.weight = self.bert.embeddings.word_embeddings.weight

    def build_glyce_embed(self, vocab_dir, font_path, font_size=32):
        vocab_path = os.path.join(vocab_dir, 'vocab.txt')
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = [s.strip() for s in f]

        font = ImageFont.truetype(font_path, size=font_size)

        char_images = []
        for char in vocab:
            if len(char) != 1 or (not _is_chinese_char(ord(char))):
                char_images.append(np.zeros((font_size, font_size)).astype(np.float32))
                continue
            image = font.getmask(char)
            image = np.asarray(image).astype(np.float32).reshape(image.size[::-1])  # Must be [::-1]

            # Crop
            image = image[:font_size, :font_size]

            # Pad
            if image.size != (font_size, font_size):
                back_image = np.zeros((font_size, font_size)).astype(np.float32)
                offset0 = (font_size - image.shape[0]) // 2
                offset1 = (font_size - image.shape[1]) // 2
                back_image[offset0:offset0 + image.shape[0], offset1:offset1 + image.shape[1]] = image
                image = back_image

            char_images.append(image)
        char_images = np.array(char_images)
        char_images = (char_images - np.mean(char_images)) / np.std(char_images)
        char_images = torch.from_numpy(char_images).reshape(char_images.shape[0], -1)
        assert char_images.shape == (21128, 1024)
        self.char_images.weight.data.copy_(char_images)

    @staticmethod
    def build_batch(batch, tokenizer):
        src_idx = batch['src_idx'].flatten().tolist()
        chars = tokenizer.convert_ids_to_tokens(src_idx)
        pho_idx, pho_lens = pho2_convertor.convert(chars)
        batch['pho_idx'] = pho_idx
        batch['pho_lens'] = pho_lens
        return batch

    def forward(self, batch):
        input_ids = batch['src_idx']
        attention_mask = batch['masks']
        loss_mask = batch['loss_masks']
        pho_idx = batch['pho_idx']
        pho_lens = batch['pho_lens']
        label_ids = batch['tgt_idx'] if 'tgt_idx' in batch else None

        input_shape = input_ids.size()

        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)[0]
        
        pho_embeddings = self.pho_embeddings(pho_idx)
        pho_embeddings = torch.nn.utils.rnn.pack_padded_sequence(
            input=pho_embeddings,
            lengths=pho_lens,
            batch_first=True,
            enforce_sorted=False,
        )
        _, pho_hiddens = self.pho_gru(pho_embeddings)
        pho_hiddens = pho_hiddens.squeeze(0).reshape(input_shape[0], input_shape[1], -1).contiguous()
        
        src_idxs = input_ids.view(-1)
        images = self.char_images(src_idxs).reshape(src_idxs.shape[0], 1, 32, 32).contiguous()
        res_hiddens = self.resnet(images)
        res_hiddens = res_hiddens.reshape(input_shape[0], input_shape[1], -1).contiguous()
        pho_res_embeddings = pho_hiddens + res_hiddens
        pho_res_outputs = self.pho_res_model(inputs_embeds=pho_res_embeddings, attention_mask=attention_mask)[0]

        concated_outputs = torch.cat((bert_outputs, pho_res_outputs), dim=-1)
        concated_outputs = self.integrate(concated_outputs)

        outputs = self.output_block(inputs_embeds=concated_outputs,
                    position_ids=torch.zeros(input_ids.size(), dtype=torch.long, device=input_ids.device),
                    attention_mask=attention_mask)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if label_ids is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            active_loss = loss_mask.view(-1) == 1
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = label_ids.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            outputs = (loss,) + outputs
        return outputs 

class SpellBertPho2ResArch2(BertPreTrainedModel):
    def __init__(self, config):
        super(SpellBertPho2ResArch2, self).__init__(config)

        self.vocab_size = config.vocab_size
        self.bert = BertModel(config)

        self.pho_embeddings = nn.Embedding(pho2_convertor.get_pho_size(), config.hidden_size, padding_idx=0)
        self.pho_gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )
        pho_config = deepcopy(config)
        pho_config.num_hidden_layers = 4
        self.pho_model = BertModel(pho_config)

        self.char_images = nn.Embedding(config.vocab_size, 1024)
        self.char_images.weight.requires_grad = False

        if config.image_model_type == 0:
            self.resnet = CharResNet()
        elif config.image_model_type == 1:
            self.resnet = CharResNet1()
        else:
            raise NotImplementedError('invalid image_model_type %d'%config.image_model_type)
        self.resnet_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.integrate = nn.Linear(3*config.hidden_size, config.hidden_size)
        out_config = deepcopy(config)
        out_config.num_hidden_layers = 2
        self.output_block = BertModel(out_config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)

        self.init_weights()

    def tie_cls_weight(self):
        self.classifier.weight = self.bert.embeddings.word_embeddings.weight

    def build_glyce_embed(self, vocab_dir, font_path, font_size=32):
        vocab_path = os.path.join(vocab_dir, 'vocab.txt')
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = [s.strip() for s in f]

        font = ImageFont.truetype(font_path, size=font_size)

        char_images = []
        for char in vocab:
            if len(char) != 1 or (not _is_chinese_char(ord(char))):
                char_images.append(np.zeros((font_size, font_size)).astype(np.float32))
                continue
            image = font.getmask(char)
            image = np.asarray(image).astype(np.float32).reshape(image.size[::-1])  # Must be [::-1]

            # Crop
            image = image[:font_size, :font_size]

            # Pad
            if image.size != (font_size, font_size):
                back_image = np.zeros((font_size, font_size)).astype(np.float32)
                offset0 = (font_size - image.shape[0]) // 2
                offset1 = (font_size - image.shape[1]) // 2
                back_image[offset0:offset0 + image.shape[0], offset1:offset1 + image.shape[1]] = image
                image = back_image

            char_images.append(image)
        char_images = np.array(char_images)
        char_images = (char_images - np.mean(char_images)) / np.std(char_images)
        char_images = torch.from_numpy(char_images).reshape(char_images.shape[0], -1)
        assert char_images.shape == (21128, 1024)
        self.char_images.weight.data.copy_(char_images)

    @staticmethod
    def build_batch(batch, tokenizer):
        src_idx = batch['src_idx'].flatten().tolist()
        chars = tokenizer.convert_ids_to_tokens(src_idx)
        pho_idx, pho_lens = pho2_convertor.convert(chars)
        batch['pho_idx'] = pho_idx
        batch['pho_lens'] = pho_lens
        return batch

    def forward(self, batch):
        input_ids = batch['src_idx']
        attention_mask = batch['masks']
        loss_mask = batch['loss_masks']
        pho_idx = batch['pho_idx']
        pho_lens = batch['pho_lens']
        label_ids = batch['tgt_idx'] if 'tgt_idx' in batch else None

        input_shape = input_ids.size()

        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)[0]
        
        pho_embeddings = self.pho_embeddings(pho_idx)
        pho_embeddings = torch.nn.utils.rnn.pack_padded_sequence(
            input=pho_embeddings,
            lengths=pho_lens,
            batch_first=True,
            enforce_sorted=False,
        )
        _, pho_hiddens = self.pho_gru(pho_embeddings)
        pho_hiddens = pho_hiddens.squeeze(0).reshape(input_shape[0], input_shape[1], -1).contiguous()
        pho_hiddens = self.pho_model(inputs_embeds=pho_hiddens, attention_mask=attention_mask)[0]

        src_idxs = input_ids.view(-1)
        images = self.char_images(src_idxs).reshape(src_idxs.shape[0], 1, 32, 32).contiguous()
        res_hiddens = self.resnet(images)
        res_hiddens = res_hiddens.reshape(input_shape[0], input_shape[1], -1).contiguous()
        res_hiddens = self.resnet_layernorm(res_hiddens)

        concated_outputs = torch.cat((bert_outputs, pho_hiddens, res_hiddens), dim=-1)
        concated_outputs = self.integrate(concated_outputs)

        outputs = self.output_block(inputs_embeds=concated_outputs,
                    position_ids=torch.zeros(input_ids.size(), dtype=torch.long, device=input_ids.device),
                    attention_mask=attention_mask)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if label_ids is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            active_loss = loss_mask.view(-1) == 1
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = label_ids.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            outputs = (loss,) + outputs
        return outputs 


class SpellBertPho2ResArch3(BertPreTrainedModel):

    def __init__(self, config):
        super(SpellBertPho2ResArch3, self).__init__(config)
        self.config = config

        self.vocab_size = config.vocab_size
        self.bert = BertModel(config)

        self.pho_embeddings = nn.Embedding(pho2_convertor.get_pho_size(), config.hidden_size, padding_idx=0)
        self.pho_gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )
        pho_config = deepcopy(config)
        pho_config.num_hidden_layers = 4
        self.pho_model = BertModel(pho_config)

        if self.config.num_fonts == 1:
            self.char_images = nn.Embedding(config.vocab_size, 1024)
            self.char_images.weight.requires_grad = False
        else:
            self.char_images_multifonts = torch.nn.Parameter(torch.rand(21128, self.config.num_fonts, 32, 32))
            self.char_images_multifonts.requires_grad = False

        if config.image_model_type == 0:
            self.resnet = CharResNet(in_channels=self.config.num_fonts)
        elif config.image_model_type == 1:
            self.resnet = CharResNet1()
        else:
            raise NotImplementedError('invalid image_model_type %d'%config.image_model_type)
        self.resnet_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)# 层norm处理。

        self.gate_net = nn.Linear(4*config.hidden_size, 3)

        out_config = deepcopy(config)
        out_config.num_hidden_layers = 3
        self.output_block = BertModel(out_config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)

        self.init_weights()

    def tie_cls_weight(self):
        self.classifier.weight = self.bert.embeddings.word_embeddings.weight

    def build_glyce_embed(self, vocab_dir, font_path, font_size=32):
        vocab_path = os.path.join(vocab_dir, 'vocab.txt')
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = [s.strip() for s in f]

        font = ImageFont.truetype(font_path, size=font_size)

        char_images = []
        for char in vocab:
            if len(char) != 1 or (not _is_chinese_char(ord(char))):
                char_images.append(np.zeros((font_size, font_size)).astype(np.float32))
                continue
            image = font.getmask(char)
            image = np.asarray(image).astype(np.float32).reshape(image.size[::-1])  # Must be [::-1]

            # Crop
            image = image[:font_size, :font_size]

            # Pad
            if image.size != (font_size, font_size):
                back_image = np.zeros((font_size, font_size)).astype(np.float32)
                offset0 = (font_size - image.shape[0]) // 2
                offset1 = (font_size - image.shape[1]) // 2
                back_image[offset0:offset0 + image.shape[0], offset1:offset1 + image.shape[1]] = image
                image = back_image

            char_images.append(image)
        char_images = np.array(char_images)
        char_images = (char_images - np.mean(char_images)) / np.std(char_images)
        char_images = torch.from_numpy(char_images).reshape(char_images.shape[0], -1)
        assert char_images.shape == (21128, 1024)
        self.char_images.weight.data.copy_(char_images)

    # Add by hengdaxu
    def build_glyce_embed_multifonts(self, vocab_dir, num_fonts, use_traditional_font, font_size=32):
        font_paths = [
            ('simhei.ttf', False),
            ('xiaozhuan.ttf', False),
            ('simhei.ttf', True),
        ]
        font_paths = font_paths[:num_fonts]
        if use_traditional_font:
            font_paths = font_paths[:-1]
            font_paths.append(('simhei.ttf', True))
            self.converter = opencc.OpenCC('s2t.json')

        images_list = []
        for font_path, use_traditional in font_paths:
            images = self.build_glyce_embed_onefont(
                vocab_dir=vocab_dir,
                font_path=font_path,
                font_size=font_size,
                use_traditional=use_traditional,
            )
            images_list.append(images)
        # images_list  num_font x vocable_size 
        char_images = torch.stack(images_list, dim=1).contiguous()# 把一个汉字对应的所有图片都放到一起了。
        # images_list   vocable_size x num_font
        self.char_images_multifonts.data.copy_(char_images)

    # Add by hengdaxu
    def build_glyce_embed_onefont(self, vocab_dir, font_path, font_size, use_traditional):
        vocab_path = os.path.join(vocab_dir, 'vocab.txt')
        with open(vocab_path) as f:
            vocab = [s.strip() for s in f.readlines()]
        if use_traditional:
            vocab = [self.converter.convert(c) if len(c) == 1 else c for c in vocab]

        font = ImageFont.truetype(font_path, size=font_size)

        char_images = []
        for char in vocab:
            if len(char) > 1:
                char_images.append(np.zeros((font_size, font_size)).astype(np.float32))
                continue
            image = font.getmask(char)
            image = np.asarray(image).astype(np.float32).reshape(image.size[::-1])  # Must be [::-1]

            # Crop
            image = image[:font_size, :font_size]

            # Pad
            if image.size != (font_size, font_size):
                back_image = np.zeros((font_size, font_size)).astype(np.float32)
                offset0 = (font_size - image.shape[0]) // 2
                offset1 = (font_size - image.shape[1]) // 2
                back_image[offset0:offset0 + image.shape[0], offset1:offset1 + image.shape[1]] = image
                image = back_image

            char_images.append(image)
        char_images = np.array(char_images)
        char_images = (char_images - np.mean(char_images)) / np.std(char_images)
        char_images = torch.from_numpy(char_images).contiguous()
        return char_images

    @staticmethod
    def build_batch(batch, tokenizer):
        src_idx = batch['src_idx'].flatten().tolist()
        chars = tokenizer.convert_ids_to_tokens(src_idx)
        pho_idx, pho_lens = pho2_convertor.convert(chars)
        batch['pho_idx'] = pho_idx
        batch['pho_lens'] = pho_lens
        return batch

    def forward(self, batch):
        input_ids = batch['src_idx']
        attention_mask = batch['masks']
        loss_mask = batch['loss_masks']
        pho_idx = batch['pho_idx']
        pho_lens = batch['pho_lens']
        label_ids = batch['tgt_idx'] if 'tgt_idx' in batch else None

        input_shape = input_ids.size()

        bert_hiddens = self.bert(input_ids, attention_mask=attention_mask)[0]
        
        pho_embeddings = self.pho_embeddings(pho_idx)
        pho_embeddings = torch.nn.utils.rnn.pack_padded_sequence(
            input=pho_embeddings,
            lengths=pho_lens,
            batch_first=True,
            enforce_sorted=False,
        )
        _, pho_hiddens = self.pho_gru(pho_embeddings)
        pho_hiddens = pho_hiddens.squeeze(0).reshape(input_shape[0], input_shape[1], -1).contiguous()
        pho_hiddens = self.pho_model(inputs_embeds=pho_hiddens, attention_mask=attention_mask)[0]

        src_idxs = input_ids.view(-1)

        if self.config.num_fonts == 1:
            images = self.char_images(src_idxs).reshape(src_idxs.shape[0], 1, 32, 32).contiguous()
        else:
            images = self.char_images_multifonts.index_select(dim=0, index=src_idxs)

        res_hiddens = self.resnet(images)
        res_hiddens = res_hiddens.reshape(input_shape[0], input_shape[1], -1).contiguous()# 批量大小，句子长度
        res_hiddens = self.resnet_layernorm(res_hiddens)

        bert_hiddens_mean = (bert_hiddens * attention_mask.to(torch.float).unsqueeze(2)).sum(dim=1) / attention_mask.to(torch.float).sum(dim=1, keepdim=True)
        bert_hiddens_mean = bert_hiddens_mean.unsqueeze(1).expand(-1, bert_hiddens.size(1), -1)

        concated_outputs = torch.cat((bert_hiddens, pho_hiddens, res_hiddens, bert_hiddens_mean), dim=-1)
        gated_values = self.gate_net(concated_outputs)
        # B * S * 3
        g0 = torch.sigmoid(gated_values[:,:,0].unsqueeze(-1))
        g1 = torch.sigmoid(gated_values[:,:,1].unsqueeze(-1))
        g2 = torch.sigmoid(gated_values[:,:,2].unsqueeze(-1))
        
        hiddens = g0* bert_hiddens + g1* pho_hiddens + g2* res_hiddens

        outputs = self.output_block(inputs_embeds=hiddens,
                    position_ids=torch.zeros(input_ids.size(), dtype=torch.long, device=input_ids.device),
                    attention_mask=attention_mask)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if label_ids is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            active_loss = loss_mask.view(-1) == 1
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = label_ids.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            outputs = (loss,) + outputs
        return outputs 



class SpellBertPho2ResArch3MLM(BertPreTrainedModel):
    def __init__(self, config):
        super(SpellBertPho2ResArch3MLM, self).__init__(config)

        self.vocab_size = config.vocab_size
        self.bert = BertModel(config)

        self.pho_embeddings = nn.Embedding(pho2_convertor.get_pho_size(), config.hidden_size, padding_idx=0)
        self.pho_gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )
        pho_config = deepcopy(config)
        pho_config.num_hidden_layers = 4
        self.pho_model = BertModel(pho_config)

        self.char_images = nn.Embedding(config.vocab_size, 1024)
        self.char_images.weight.requires_grad = False

        if config.image_model_type == 0:
            self.resnet = CharResNet()
        elif config.image_model_type == 1:
            self.resnet = CharResNet1()
        else:
            raise NotImplementedError('invalid image_model_type %d'%config.image_model_type)
        self.resnet_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.gate_net = nn.Linear(4*config.hidden_size, 3)

        out_config = deepcopy(config)
        out_config.num_hidden_layers = 3
        self.output_block = BertModel(out_config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

    def tie_cls_weight(self):
        #self.classifier.weight = self.bert.embeddings.word_embeddings.weight
        pass
        
    def build_glyce_embed(self, vocab_dir, font_path, font_size=32):
        vocab_path = os.path.join(vocab_dir, 'vocab.txt')
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = [s.strip() for s in f]

        font = ImageFont.truetype(font_path, size=font_size)

        char_images = []
        for char in vocab:
            if len(char) != 1 or (not _is_chinese_char(ord(char))):
                char_images.append(np.zeros((font_size, font_size)).astype(np.float32))
                continue
            image = font.getmask(char)
            image = np.asarray(image).astype(np.float32).reshape(image.size[::-1])  # Must be [::-1]

            # Crop
            image = image[:font_size, :font_size]

            # Pad
            if image.size != (font_size, font_size):
                back_image = np.zeros((font_size, font_size)).astype(np.float32)
                offset0 = (font_size - image.shape[0]) // 2
                offset1 = (font_size - image.shape[1]) // 2
                back_image[offset0:offset0 + image.shape[0], offset1:offset1 + image.shape[1]] = image
                image = back_image

            char_images.append(image)
        char_images = np.array(char_images)
        char_images = (char_images - np.mean(char_images)) / np.std(char_images)
        char_images = torch.from_numpy(char_images).reshape(char_images.shape[0], -1)
        assert char_images.shape == (21128, 1024)
        self.char_images.weight.data.copy_(char_images)

    @staticmethod
    def build_batch(batch, tokenizer):
        src_idx = batch['src_idx'].flatten().tolist()
        chars = tokenizer.convert_ids_to_tokens(src_idx)
        pho_idx, pho_lens = pho2_convertor.convert(chars)
        batch['pho_idx'] = pho_idx
        batch['pho_lens'] = pho_lens
        return batch

    def forward(self, batch):
        input_ids = batch['src_idx']
        attention_mask = batch['masks']
        loss_mask = batch['loss_masks']
        pho_idx = batch['pho_idx']
        pho_lens = batch['pho_lens']
        label_ids = batch['tgt_idx'] if 'tgt_idx' in batch else None

        input_shape = input_ids.size()

        bert_hiddens = self.bert(input_ids, attention_mask=attention_mask)[0]
        
        pho_embeddings = self.pho_embeddings(pho_idx)
        pho_embeddings = torch.nn.utils.rnn.pack_padded_sequence(
            input=pho_embeddings,
            lengths=pho_lens,
            batch_first=True,
            enforce_sorted=False,
        )
        _, pho_hiddens = self.pho_gru(pho_embeddings)
        pho_hiddens = pho_hiddens.squeeze(0).reshape(input_shape[0], input_shape[1], -1).contiguous()
        pho_hiddens = self.pho_model(inputs_embeds=pho_hiddens, attention_mask=attention_mask)[0]

        src_idxs = input_ids.view(-1)
        images = self.char_images(src_idxs).reshape(src_idxs.shape[0], 1, 32, 32).contiguous()
        res_hiddens = self.resnet(images)
        res_hiddens = res_hiddens.reshape(input_shape[0], input_shape[1], -1).contiguous()
        res_hiddens = self.resnet_layernorm(res_hiddens)

        bert_hiddens_mean = (bert_hiddens * attention_mask.to(torch.float).unsqueeze(2)).sum(dim=1) / attention_mask.to(torch.float).sum(dim=1, keepdim=True)
        bert_hiddens_mean = bert_hiddens_mean.unsqueeze(1).expand(-1, bert_hiddens.size(1), -1)

        concated_outputs = torch.cat((bert_hiddens, pho_hiddens, res_hiddens, bert_hiddens_mean), dim=-1)
        gated_values = self.gate_net(concated_outputs)
        # B * S * 3
        g0 = torch.sigmoid(gated_values[:,:,0].unsqueeze(-1))
        g1 = torch.sigmoid(gated_values[:,:,1].unsqueeze(-1))
        g2 = torch.sigmoid(gated_values[:,:,2].unsqueeze(-1))
        
        hiddens = g0* bert_hiddens + g1* pho_hiddens + g2* res_hiddens

        outputs = self.output_block(inputs_embeds=hiddens,
                    position_ids=torch.zeros(input_ids.size(), dtype=torch.long, device=input_ids.device),
                    attention_mask=attention_mask)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.cls(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if label_ids is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            active_loss = loss_mask.view(-1) == 1
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = label_ids.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            outputs = (loss,) + outputs
        return outputs 


class SpellBertPho2ResArch4(BertPreTrainedModel):
    def __init__(self, config):
        super(SpellBertPho2ResArch4, self).__init__(config)

        self.vocab_size = config.vocab_size
        self.bert = BertModel(config)

        self.pho_embeddings = nn.Embedding(pho2_convertor.get_pho_size(), config.hidden_size, padding_idx=0)
        self.pho_gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )
        pho_config = deepcopy(config)
        pho_config.num_hidden_layers = 4
        self.pho_model = BertModel(pho_config)

        self.char_images = nn.Embedding(config.vocab_size, 1024)
        self.char_images.weight.requires_grad = False

        if config.image_model_type == 0:
            self.resnet = CharResNet()
        elif config.image_model_type == 1:
            self.resnet = CharResNet1()
        else:
            raise NotImplementedError('invalid image_model_type %d'%config.image_model_type)
        self.resnet_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.gate_net = nn.Linear(4*config.hidden_size, 3)

        out_config = deepcopy(config)
        out_config.num_hidden_layers = 3
        self.output_block = BertModel(out_config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)

        self.init_weights()

    def tie_cls_weight(self):
        self.classifier.weight = self.bert.embeddings.word_embeddings.weight

    def build_glyce_embed(self, vocab_dir, font_path, font_size=32):
        vocab_path = os.path.join(vocab_dir, 'vocab.txt')
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = [s.strip() for s in f]

        font = ImageFont.truetype(font_path, size=font_size)

        char_images = []
        for char in vocab:
            if len(char) != 1 or (not _is_chinese_char(ord(char))):
                char_images.append(np.zeros((font_size, font_size)).astype(np.float32))
                continue
            image = font.getmask(char)
            image = np.asarray(image).astype(np.float32).reshape(image.size[::-1])  # Must be [::-1]

            # Crop
            image = image[:font_size, :font_size]

            # Pad
            if image.size != (font_size, font_size):
                back_image = np.zeros((font_size, font_size)).astype(np.float32)
                offset0 = (font_size - image.shape[0]) // 2
                offset1 = (font_size - image.shape[1]) // 2
                back_image[offset0:offset0 + image.shape[0], offset1:offset1 + image.shape[1]] = image
                image = back_image

            char_images.append(image)
        char_images = np.array(char_images)
        char_images = (char_images - np.mean(char_images)) / np.std(char_images)
        char_images = torch.from_numpy(char_images).reshape(char_images.shape[0], -1)
        assert char_images.shape == (21128, 1024)
        self.char_images.weight.data.copy_(char_images)

    @staticmethod
    def build_batch(batch, tokenizer):
        src_idx = batch['src_idx'].flatten().tolist()
        chars = tokenizer.convert_ids_to_tokens(src_idx)
        pho_idx, pho_lens = pho2_convertor.convert(chars)
        batch['pho_idx'] = pho_idx
        batch['pho_lens'] = pho_lens
        return batch

    def forward(self, batch):
        input_ids = batch['src_idx']
        attention_mask = batch['masks']
        loss_mask = batch['loss_masks']
        pho_idx = batch['pho_idx']
        pho_lens = batch['pho_lens']
        label_ids = batch['tgt_idx'] if 'tgt_idx' in batch else None

        input_shape = input_ids.size()

        bert_hiddens = self.bert(input_ids, attention_mask=attention_mask)[0]
        
        pho_embeddings = self.pho_embeddings(pho_idx)
        pho_embeddings = torch.nn.utils.rnn.pack_padded_sequence(
            input=pho_embeddings,
            lengths=pho_lens,
            batch_first=True,
            enforce_sorted=False,
        )
        _, pho_hiddens = self.pho_gru(pho_embeddings)
        pho_hiddens = pho_hiddens.squeeze(0).reshape(input_shape[0], input_shape[1], -1).contiguous()
        pho_hiddens = self.pho_model(inputs_embeds=pho_hiddens, attention_mask=attention_mask)[0]

        src_idxs = input_ids.view(-1)
        images = self.char_images(src_idxs).reshape(src_idxs.shape[0], 1, 32, 32).contiguous()
        res_hiddens = self.resnet(images)
        res_hiddens = res_hiddens.reshape(input_shape[0], input_shape[1], -1).contiguous()
        res_hiddens = self.resnet_layernorm(res_hiddens)

        bert_hiddens_mean = (bert_hiddens * attention_mask.to(torch.float).unsqueeze(2)).sum(dim=1) / attention_mask.to(torch.float).sum(dim=1, keepdim=True)
        bert_hiddens_mean = bert_hiddens_mean.unsqueeze(1).expand(-1, bert_hiddens.size(1), -1)

        concated_outputs = torch.cat((bert_hiddens, pho_hiddens, res_hiddens, bert_hiddens_mean), dim=-1)
        gated_values = self.gate_net(concated_outputs)
        gated_values = nn.functional.softmax(gated_values, dim=-1)
        # B * S * 3
        g0 = gated_values[:,:,0].unsqueeze(-1)
        g1 = gated_values[:,:,1].unsqueeze(-1)
        g2 = gated_values[:,:,2].unsqueeze(-1)
        
        hiddens = g0* bert_hiddens + g1* pho_hiddens + g2* res_hiddens

        outputs = self.output_block(inputs_embeds=hiddens,
                    position_ids=torch.zeros(input_ids.size(), dtype=torch.long, device=input_ids.device),
                    attention_mask=attention_mask)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if label_ids is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            active_loss = loss_mask.view(-1) == 1
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = label_ids.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            outputs = (loss,) + outputs
        return outputs 



class Pho2ResPretrain(BertPreTrainedModel):
    def __init__(self, config):
        super(Pho2ResPretrain, self).__init__(config)
        self.config = config

        self.vocab_size = config.vocab_size

        self.char_images = nn.Embedding(config.vocab_size, 1024)#这里的意思是，每一个汉字的图片都转换成1024维度的图片。
        self.char_images.weight.requires_grad = False# 每个字符对应一个向量，但是不要求梯度，也就是不再变化的意思？

        self.pho_embeddings = nn.Embedding(pho2_convertor.get_pho_size(), config.hidden_size, padding_idx=0)#这个是声音的嵌入层，5个读音加上拼音字母
        self.pho_gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )
        self.resnet = CharResNet()# 这个是给图片识别使用的网络。
        pho_res_config = deepcopy(config)
        pho_res_config.num_hidden_layers = 4
        self.pho_res_model = BertModel(pho_res_config)

        self.cls2 = BertOnlyMLMHead(config)

        self.init_weights()

    def build_glyce_embed(self, vocab_dir, font_path, font_size=32):
        vocab_path = os.path.join(vocab_dir, 'vocab.txt')
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = [s.strip() for s in f]

        font = ImageFont.truetype(font_path, size=font_size)

        char_images = []
        for char in vocab:
            if len(char) != 1 or (not _is_chinese_char(ord(char))):
                char_images.append(np.zeros((font_size, font_size)).astype(np.float32))
                continue
            image = font.getmask(char)
            image = np.asarray(image).astype(np.float32).reshape(image.size[::-1])  # Must be [::-1]

            # Crop
            image = image[:font_size, :font_size]

            # Pad
            if image.size != (font_size, font_size):
                back_image = np.zeros((font_size, font_size)).astype(np.float32)
                offset0 = (font_size - image.shape[0]) // 2
                offset1 = (font_size - image.shape[1]) // 2
                back_image[offset0:offset0 + image.shape[0], offset1:offset1 + image.shape[1]] = image
                image = back_image

            char_images.append(image)# 给每个汉字都配上了图片
        char_images = np.array(char_images)
        char_images = (char_images - np.mean(char_images)) / np.std(char_images)# 归一化图片
        char_images = torch.from_numpy(char_images).reshape(char_images.shape[0], -1)
        assert char_images.shape == (21128, 1024)
        self.char_images.weight.data.copy_(char_images)

    @staticmethod
    def build_batch(batch, tokenizer):
        src_idx = batch['tgt_idx'].flatten().tolist()# 变成了batch_size*max_seq_lenght 因为里面本身，此时的数据已经填充了000
        chars = tokenizer.convert_ids_to_tokens(src_idx)# 重新变成了token的数据。
        pho_idx, pho_lens = pho2_convertor.convert(chars)
        batch['pho_idx'] = pho_idx
        batch['pho_lens'] = pho_lens
        return batch

    def forward(self, batch):
        input_ids = batch['tgt_idx']
        attention_mask = batch['masks']
        loss_mask = batch['loss_masks']
        pho_idx = batch['pho_idx']
        pho_lens = batch['pho_lens']

        input_shape = input_ids.size()# batch_size*max_seq_length
        
        pho_embeddings = self.pho_embeddings(pho_idx)# 这个batch_size是1024的字符，然后每个字符都有自己的汉语拼音。
        pho_embeddings = torch.nn.utils.rnn.pack_padded_sequence(
            input=pho_embeddings,
            lengths=pho_lens,
            batch_first=True,
            enforce_sorted=False,
        )# [1024,6,768]->[1340,768],为什么要执行这一步呢？这一步的意义是什么呢？1340=sum(pho_lens)是什么意思呢？是把所有的拼音都处理一下了吗？
        # pack是压缩的意思，这一步的意思是压缩数据。
        _, pho_hiddens = self.pho_gru(pho_embeddings)# 进行操作的时候，应该会自动恢复到1024的字符长度。应该是自动执行了pad_pack_sequence
        pho_hiddens = pho_hiddens.squeeze(0).reshape(input_shape[0], input_shape[1], -1).contiguous()# 对显存进行的操作，把这部分数据变化成连续的数据。
        
        src_idxs = input_ids.view(-1)# 把一个batch中所有的额汉字 都 flatten
        # 等等以后这个地方注意一下，他的输入是

        if self.config.num_fonts == 1:
            # 这块儿是什么意思？怎么还加入图片的模态。
            images = self.char_images(src_idxs).reshape(src_idxs.shape[0], 1, 32, 32).contiguous()
        else:
            images = self.char_images.index_select(dim=0, index=src_idxs)

        res_hiddens = self.resnet(images)
        res_hiddens = res_hiddens.reshape(input_shape[0], input_shape[1], -1).contiguous()
        pho_res_embeddings = pho_hiddens + res_hiddens
        sequence_output = self.pho_res_model(inputs_embeds=pho_res_embeddings, attention_mask=attention_mask)[0]

        prediction_scores = self.cls2(sequence_output)

        loss_fct = CrossEntropyLoss()
        # Only keep active parts of the loss
        active_loss = loss_mask.view(-1) == 1
        active_logits = prediction_scores.view(-1, self.vocab_size)[active_loss]
        active_labels = src_idxs[active_loss]
        loss = loss_fct(active_logits, active_labels)
        outputs = (loss, active_logits.argmax(dim=-1), active_labels, )
        return outputs 

class Pho2Pretrain(BertPreTrainedModel):
    def __init__(self, config):
        super(Pho2Pretrain, self).__init__(config)

        self.vocab_size = config.vocab_size

        self.pho_embeddings = nn.Embedding(pho2_convertor.get_pho_size(), config.hidden_size, padding_idx=0)
        self.pho_gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )
        pho_config = deepcopy(config)
        pho_config.num_hidden_layers = 4
        self.pho_model = BertModel(pho_config)

        self.cls2 = BertOnlyMLMHead(config)

        self.init_weights()

    @staticmethod
    def build_batch(batch, tokenizer):
        src_idx = batch['tgt_idx'].flatten().tolist()
        chars = tokenizer.convert_ids_to_tokens(src_idx)
        pho_idx, pho_lens = pho2_convertor.convert(chars)
        batch['pho_idx'] = pho_idx
        batch['pho_lens'] = pho_lens
        return batch

    def forward(self, batch):
        input_ids = batch['tgt_idx']
        attention_mask = batch['masks']
        loss_mask = batch['loss_masks']
        pho_idx = batch['pho_idx']
        pho_lens = batch['pho_lens']

        input_shape = input_ids.size()
        
        pho_embeddings = self.pho_embeddings(pho_idx)
        pho_embeddings = torch.nn.utils.rnn.pack_padded_sequence(
            input=pho_embeddings,
            lengths=pho_lens,
            batch_first=True,
            enforce_sorted=False,
        )
        _, pho_hiddens = self.pho_gru(pho_embeddings)
        pho_hiddens = pho_hiddens.squeeze(0).reshape(input_shape[0], input_shape[1], -1).contiguous()
        # 这个时候把
        sequence_output = self.pho_model(inputs_embeds=pho_hiddens, attention_mask=attention_mask)[0]
        # 这个应该是bert的头的输出了，输出的结构是，2，512，768，没有问题。

        prediction_scores = self.cls2(sequence_output)
        # 现在的做法应该是，先输入是正确的，然后让他预测也是正确的，之后在微调吧？
        loss_fct = CrossEntropyLoss()
        # Only keep active parts of the loss
        active_loss = loss_mask.view(-1) == 1
        active_logits = prediction_scores.view(-1, self.vocab_size)[active_loss]# 这块应该就是不计算那些标点符号，填充符号的损失了。
        active_labels = input_ids.view(-1)[active_loss]
        loss = loss_fct(active_logits, active_labels)
        outputs = (loss, active_logits.argmax(dim=-1), active_labels, )
        return outputs 

class ResPretrain(BertPreTrainedModel):
    def __init__(self, config):
        super(ResPretrain, self).__init__(config)
        self.config = config

        self.vocab_size = config.vocab_size

        if self.config.num_fonts == 1:
            self.char_images = nn.Embedding(config.vocab_size, 1024)
            self.char_images.weight.requires_grad = False
        else:
            self.char_images_multifonts = torch.nn.Parameter(torch.rand(21128, self.config.num_fonts, 32, 32))
            self.char_images_multifonts.requires_grad = False

        if config.image_model_type == 0:
            self.resnet = CharResNet(in_channels=self.config.num_fonts)# 懂了，通过chanel的数量做输入。多个图片，就是多个channel的。
        elif config.image_model_type == 1:
            self.resnet = CharResNet1()
        else:
            raise NotImplementedError('invalid image_model_type %d'%config.image_model_type)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls3 = nn.Linear(config.hidden_size, config.vocab_size)# 线性层的预测分数。

        self.init_weights()

    def build_glyce_embed(self, vocab_dir, font_path, font_size=32):
        vocab_path = os.path.join(vocab_dir, 'vocab.txt')
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = [s.strip() for s in f]

        font = ImageFont.truetype(font_path, size=font_size)

        char_images = []
        for char in vocab:
            if len(char) != 1 or (not _is_chinese_char(ord(char))):
                char_images.append(np.zeros((font_size, font_size)).astype(np.float32))
                continue
            image = font.getmask(char)
            image = np.asarray(image).astype(np.float32).reshape(image.size[::-1])  # Must be [::-1]

            # Crop
            image = image[:font_size, :font_size]

            # Pad
            if image.size != (font_size, font_size):
                back_image = np.zeros((font_size, font_size)).astype(np.float32)
                offset0 = (font_size - image.shape[0]) // 2
                offset1 = (font_size - image.shape[1]) // 2
                back_image[offset0:offset0 + image.shape[0], offset1:offset1 + image.shape[1]] = image
                image = back_image

            char_images.append(image)
        char_images = np.array(char_images)
        char_images = (char_images - np.mean(char_images)) / np.std(char_images)
        char_images = torch.from_numpy(char_images).reshape(char_images.shape[0], -1)
        assert char_images.shape == (21128, 1024)
        self.char_images.weight.data.copy_(char_images)

    # Add by hengdaxu
    def build_glyce_embed_multifonts(self, vocab_dir, num_fonts, use_traditional_font, font_size=32):
        # 多个字体的情况
        font_paths = [
            ('simhei.ttf', False),
            ('xiaozhuan.ttf', False),
            ('simhei.ttf', True),
        ]
        font_paths = font_paths[:num_fonts]
        if use_traditional_font:
            font_paths = font_paths[:-1]
            font_paths.append(('simhei.ttf', True))
            self.converter = opencc.OpenCC('s2t.json')

        images_list = []
        for font_path, use_traditional in font_paths:
            images = self.build_glyce_embed_onefont(
                vocab_dir=vocab_dir,
                font_path=font_path,
                font_size=font_size,
                use_traditional=use_traditional,
            )
            images_list.append(images)

        char_images = torch.stack(images_list, dim=1).contiguous()
        self.char_images_multifonts.data.copy_(char_images)

    # Add by hengdaxu
    def build_glyce_embed_onefont(self, vocab_dir, font_path, font_size, use_traditional):
        vocab_path = os.path.join(vocab_dir, 'vocab.txt')
        with open(vocab_path) as f:
            vocab = [s.strip() for s in f.readlines()]
        if use_traditional:
            vocab = [self.converter.convert(c) if len(c) == 1 else c for c in vocab]

        font = ImageFont.truetype(font_path, size=font_size)

        char_images = []
        for char in vocab:
            if len(char) > 1:
                char_images.append(np.zeros((font_size, font_size)).astype(np.float32))
                continue
            image = font.getmask(char)
            image = np.asarray(image).astype(np.float32).reshape(image.size[::-1])  # Must be [::-1]

            # Crop
            image = image[:font_size, :font_size]

            # Pad
            if image.size != (font_size, font_size):
                back_image = np.zeros((font_size, font_size)).astype(np.float32)
                offset0 = (font_size - image.shape[0]) // 2
                offset1 = (font_size - image.shape[1]) // 2
                back_image[offset0:offset0 + image.shape[0], offset1:offset1 + image.shape[1]] = image
                image = back_image

            char_images.append(image)
        char_images = np.array(char_images)
        char_images = (char_images - np.mean(char_images)) / np.std(char_images)
        char_images = torch.from_numpy(char_images).contiguous()
        return char_images

    @staticmethod
    def build_batch(batch, tokenizer):
        return batch

    def forward(self, batch):
        input_ids = batch['input_ids'] # (N, )        

        if self.config.num_fonts == 1:
            images = self.char_images(input_ids).reshape(input_ids.shape[0], 1, 32, 32).contiguous()
        else:
            images = self.char_images_multifonts.index_select(dim=0, index=input_ids)

        res_hiddens = self.resnet(images)
        res_hiddens = self.dropout(res_hiddens)        
        prediction_scores = self.cls3(res_hiddens)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(prediction_scores, input_ids)
        outputs = (loss, prediction_scores.argmax(dim=-1), input_ids, )
        return outputs 

