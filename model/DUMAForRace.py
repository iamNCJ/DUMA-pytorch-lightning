from typing import Any, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MultiheadAttention
from transformers import BertConfig, BertModel, AdamW, get_linear_schedule_with_warmup
from transformers.modeling_outputs import MultipleChoiceModelOutput


def separate_seq2(sequence_output, flat_input_ids):
    qa_seq_output = sequence_output.new(sequence_output.size()).zero_()
    qa_mask = torch.ones((sequence_output.shape[0], sequence_output.shape[1]),
                         device=sequence_output.device,
                         dtype=torch.bool)
    p_seq_output = sequence_output.new(sequence_output.size()).zero_()
    p_mask = torch.ones((sequence_output.shape[0], sequence_output.shape[1]),
                        device=sequence_output.device,
                        dtype=torch.bool)
    for i in range(flat_input_ids.size(0)):
        sep_lst = []
        for idx, e in enumerate(flat_input_ids[i]):
            if e == 102:
                sep_lst.append(idx)
        assert len(sep_lst) == 2
        qa_seq_output[i, :sep_lst[0] - 1] = sequence_output[i, 1:sep_lst[0]]
        qa_mask[i, :sep_lst[0] - 1] = 0
        p_seq_output[i, :sep_lst[1] - sep_lst[0] - 1] = sequence_output[i, sep_lst[0] + 1: sep_lst[1]]
        p_mask[i, :sep_lst[1] - sep_lst[0] - 1] = 0
    return qa_seq_output, p_seq_output, qa_mask, p_mask


class DUMALayer(nn.Module):
    def __init__(self, d_model_size, num_heads):
        super(DUMALayer, self).__init__()
        self.attn_qa = MultiheadAttention(d_model_size, num_heads)
        self.attn_p = MultiheadAttention(d_model_size, num_heads)

    def forward(self, qa_seq_representation, p_seq_representation, qa_mask=None, p_mask=None):
        qa_seq_representation = qa_seq_representation.permute([1, 0, 2])
        p_seq_representation = p_seq_representation.permute([1, 0, 2])
        enc_output_qa, _ = self.attn_qa(
            value=qa_seq_representation, key=qa_seq_representation, query=p_seq_representation, key_padding_mask=qa_mask
        )
        enc_output_p, _ = self.attn_p(
            value=p_seq_representation, key=p_seq_representation, query=qa_seq_representation, key_padding_mask=p_mask
        )
        return enc_output_qa.permute([1, 0, 2]), enc_output_p.permute([1, 0, 2])


class DUMAForRace(pl.LightningModule):
    def __init__(
            self,
            pretrained_model: str = 'bert-large-uncased',
            learning_rate: float = 2e-5,
            gradient_accumulation_steps: int = 1,
            num_train_epochs: float = 3.0,
            train_batch_size: int = 32,
            warmup_proportion: float = 0.1,
            train_all: bool = False,
            use_bert_adam: bool = True,
    ):
        super().__init__()
        self.config = BertConfig.from_pretrained(pretrained_model, num_choices=4)
        self.bert = BertModel.from_pretrained(pretrained_model, config=self.config)
        self.duma = DUMALayer(d_model_size=self.config.hidden_size, num_heads=self.config.num_attention_heads)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        self.classifier = nn.Linear(self.config.hidden_size, 1)

        if not train_all:
            for param in self.bert.parameters():
                param.requires_grad = False
            for param in self.bert.pooler.parameters():
                param.requires_grad = True

        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_train_epochs = num_train_epochs
        self.train_batch_size = train_batch_size
        self.warmup_proportion = warmup_proportion
        self.use_bert_adam = use_bert_adam

        self.warmup_steps = 0
        self.total_steps = 0

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            # Get dataloader by calling it - train_dataloader() is called after setup() by default
            train_loader = self.train_dataloader()

            # Calculate total steps
            self.total_steps = int(len(train_loader.dataset)
                                   / self.train_batch_size / self.gradient_accumulation_steps * self.num_train_epochs)
            self.warmup_steps = int(self.total_steps * self.warmup_proportion)

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
        ) if self.use_bert_adam else torch.optim.Adam(
            optimizer_grouped_parameters,
            lr=self.learning_rate
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_output = outputs.last_hidden_state
        qa_seq_output, p_seq_output, qa_mask, p_mask = separate_seq2(last_output, input_ids)
        enc_output_qa, enc_output_p = self.duma(qa_seq_output, p_seq_output, qa_mask, p_mask)
        fused_output = torch.cat([enc_output_qa, enc_output_p], dim=1)
        pooled_output = torch.mean(fused_output, dim=1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                logits = self.classifier(dropout(pooled_output))
            else:
                logits += self.classifier(dropout(pooled_output))
        logits = logits / len(self.dropouts)
        reshaped_logits = F.softmax(logits.view(-1, num_choices), dim=1)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def compute(self, batch):
        outputs = self(
            input_ids=batch['input_ids'].reshape(batch['input_ids'].shape[0], 4, -1),
            token_type_ids=batch['token_type_ids'].reshape(batch['token_type_ids'].shape[0], 4, -1),
            attention_mask=batch['attention_mask'].reshape(batch['attention_mask'].shape[0], 4, -1),
            labels=batch['label'],
        )
        labels_hat = torch.argmax(outputs.logits, dim=1)
        correct_count = torch.sum(batch['label'] == labels_hat)
        return outputs.loss, correct_count

    def training_step(self, batch, batch_idx):
        loss, correct_count = self.compute(batch)
        self.log('train_loss', loss)
        self.log('train_acc', correct_count.float() / len(batch['label']))

        return loss

    def validation_step(self, batch, batch_idx):
        loss, correct_count = self.compute(batch)

        return {
            "val_loss": loss,
            "correct_count": correct_count,
            "batch_size": len(batch['label'])
        }

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        val_acc = sum([out["correct_count"] for out in outputs]).float() / sum(out["batch_size"] for out in outputs)
        val_loss = sum([out["val_loss"] for out in outputs]) / len(outputs)
        self.log('val_acc', val_acc)
        self.log('val_loss', val_loss)

    def test_step(self, batch, batch_idx):
        loss, correct_count = self.compute(batch)

        return {
            "test_loss": loss,
            "correct_count": correct_count,
            "batch_size": len(batch['label'])
        }

    def test_epoch_end(self, outputs: List[Any]) -> None:
        test_acc = sum([out["correct_count"] for out in outputs]).float() / sum(out["batch_size"] for out in outputs)
        test_loss = sum([out["test_loss"] for out in outputs]) / len(outputs)
        self.log('test_acc', test_acc)
        self.log('test_loss', test_loss)
