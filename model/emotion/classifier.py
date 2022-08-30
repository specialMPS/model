import torch
import torch.nn as nn
from kobert_transformers import get_kobert_model
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel

from model.model.emotion.configuration import get_kobert_config


class KoBERTforSequenceClassification(BertPreTrainedModel):
    def __init__(self,
                 num_labels=9,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 ):
        super().__init__(get_kobert_config())

        self.num_labels = num_labels
        self.kobert = get_kobert_model()
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.kobert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)