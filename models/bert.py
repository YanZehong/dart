import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import register_model
from transformers import BertModel

@register_model("bert_truncation")
class BertTruncation(nn.Module):
    """
        The bert truncation model for document-level aspect-based sentiment classification.
    """
    def __init__(self, conf) -> None:
        super().__init__()

        self.conf = conf
        model_conf = conf.model
        data_conf = conf.data

        self.bert = BertModel.from_pretrained(model_conf.backbone)

        bert_hidden_size = self.bert.config.hidden_size

        self.clf = nn.Sequential(
            nn.Linear(bert_hidden_size, bert_hidden_size),
            nn.Tanh(),
            nn.Dropout(p=model_conf.dropout),
            nn.Linear(bert_hidden_size, model_conf.num_class),
        )

    def forward(self, input_ids, attention_mask, token_type_ids, **kwargs):
        """
        Input format: 
            [CLS] <aspect> [SEP] <doc>
        Args:
            input_ids (batch_size, sequence_length):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (batch_size, sequence_length):
                Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            token_type_ids (batch_size, sequence_length):
                Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]:
                - 0 corresponds to a *sentence A* token,
                - 1 corresponds to a *sentence B* token.
        """

        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        last_hidden_states = outputs.last_hidden_state

        doc_emb = last_hidden_states[:,0,:]
        logits = self.clf(doc_emb)

        return {"logits": logits}