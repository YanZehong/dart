import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import register_model
from transformers import LongformerModel

@register_model("longformer")
class Longformer(nn.Module):
    """
        The Longformer model for document-level aspect-based sentiment classification.
    """
    def __init__(self, conf) -> None:
        super().__init__()

        self.conf = conf
        model_conf = conf.model
        data_conf = conf.data
        
        self.longformer = LongformerModel.from_pretrained(model_conf.backbone)
        self.hidden_size = self.longformer.config.hidden_size
        self.clf = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(p=model_conf.dropout),
            nn.Linear(self.hidden_size, model_conf.num_class),
        )

    def forward(self, input_ids, attention_mask, global_attention_mask, **kwargs):
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
            global_attention_mask (batch_size, sequence_length):
                Mask to decide the attention given on each token, local attention or global attention. Tokens with global
                attention attends to all other tokens, and all other tokens attend to them. This is important for
                task-specific finetuning because it makes the model more flexible at representing the task. For example,
                for classification, the <s> token should be given global attention. For QA, all question tokens should also
                have global attention. Please refer to the [Longformer paper](https://arxiv.org/abs/2004.05150) for more
                details. Mask values selected in `[0, 1]`:
                - 0 for local attention (a sliding window attention),
                - 1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).
        """
        
        outputs = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask).last_hidden_state

        logits = self.clf(outputs[:, 0, :])
        
        return {"logits": logits}

   