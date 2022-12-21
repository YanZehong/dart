import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentiveAggregation(nn.Module):

    def __init__(self, input_size) -> None:
        super().__init__()

        self.hidden_map = nn.Sequential(nn.Linear(input_size, input_size),
                                        nn.Tanh())

    def forward(self, embs, emb_mask, q):
        """
        embs: [bsz, num_seq, hidden_size]
        emb_mask: [bsz, num_seq]
        q: [bsz, hidden_size]
        Return: [bsz, hidden_size]
        """
        hidden_size = torch.tensor(q.shape[1], dtype=torch.float)
        attn_logit = torch.sum(self.hidden_map(embs) * q[:, None, :], dim=2)  # [bsz, num_seq]
        
        attn_logit = torch.where(emb_mask, attn_logit, torch.ones_like(attn_logit) * -torch.inf) # [bsz, num_seq]
        attn_score = torch.softmax(attn_logit / torch.sqrt(hidden_size), dim=1)  # scaling factor [bsz, num_seq]
        
        attn_emb = torch.sum(attn_score[:, :, None] * embs, dim=1)  # [bsz, hidden_size]

        return attn_emb

class LocalAttentiveAggregation(nn.Module):

    def __init__(self, input_size) -> None:
        super().__init__()
        self.local_pooling = AttentiveAggregation(input_size)

    def forward(self, embs, emb_mask):
        """
        embs: [bsz, num_sent, num_token, hidden_size]
        emb_mask: [bsz, num_sent, num_token]
        Return: [bsz, num_sent, hidden_size]
        """
        bsz, num_sent, num_token, hidden_size = embs.shape
        local_embs = torch.zeros((bsz, num_sent, hidden_size), dtype=embs.dtype, device=embs.device)
        for i in range(num_sent):
            local_embs[:, i, :] = self.local_pooling(embs[:, i, :, :], emb_mask[:, i, :], embs[:, i, 1, :])
        return local_embs