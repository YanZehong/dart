import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import register_model
from .modeling_dart import AttentiveAggregation, LocalAttentiveAggregation
from transformers import BertModel, BigBirdModel, RobertaModel, AutoModel
import pickle


@register_model("dart")
class HierarchicalDART(nn.Module):

    def __init__(self, conf) -> None:
        super().__init__()

        self.conf = conf
        interact_encoder_conf = self.conf.model.interact_encoder
        refine_encoder_conf = self.conf.model.refine_encoder

        if self.conf.model.backbone in ["bert-base-uncased", "bert-large-uncased"]:
            self.sent_encoder = BertModel.from_pretrained(self.conf.model.backbone)
        elif self.conf.model.backbone in ["google/bigbird-roberta-base", "google/bigbird-roberta-large"]:
            self.sent_encoder = BigBirdModel.from_pretrained(self.conf.model.backbone)
        
        if self.conf.data.name == "trip_advisor":
            print("################ Loading aspect embeddings of TripAdvisor")
            emb_name = "ta_aspects_" + str(interact_encoder_conf.d_model) + "_emb.dat"
        elif self.conf.data.name == "beer_advocate":
            print("################ Loading aspect embeddings of BeerAdvocate")
            emb_name = "ba_aspects_" + str(interact_encoder_conf.d_model) + "_emb.dat"
        elif self.conf.data.name == "persent":
            print("################ Loading aspect embeddings of PerSenT")
            emb_name = "ps_6aspects_" + str(interact_encoder_conf.d_model) + "_emb.dat"
        
        with open (self.conf.root_dir + "/dataset/aspect_embs/" + emb_name, "rb") as f:
            self.fixed_aspect_emb = pickle.load(f)

        self.pos_emb_layer = nn.Embedding(self.conf.data.max_num_sent+1,
                                          interact_encoder_conf.d_model,
                                          padding_idx=0)

        interact_trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=interact_encoder_conf.d_model,
            nhead=interact_encoder_conf.num_head,
            dim_feedforward=interact_encoder_conf.ff_dim,
            dropout=interact_encoder_conf.dropout,
            batch_first=True,
        )

        self.interact_encoder = nn.TransformerEncoder(
            interact_trans_encoder_layer, num_layers=interact_encoder_conf.num_layers)

        refine_trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=refine_encoder_conf.d_model,
            nhead=refine_encoder_conf.num_head,
            dim_feedforward=refine_encoder_conf.ff_dim,
            dropout=refine_encoder_conf.dropout,
            batch_first=True,
        )
        self.refine_encoder = nn.TransformerEncoder(
            refine_trans_encoder_layer, num_layers=refine_encoder_conf.num_layers)
        
        self.aspect_emb_layer = nn.Embedding.from_pretrained(self.fixed_aspect_emb, freeze=False)

        self.local_pooling = LocalAttentiveAggregation(input_size=refine_encoder_conf.d_model)
        self.global_pooling = AttentiveAggregation(input_size=refine_encoder_conf.d_model)

        self.clf = nn.Sequential(
            nn.Linear(refine_encoder_conf.d_model, refine_encoder_conf.d_model),
            nn.Tanh(),
            nn.Dropout(p=self.conf.model.dropout),
            nn.Linear(refine_encoder_conf.d_model, self.conf.model.num_class),
        )
    
    def forward(self, input_ids, attention_mask, token_type_ids, sent_pos_ids, aspect_ids, **kwargs):
        """
        input sentence format: [CLS] <aspect> [SEP] <sent>
        input_ids: [bsz, num_sent, num_token]
        attention_mask: [bsz, num_sent, num_token]
        token_type_ids: [bsz, num_sent, num_token]
        sent_pos_ids: [bsz, num_sent] (0 for padding)
        aspect_ids: [bsz, ]
        """
    
        bsz, num_sent, num_token = input_ids.shape
        sent_mask = torch.clone(attention_mask[:, :, 0]).detach() # [bsz, num_sent]

        """1st: sentence encoder"""
        flatten_input_ids = input_ids.reshape((bsz * num_sent, num_token)) # [bsz * num_sent, num_token]
        flatten_attention_mask = attention_mask.reshape((bsz * num_sent, num_token)) # [bsz * num_sent, num_token]
        flatten_attention_mask[:, 0] = True
        flatten_token_type_ids = token_type_ids.reshape((bsz * num_sent, num_token)) # [bsz * num_sent, num_token]
        
        sent_embs = self.sent_encoder(input_ids=flatten_input_ids,
                            attention_mask=flatten_attention_mask,
                            token_type_ids=flatten_token_type_ids).last_hidden_state # [bsz * num_sent, num_token, hidden_size]
        
        cls_embs = sent_embs[:, 0, :]  # [bsz * num_sent, hidden_size]

        """2nd: interaction"""
        pos_emb = self.pos_emb_layer(sent_pos_ids) # [bsz, num_sent, hidden_size]
        cls_embs = cls_embs.reshape((bsz, num_sent, -1)) + pos_emb # [bsz, num_sent, hidden_size]
        cls_embs = self.interact_encoder(src=cls_embs, src_key_padding_mask=~sent_mask) # [bsz, num_sent, hidden_size]
        
        """3rd: refinement"""
        sent_embs = torch.cat(
            [cls_embs.reshape((bsz * num_sent, 1, -1)), sent_embs[:, 1:, :]],
            dim=1)  # [bsz * num_sent, num_token, hidden_size_2]

        sent_embs = self.refine_encoder(
            sent_embs,
            src_key_padding_mask=~flatten_attention_mask,
        ).reshape((bsz, num_sent, num_token, -1)) # [bsz, num_sent, num_token, hidden_size]

        """4th: modified aspect-specific aggregation"""
        local_embs = self.local_pooling(sent_embs, attention_mask) # [bsz, num_sent, hidden_size]
        aspect_emb = self.aspect_emb_layer(aspect_ids)
        doc_emb = self.global_pooling(local_embs, sent_mask, aspect_emb) # [bsz, hidden_size]

        logits = self.clf(doc_emb)
        
        return {"logits": logits}