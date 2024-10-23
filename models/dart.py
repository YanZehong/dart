import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import register_model
from .modeling_dart import AttentiveAggregation, LocalAttentiveAggregation
from transformers import BigBirdModel, BigBirdConfig, AutoModel
import pickle


@register_model("dart")
class HierarchicalDART(nn.Module):
    """
        The DART model for document-level aspect-based sentiment classification.
    """
    def __init__(self, conf) -> None:
        super().__init__()

        self.conf = conf
        interact_encoder_conf = self.conf.model.interact_encoder
        refine_encoder_conf = self.conf.model.refine_encoder
        if self.conf.model.backbone:
            self.sent_encoder = AutoModel.from_pretrained(self.conf.model.backbone, attention_type = "original_full")
        else:
            self.sent_encoder = BigBirdModel(BigBirdConfig())
        
        if self.conf.data.num_aspect > 0:
            if self.conf.data.name == "trip_advisor":
                print("################ Loading aspect embeddings of TripAdvisor")
                emb_name = "ta_aspects_" + str(interact_encoder_conf.d_model) + "_emb.dat"
            elif self.conf.data.name == "beer_advocate":
                print("################ Loading aspect embeddings of BeerAdvocate")
                emb_name = "ba_aspects_" + str(interact_encoder_conf.d_model) + "_emb.dat"
            elif self.conf.data.name == "social_news":
                print("################ Loading aspect embeddings of SocialNews")
                emb_name = "sn_6aspects_" + str(interact_encoder_conf.d_model) + "_emb.dat"
            file_name = self.conf.root_dir + "/dataset/aspect_embs/" + self.conf.model.backbone.split('-')[0] + "_" + emb_name
            print(f"################ Loading from {file_name}")
            with open (file_name, "rb") as f:
                self.fixed_aspect_emb = pickle.load(f)
            self.aspect_emb_layer = nn.Embedding.from_pretrained(self.fixed_aspect_emb, freeze=False)
        else:
            self.aspect_emb_layer = nn.Embedding(1,interact_encoder_conf.d_model)

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
        
        self.query_cls = False if self.conf.data.num_aspect > 0 else True
        self.local_pooling = LocalAttentiveAggregation(input_size=refine_encoder_conf.d_model, query_cls=self.query_cls)
        self.global_pooling = AttentiveAggregation(input_size=refine_encoder_conf.d_model)

        self.clf = nn.Sequential(
            nn.Linear(refine_encoder_conf.d_model, refine_encoder_conf.d_model),
            nn.Tanh(),
            nn.Dropout(p=self.conf.model.dropout),
            nn.Linear(refine_encoder_conf.d_model, self.conf.model.num_class),
        )
    
    def forward(self, input_ids, attention_mask, token_type_ids, sent_pos_ids, aspect_ids, **kwargs):
        """
        Input format: 
            [CLS] <aspect> [SEP] <doc>
        Args:
            input_ids (batch_size, num_sent, num_token):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (batch_size, num_sent, num_token):
                Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            token_type_ids (batch_size, num_sent, num_token):
                Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]:
                - 0 corresponds to a *sentence A* token,
                - 1 corresponds to a *sentence B* token.
            sent_pos_ids (batch_size, num_sent):
                Positional encodings to the input embeddings at the interaction block, in order for the transformer to make 
                full use of the order of sentences in a document. (0 for padding)
            aspect_ids (batch_size, ):
                Indices indicate the aspect of the corresponding sentiment label.
        """
    
        bsz, num_sent, num_token = input_ids.shape
        sent_mask = torch.clone(attention_mask[:, :, 0].bool()).detach() 

        """1st: Sentence Encoding Block"""
        flatten_input_ids = input_ids.reshape((bsz * num_sent, num_token))
        flatten_attention_mask = attention_mask.reshape((bsz * num_sent, num_token)) 
        # flatten_attention_mask[:, 0] = True
        flatten_attention_mask[:, 0] = 1.0
        flatten_token_type_ids = token_type_ids.reshape((bsz * num_sent, num_token)) 
        
        if 't5' in self.conf.model.backbone:
            sent_embs = self.sent_encoder(input_ids=flatten_input_ids,
                            attention_mask=flatten_attention_mask).last_hidden_state 
        else:
            sent_embs = self.sent_encoder(input_ids=flatten_input_ids,
                                attention_mask=flatten_attention_mask,
                                token_type_ids=flatten_token_type_ids).last_hidden_state 
            
        cls_embs = sent_embs[:, 0, :]  

        """2nd: Global Context Interaction Block"""
        pos_emb = self.pos_emb_layer(sent_pos_ids) 
        cls_embs = cls_embs.reshape((bsz, num_sent, -1)) + pos_emb 
        cls_embs = self.interact_encoder(src=cls_embs, src_key_padding_mask=~sent_mask) 
        
        
        sent_embs = torch.cat(
            [cls_embs.reshape((bsz * num_sent, 1, -1)), sent_embs[:, 1:, :]],
            dim=1) 

        sent_embs = self.refine_encoder(
            sent_embs,
            src_key_padding_mask=~flatten_attention_mask.bool(),
        ).reshape((bsz, num_sent, num_token, -1)) 

        """3rd: Aspect Aggregation Block"""
        aspect_embs = self.aspect_emb_layer(aspect_ids)
        
        local_embs = self.local_pooling(sent_embs, attention_mask.bool())   
        doc_emb = self.global_pooling(local_embs, sent_mask, aspect_embs) 

        """4th: MLP"""
        logits = self.clf(doc_emb)
        
                
        output_dict = {"logits": logits}
        return output_dict
        