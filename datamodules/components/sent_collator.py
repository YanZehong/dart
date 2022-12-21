from .registry import register_collator
import torch

@register_collator("sent_collator")
class SentenceCollator:

    def __init__(self, conf):
        self.conf = conf

    def __call__(self, batch):
        conf = self.conf
        bsz = len(batch)
        
        sent_ids = torch.zeros(
            (bsz, conf.data.max_num_sent, conf.data.max_num_token_per_sent),
            dtype=torch.long)
        attention_mask = torch.zeros_like(sent_ids, dtype=torch.bool)
        segment_ids = torch.zeros_like(sent_ids, dtype=torch.long)
        sent_pos_ids = torch.zeros((bsz, conf.data.max_num_sent), dtype=torch.long)
        label_ids = torch.zeros((bsz, ), dtype=torch.long)
        aspect_ids = torch.zeros_like(label_ids, dtype=torch.long)

        for bsz_idx, x in enumerate(batch):
            sent_ids_list = x["sent_ids_list"] 
            segment_ids_list = x["segment_ids_list"]
            sent_pos_ids_list = x["sent_pos_ids_list"]
            doc_aspect_id_ = x["doc_aspect_id"]
            doc_label_ = x["label_id"]

            for j in range(len(sent_ids_list)):
                sent_pos_ids[bsz_idx, j] = sent_pos_ids_list[j]
                for k in range(len(sent_ids_list[j])):
                    sent_ids[bsz_idx, j, k] = sent_ids_list[j][k]
                    attention_mask[bsz_idx, j, k] = True
                    segment_ids[bsz_idx, j, k] = segment_ids_list[j][k]

            label_ids[bsz_idx] = doc_label_
            aspect_ids[bsz_idx] = doc_aspect_id_

        output = {
            "input_ids": sent_ids,
            "attention_mask": attention_mask,
            "token_type_ids": segment_ids,
            "sent_pos_ids": sent_pos_ids,
            "label_ids": label_ids,
            "aspect_ids": aspect_ids,
        }

        return output