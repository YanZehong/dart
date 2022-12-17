from .registry import register_collator
import torch


@register_collator("doc_collator")
class DocumentCollator:

    def __init__(self, conf):
        self.conf = conf

    def __call__(self, batch):
        conf = self.conf
        bsz = len(batch)

        input_ids = torch.zeros((bsz, conf.data.max_num_seq),
                                dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids, dtype=torch.float)
        token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
        
        label_ids = torch.zeros((bsz, ), dtype=torch.long)
        aspect_ids = torch.zeros_like(label_ids, dtype=torch.long)

        for bsz_idx, x in enumerate(batch):
            doc_ids = x["doc_ids"]
            segment_ids = x["segment_ids"]
            doc_label_ = x["label_id"]
            doc_aspect_id_ = x["doc_aspect_id"]
            
            for k in range(len(doc_ids)):
                input_ids[bsz_idx, k] = doc_ids[k]
                attention_mask[bsz_idx, k] = 1
                token_type_ids[bsz_idx, k] = segment_ids[k]

            label_ids[bsz_idx] = doc_label_
            aspect_ids[bsz_idx] = doc_aspect_id_

        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "aspect_ids": aspect_ids,
            "label_ids": label_ids,
            # "raw": batch
        }

        return output