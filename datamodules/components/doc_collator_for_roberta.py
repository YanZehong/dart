from .registry import register_collator
import torch


@register_collator("doc_collator_for_roberta")
class DocumentCollatorForRoBERTa:

    def __init__(self, conf):
        self.conf = conf

    def __call__(self, batch):
        conf = self.conf
        bsz = len(batch)
        input_ids = torch.ones((bsz, conf.data.max_num_seq), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids, dtype=torch.float)
        global_attention_mask = torch.zeros((bsz, conf.data.max_num_seq),
                                            dtype=torch.float)

        label_ids = torch.zeros((bsz, ), dtype=torch.long)
        aspect_ids = torch.zeros_like(label_ids, dtype=torch.long)

        for bsz_idx, x in enumerate(batch):

            doc_label_ = x["label_id"]
            doc_aspect_id_ = x["doc_aspect_id"]

            input_ids[bsz_idx, :] = torch.tensor(x["doc_ids"],
                                                 dtype=torch.long)
            attention_mask[bsz_idx, :] = torch.tensor(x["attention_mask"],
                                                      dtype=torch.float)
            

            label_ids[bsz_idx] = doc_label_
            aspect_ids[bsz_idx] = doc_aspect_id_
        global_attention_mask[:, [0, 1]] = 1

        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "global_attention_mask": global_attention_mask,
            "aspect_ids": aspect_ids,
            "label_ids": label_ids,
        }

        return output