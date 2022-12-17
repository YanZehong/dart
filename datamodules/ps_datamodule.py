from .registry import register_datamodule
import pytorch_lightning as pl
import os
import csv
from transformers import BertTokenizer, BigBirdTokenizer, RobertaTokenizer, LongformerTokenizer
import torch
from torch.utils.data import DataLoader
from .components import build_collator
from .misc import truncate, persent_label_map
import json
import pickle

_pjoin = os.path.join


@register_datamodule("persent")
class PersentDatamodule(pl.LightningDataModule):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.keep_auxiliary = True

        if conf.model.arch in ["bert_truncation", "big_bird"]:
            self.file_path = '/home/zehong/DART/outputs/corpus_' + str(self.conf.data.name) + '_' + str(conf.model.arch) + '_' + str(self.conf.data.max_num_seq) + '.json'
            if not os.path.exists(self.file_path):
                self.convert_data_to_features()
                self.process()
            else:
                with open(self.file_path) as f:
                    self.packed_data = json.load(f)
                
        elif conf.model.arch in ["roberta_truncation", "longformer"]:
            self.file_path = '/home/zehong/DART/outputs/corpus_' + str(
                self.conf.data.name) + '_' + str(conf.model.arch) + '_' + str(
                    self.conf.data.max_num_seq) + '.pickle'
            if not os.path.exists(self.file_path):
                self.convert_data_to_features()
                self.process_for_roberta()
            else:
                with open(self.file_path, 'rb') as f:
                    self.packed_data = pickle.load(f)

        elif conf.model.arch in ["dart", "dart_base"]:
            self.file_path = '/home/zehong/DART/outputs/corpus_' + str(self.conf.data.name) + '_' + str(self.conf.data.max_num_sent) + '_' + str(self.conf.data.max_num_token_per_sent) + '.json'
            if not os.path.exists(self.file_path):
                self.convert_data_to_features()
                self.sentence_process()
            else:
                with open(self.file_path) as f:
                    self.packed_data = json.load(f)
                
    
    def convert_data_to_features(self):
        conf = self.conf
        data_dir = conf.data_dir
        dataset_map = {"train": "train.csv", "dev": "dev.csv", "test": "test.csv"}
        aspect_map = {
            'crime_justice_system': 'justice', 
            'digital_online': 'online',
            'economic_issues': 'economic',
            'public_health': 'health',
            'social_inequality_human_rights': 'rights', 
            'work_occupation': 'work',
        }
        output = {}
        
        for k, v in dataset_map.items():
            csv_path = _pjoin(data_dir, v)
            with open(csv_path, "r") as f:
                csv_reader = csv.reader(f, delimiter=",")
                for idx, row in enumerate(csv_reader):
                    if idx == 0:
                        aspect_names = row[7:10] + [row[12]] + [row[14]] + [row[16]]
                        aspects = [aspect_map[aspect_name] for aspect_name in aspect_names]
                        continue
                    mode = k
                    if mode not in output: output[mode] = []

                    doc_id = row[0]
                    doc_text = row[2]
                    doc_paragraphs = row[3]
                    doc_sentences = row[4]
                    doc_entity = row[5]
                    aspect_labels = row[7:10] + [row[12]] + [row[14]] + [row[16]]
                    aspect_labels = [int(x) for x in aspect_labels]
                    for aspect, label in zip(aspects, aspect_labels):
                        if label == -1 or label == 1 or label == 3: continue
                    
                        feature = {}
                        feature["doc_id"] = doc_id
                        feature["doc_text"] = doc_text
                        feature["doc_paragraphs"] = doc_paragraphs
                        feature["doc_sentences"] = doc_sentences
                        feature["doc_entity"] = doc_entity
                        feature["doc_aspect"] = aspect
                        feature["doc_aspect_label"] = label
                    
                        output[mode] += [feature]
        output["aspects2id"] = {aspect: idx for idx, aspect in enumerate(aspects)}
        self.packed_data = output
    
    def process(self):
        conf = self.conf
        if conf.model.arch == "bert_truncation":
            tokenizer = BertTokenizer.from_pretrained(self.conf.model.backbone)
        elif conf.model.arch == "big_bird":
            tokenizer = BigBirdTokenizer.from_pretrained(self.conf.model.backbone)
        
        aspects2id = self.packed_data["aspects2id"]
        for mode in ["train", "dev", "test"]:
            feature_list = []
            raw_data = self.packed_data[mode]

            for x in raw_data:
                doc = x["doc_text"] 
                doc_aspect = x["doc_aspect"]
                aspect_label = persent_label_map(x["doc_aspect_label"])

                tok_doc = tokenizer.tokenize(doc)
                if doc_aspect and self.keep_auxiliary:
                    tok_aspect = tokenizer.tokenize(doc_aspect)
                    trunc_tok_doc = truncate(
                        tok_doc,
                        max_len=conf.data.max_num_seq - 3 - len(tok_aspect),
                    )
                    concat_tok_doc = (["[CLS]"] + tok_aspect + ["[SEP]"] +
                                       trunc_tok_doc + ["[SEP]"])
                    segment_ids = [0] * (len(tok_aspect) +2) + [1] * (len(trunc_tok_doc) + 1)
                else:
                    trunc_tok_doc = truncate(tok_doc, max_len=conf.data.max_num_seq - 2)
                    concat_tok_doc = (["[CLS]"] + trunc_tok_doc + ["[SEP]"])
                    segment_ids = [0] * (len(trunc_tok_doc) + 2)
                
                doc_ids = tokenizer.convert_tokens_to_ids(concat_tok_doc)

                x.update({
                        "doc_list": concat_tok_doc,
                        "doc_ids": doc_ids,
                        "segment_ids": segment_ids,
                        "doc_aspect_id": aspects2id[doc_aspect],
                        "label_id": aspect_label,
                })
                feature_list += [x]
            self.packed_data[mode] = feature_list
            
        with open(self.file_path, "w") as fw:
            json.dump(self.packed_data, fw, indent=4)
    
    def process_for_roberta(self):
        '''
        Since the Longformer is based on RoBERTa, it doesn’t have token_type_ids. 
        You don’t need to indicate which token belongs to which segment. Just 
        separate your segments with the separation token tokenizer.sep_token (or </s>).
        '''
        conf = self.conf
        if conf.model.arch == "longformer":
            print("######## Use longformer tokenizer")
            tokenizer = LongformerTokenizer.from_pretrained(
                self.conf.model.backbone)
        elif conf.model.arch == "roberta_truncation":
            print("######## Use RoBERTa tokenizer")
            tokenizer = RobertaTokenizer.from_pretrained(
                self.conf.model.backbone)

        aspects2id = self.packed_data["aspects2id"]
        for mode in ["train", "dev", "test"]:
            feature_list = []
            raw_data = self.packed_data[mode]

            for x in raw_data:
                doc = x["doc_text"]
                doc_aspect = x["doc_aspect"]
                aspect_label = persent_label_map(x["doc_aspect_label"])
                encoded_input = tokenizer.encode_plus(
                    text=doc_aspect,
                    text_pair=doc,
                    max_length=conf.data.max_num_seq,
                    truncation="only_second",
                    padding="max_length",
                )
                x.update({
                    "doc_list": doc_aspect + ' ' + doc,
                    "doc_ids": encoded_input["input_ids"],
                    "attention_mask": encoded_input["attention_mask"],
                    "doc_aspect_id": aspects2id[doc_aspect],
                    "label_id": aspect_label,
                })
                feature_list += [x]
            self.packed_data[mode] = feature_list
        with open(self.file_path, "wb") as f:
            pickle.dump(self.packed_data, f)

    
    def sentence_process(self):
        conf = self.conf
        if self.conf.model.backbone == "bert-base-uncased":
            tokenizer = BertTokenizer.from_pretrained(self.conf.model.backbone)
        elif self.conf.model.backbone == "google/bigbird-roberta-base":
            tokenizer = BigBirdTokenizer.from_pretrained(self.conf.model.backbone)
        elif self.conf.model.backbone == "roberta-base":
            tokenizer = RobertaTokenizer.from_pretrained(self.conf.model.backbone)
        
        aspects2id = self.packed_data["aspects2id"]
        for mode in ["train", "dev", "test"]:
            feature_list = []
            raw_data = self.packed_data[mode]

            for x in raw_data:
                sentences = eval(x["doc_sentences"])
                # sentences = eval(x["doc_paragraphs"]) # process paragraph chunks
                doc_aspect = x["doc_aspect"]
                aspect_label = persent_label_map(x["doc_aspect_label"])
                
                sent_list = []
                sent_ids_list = []
                segment_ids_list = []
                sent_pos_ids_list = []

                for sent_idx, sent in enumerate(sentences):
                    tok_sent = tokenizer.tokenize(sent)
                    if doc_aspect and self.keep_auxiliary:
                        tok_aspect = tokenizer.tokenize(doc_aspect)
                        trunc_tok_sent = truncate(
                            tok_sent,
                            max_len=conf.data.max_num_token_per_sent - 3 - len(tok_aspect),
                        )
                        concat_tok_sent = (["[CLS]"] + tok_aspect + ["[SEP]"] +
                                        trunc_tok_sent + ["[SEP]"])
                        segment_ids = [0] * (len(tok_aspect) +2) + [1] * (len(trunc_tok_sent) + 1)
                    else:
                        trunc_tok_sent = truncate(tok_sent, max_len=conf.data.max_num_token_per_sent - 2)
                        concat_tok_sent = (["[CLS]"] + trunc_tok_sent + ["[SEP]"])
                        segment_ids = [0] * (len(trunc_tok_sent) + 2)
                    
                    sent_ids = tokenizer.convert_tokens_to_ids(concat_tok_sent)
                    
                    sent_list += [concat_tok_sent]
                    sent_ids_list += [sent_ids]
                    segment_ids_list += [segment_ids]
                    sent_pos_ids_list += [sent_idx+1]
                
                sent_list = truncate(sent_list, max_len=conf.data.max_num_sent)
                sent_ids_list = truncate(sent_ids_list, max_len=conf.data.max_num_sent)
                segment_ids_list = truncate(segment_ids_list, max_len=conf.data.max_num_sent)
                sent_pos_ids_list = truncate(sent_pos_ids_list, max_len=conf.data.max_num_sent)

                x.update({
                        "sent_list": sent_list,
                        "sent_ids_list": sent_ids_list,
                        "segment_ids_list": segment_ids_list,
                        "sent_pos_ids_list":sent_pos_ids_list,
                        "doc_aspect_id": aspects2id[doc_aspect],
                        "label_id": aspect_label,
                })
                feature_list += [x]
            self.packed_data[mode] = feature_list
        with open(self.file_path, "w") as fw:
            json.dump(self.packed_data, fw, indent=4)

    def train_dataloader(self):
        collate_fn = build_collator(conf=self.conf)
        loader = DataLoader(
            self.packed_data["train"],
            batch_size=self.conf.train.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.conf.train.num_workers,
            prefetch_factor=self.conf.train.prefetch_factor,
        )
        return loader

    def val_dataloader(self):
        collate_fn = build_collator(conf=self.conf)
        loader = DataLoader(
            self.packed_data["dev"],
            batch_size=self.conf.dev.batch_size,
            collate_fn=collate_fn,
            num_workers=self.conf.dev.num_workers,
            prefetch_factor=self.conf.dev.prefetch_factor,
        )
        return loader

    def test_dataloader(self):
        collate_fn = build_collator(conf=self.conf)
        loader = DataLoader(
            self.packed_data["test"],
            batch_size=self.conf.test.batch_size,
            collate_fn=collate_fn,
            num_workers=self.conf.test.num_workers,
            prefetch_factor=self.conf.test.prefetch_factor,
        )
        return loader