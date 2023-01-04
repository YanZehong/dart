from .registry import register_datamodule
import pytorch_lightning as pl
import os
import csv
from transformers import BertTokenizer, BigBirdTokenizer, RobertaTokenizer, LongformerTokenizer
import torch
from torch.utils.data import DataLoader
from .components import build_collator
from .misc import truncate, beer_advocate_label_map
import json
import pickle

_pjoin = os.path.join

@register_datamodule("beer_advocate")
class BeerAdvocateDatamodule(pl.LightningDataModule):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.keep_auxiliary = True

        if conf.model.arch in ["bert_truncation", "big_bird"]:
            self.file_path = self.conf.root_dir + '/outputs/corpus_' + str(
                self.conf.data.name) + '_' + str(conf.model.arch) + '_' + str(
                    self.conf.data.max_num_seq) + '.pickle'
            if not os.path.exists(self.file_path):
                self.convert_data_to_features()
                self.process()
            else:
                with open(self.file_path, 'rb') as f:
                    self.packed_data = pickle.load(f)
                
        elif conf.model.arch in ["longformer"]:
            self.file_path = self.conf.root_dir + '/outputs/corpus_' + str(
                self.conf.data.name) + '_' + str(conf.model.arch) + '_' + str(
                    self.conf.data.max_num_seq) + '.pickle'
            if not os.path.exists(self.file_path):
                self.convert_data_to_features()
                self.process_for_roberta()
            else:
                with open(self.file_path, 'rb') as f:
                    self.packed_data = pickle.load(f)

        elif conf.model.arch in ["dart"]:
            self.file_path = self.conf.root_dir + '/outputs/corpus_' + str(
                self.conf.data.name) + '_' + str(self.conf.data.max_num_sent) + '_' + str(
                    self.conf.data.max_num_token_per_sent) + '.pickle'
            if not os.path.exists(self.file_path):
                self.convert_data_to_features()
                self.sentence_process()
            else:
                with open(self.file_path, 'rb') as f:
                    self.packed_data = pickle.load(f)
    
    def convert_data_to_features(self):
        conf = self.conf
        data_dir = conf.data_dir
        dataset_map = {"train": "train.csv", "dev": "dev.csv", "test": "test.csv"}
        output = {}
        
        for k, v in dataset_map.items():
            csv_path = _pjoin(data_dir, v)
            with open(csv_path, "r") as f:
                csv_reader = csv.reader(f, delimiter=",")
                for idx, row in enumerate(csv_reader):
                    if idx == 0:
                        aspects = row[5:]
                        continue
                    mode = k
                    if mode not in output: output[mode] = []

                    doc_id = row[0]
                    doc_text = row[2]
                    doc_sentences = row[3]
                    overall_rating = row[4]
                    aspect_ratings = row[5:]
                    aspect_ratings = [int(x) for x in aspect_ratings]
                    for aspect, rating in zip(aspects, aspect_ratings):
                        if rating == -1: continue
                        feature = {}
                        feature["doc_id"] = doc_id
                        feature["doc_text"] = doc_text
                        feature["doc_sentences"] = doc_sentences
                        feature["overall_rating"] = overall_rating
                        feature["doc_aspect"] = aspect
                        feature["doc_aspect_rating"] = rating
                    
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
                aspect_label = beer_advocate_label_map(x["doc_aspect_rating"])

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
            
        with open(self.file_path, "wb") as f:
            pickle.dump(self.packed_data, f)
    
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
                aspect_label = beer_advocate_label_map(x["doc_aspect_rating"])
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
                doc_aspect = x["doc_aspect"]
                aspect_label = beer_advocate_label_map(x["doc_aspect_rating"])
                
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
        with open(self.file_path, "wb") as f:
            pickle.dump(self.packed_data, f)

    def sentence_process_for_roberta(self):
        conf = self.conf
        print(f"########loading {conf.model.backbone}")
        tokenizer = RobertaTokenizer.from_pretrained(self.conf.model.backbone)
        
        aspects2id = self.packed_data["aspects2id"]
        for mode in ["train", "dev", "test"]:
            feature_list = []
            raw_data = self.packed_data[mode]

            for x in raw_data:
                sentences = eval(x["doc_sentences"])
                doc_aspect = x["doc_aspect"]
                aspect_label = beer_advocate_label_map(x["doc_aspect_rating"])
                
                sent_list = []
                sent_ids_list = []
                attention_mask_list = []
                token_type_ids_list = []
                sent_pos_ids_list = []

                for sent_idx, sent in enumerate(sentences):
                    inputs = tokenizer.encode_plus(
                        text=doc_aspect,
                        text_pair=sent,
                        padding='max_length',
                        truncation="only_second",
                        max_length=conf.data.max_num_token_per_sent,
                        return_attention_mask=True,
                        return_token_type_ids=True
                    )
                    concat_sent = (["<s>"] + [doc_aspect] + ["</s>", "</s>"] +
                                        [sent] + ["[</s>"])
                    
                    sent_list += [concat_sent]
                    sent_ids_list += [inputs.input_ids]
                    attention_mask_list += [inputs.attention_mask]
                    token_type_ids_list += [inputs.token_type_ids]
                    sent_pos_ids_list += [sent_idx+1]
                
                sent_list = truncate(sent_list, max_len=conf.data.max_num_sent)
                sent_ids_list = truncate(sent_ids_list, max_len=conf.data.max_num_sent)
                attention_mask_list = truncate(attention_mask_list, max_len=conf.data.max_num_sent)
                token_type_ids_list = truncate(token_type_ids_list, max_len=conf.data.max_num_sent)
                sent_pos_ids_list = truncate(sent_pos_ids_list, max_len=conf.data.max_num_sent)

                x.update({
                        "sent_list": sent_list,
                        "sent_ids_list": sent_ids_list,
                        "attention_mask_list": attention_mask_list,
                        "token_type_ids_list": token_type_ids_list,
                        "sent_pos_ids_list":sent_pos_ids_list,
                        "doc_aspect_id": aspects2id[doc_aspect],
                        "label_id": aspect_label,
                })
                feature_list += [x]
            self.packed_data[mode] = feature_list
        with open(self.file_path, "wb") as f:
            pickle.dump(self.packed_data, f)


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
            batch_size=self.conf.test.batch_size, # 1
            collate_fn=collate_fn,
            num_workers=self.conf.test.num_workers, # 0
            prefetch_factor=self.conf.test.prefetch_factor, # 2
        )
        return loader