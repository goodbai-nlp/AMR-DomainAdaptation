# Adapted from https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_mlm.py
# coding:utf-8
import warnings
import torch
import os
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from torch.nn.utils.rnn import pad_sequence


def padding_func(features, padding_side="right", pad_token_id=1, key="label"):
    assert key in features[0].keys(), f"{key} not in {features[0].keys()}"
    max_label_length = max(len(feature[key]) for feature in features)
    for feature in features:
        remainder = [pad_token_id] * (max_label_length - len(feature[key]))
        feature[key] = (
            feature[key] + remainder if padding_side == "right" else remainder + feature[key]
        )
    return


class LMDataSet(torch.nn.Module):
    def __init__(
        self,
        tokenizer,
        train_file,
        validation_file,
        line_by_line,
        pad_to_max_length=True,
        preprocessing_num_workers=4,
        overwrite_cache=False,
        cache_dir="",
        max_seq_length=512,
        mlm_probability=0.15,
        train_batch_size=8,
        val_batch_size=4,
        dataloader_num_workers=2,
        unified_input=True,
    ):
        super().__init__()
        self.train_file = train_file
        self.validation_file = validation_file
        self.tokenizer = tokenizer
        self.line_by_line = line_by_line
        self.pad_to_max_length = pad_to_max_length
        self.preprocessing_num_workers = preprocessing_num_workers
        self.overwrite_cache = overwrite_cache
        self.max_seq_length = max_seq_length
        self.mlm_probability = mlm_probability
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.cache_dir = cache_dir
        self.unified_inp = unified_input

    def setup(self, stage="fit"):
        extension = self.train_file.split(".")[-1]
        if extension in ("txt", "raw", "tsv"):
            extension = "text"

        data_files = {}
        data_files["train"] = self.train_file
        data_files["validation"] = self.validation_file
        datasets = load_dataset(
            extension, data_files=data_files, cache_dir=os.path.abspath(self.cache_dir)
        )

        column_names = datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        if self.line_by_line:
            # When using line_by_line, we just tokenize each nonempty line.
            padding = "max_length" if self.pad_to_max_length else False

            def tokenize_function(examples):
                # Remove empty lines
                examples["text"] = [
                    line for line in examples["text"] if len(line) > 0 and not line.isspace()
                ]

                if self.unified_inp:
                    ori_res = self.tokenizer(
                        examples["text"],
                        padding=padding,
                        truncation=True,
                        max_length=self.max_seq_length,
                        return_special_tokens_mask=True,
                    )
                    ori_res["labels"] = [itm[1:] for itm in ori_res["input_ids"]]
                    ori_res["seg_ids"] = [
                        list(map(int, [0 for _ in range(len(itm))] + [1, 1, 1]))
                        for itm in ori_res["input_ids"]
                    ]
                    ori_res["input_ids"] = [
                        list(
                            map(
                                int,
                                itm
                                + [
                                    self.tokenizer.amr_bos_token_id,
                                    self.tokenizer.mask_token_id,
                                    self.tokenizer.amr_eos_token_id,
                                ],
                            )
                        )
                        for itm in ori_res["input_ids"]
                    ]
                    ori_res["attention_mask"] = [
                        list(map(int, itm + [1, 1, 1])) for itm in ori_res["attention_mask"]
                    ]

                    return ori_res
                else:
                    ori_res = self.tokenizer(
                        examples["text"],
                        padding=padding,
                        truncation=True,
                        max_length=self.max_seq_length,
                        return_special_tokens_mask=True,
                    )
                    ori_res["labels"] = [itm[1:] for itm in ori_res["input_ids"]]
                    ori_res["seg_ids"] = [
                        [0 for _ in range(len(itm))] for itm in ori_res["attention_mask"]
                    ]
                    return ori_res

            tokenized_datasets = datasets.map(
                tokenize_function,
                batched=True,
                num_proc=self.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not self.overwrite_cache,
            )
        else:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
            # efficient when it receives the `special_tokens_mask`.
            def tokenize_function(examples):
                return self.tokenizer(examples[text_column_name], return_special_tokens_mask=True)

            tokenized_datasets = datasets.map(
                tokenize_function,
                batched=True,
                num_proc=self.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.overwrite_cache,
            )

            if self.max_seq_length is None:
                self.max_seq_length = self.tokenizer.model_max_length
            else:
                if self.max_seq_length > self.tokenizer.model_max_length:
                    warnings.warn(
                        f"The max_seq_length passed ({self.max_seq_length}) is larger than the maximum length for the"
                        f"model ({self.tokenizer.model_max_length}). Using max_seq_length={self.tokenizer.model_max_length}."
                    )
                self.max_seq_length = min(self.max_seq_length, self.tokenizer.model_max_length)

            # Main data processing function that will concatenate all texts from our dataset and generate chunks of
            # max_seq_length.
            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
                total_length = (total_length // self.max_seq_length) * self.max_seq_length
                # Split by chunks of max_len.
                result = {
                    k: [
                        t[i : i + self.max_seq_length]
                        for i in range(0, total_length, self.max_seq_length)
                    ]
                    for k, t in concatenated_examples.items()
                }
                return result

            # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
            # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
            # might be slower to preprocess.
            #
            # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=self.preprocessing_num_workers,
                load_from_cache_file=not self.overwrite_cache,
            )

        train_dataset = tokenized_datasets["train"]
        print(f"ALL {len(train_dataset)} training instances")
        eval_dataset = tokenized_datasets["validation"]
        print(f"ALL {len(eval_dataset)} validation instances")

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        # self.data_collator = data_collator

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.dataloader_num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.val_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.dataloader_num_workers,
        )

    def data_collator(self, features):
        padding_func(
            features, padding_side=self.tokenizer.padding_side, pad_token_id=2, key="seg_ids"
        )
        padding_func(
            features, padding_side=self.tokenizer.padding_side, pad_token_id=-100, key="labels"
        )
        features = self.tokenizer.pad(
            features, padding=True, pad_to_multiple_of=8, return_tensors="pt",
        )
        return features
