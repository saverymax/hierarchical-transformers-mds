"""
Module for defining datasets and handling all things data
"""
import json
import logging
import functools
import sys

import nlp
import numpy as np

import torch
from torch.utils.data import Dataset


def shift_input_right(input_ids, pad_token_id, eos_token_id):
    """
    Following BART finetuning and HuggingFace implementation, shift input ids
    one token to the right by wrapping the last non pad token (such as <\s> or <eos>) to the beginning.
    See https://github.com/pytorch/fairseq/issues/1389 for a good elaboration about ways to do this.
    """
    shifted_tokens = input_ids.clone()
    index_of_pad = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    # Add EOS token to the end of the sentence here, as it was not done during tokenization
    #shifted_tokens[:, 0] = input_ids.gather(1, index_of_pad).squeeze()
    #shifted_tokens[:, 1:] = input_ids[:, :-1]
    shifted_tokens[:, 0] = eos_token_id
    shifted_tokens[:, 1:] = input_ids[:, :-1]
    #ids_with_eos = input_ids.scatter(1, torch.tensor(index_of_pad), eos_token_id)

    return shifted_tokens


def pad_docs(articles, max_docs, article_key=None):
    """
    General function to pad set of articles to number of max docs
    Tokenizer will automatically pad the blank sequences"""
    if isinstance(articles, dict):
        articles = [articles[article_key]]
    if isinstance(articles, str):
        articles = [articles]
    assert isinstance(articles, list)
    if len(articles) < max_docs:
        pad_docs = ["" for i in range(max_docs - len(articles))]
        articles.extend(pad_docs)
    elif len(articles) > max_docs:
        articles = articles[:max_docs]
    assert len(articles) == max_docs

    if article_key is not None:
        return {article_key: articles}
    else:
        return articles


def sample_docs(articles, max_docs, all_articles, article_key=None):
    """
    Sample up to max_docs from the dataset, to use as multiple documents for summarization
    """
    # For the huggingface dataset!
    if isinstance(articles, dict):
        # For ELI5 dataset, docs will be list of potential answers.
        if isinstance(articles[article_key], list):
            articles = articles[article_key]
        else:
            articles = [articles[article_key]]
    if isinstance(articles, str):
        articles = [articles]
    assert isinstance(articles, list)
    if len(articles) < max_docs:
        indices = np.random.randint(0, len(all_articles), size=(max_docs - len(articles)))
        sample = [all_articles[i] for i in indices]
        articles.extend(sample)
    elif len(articles) > max_docs:
        articles = articles[:max_docs]
    assert len(articles) == max_docs
    np.random.shuffle(articles)
    # map wants dict returned
    if article_key is not None:
        return {article_key: articles}
    else:
        return articles


class DatasetRegistry():
    """Register each dataset"""
    def __init__(self):
        self.tasks = {
        'mediqa': MediqaAns,
        'bioasq': Bioasq,
        'ebm': EBM,
        'medlineplus': MedlineplusReviews,
        'cnn_dailymail': CnnDm,
        'eli5': ELI5,
        }

    def get_tasks(self):
        return self.tasks


class ELI5():
    """Iterable HF NLP ELI5 dataset"""

    def __init__(self, tokenizer, max_seq_len, max_docs, eos_token, path=None, cache_dir=None, doc_mixing=False, split="train"):
        logging.info("Loading ELI5 dataset")
        task = "eli5"
        if split == "val":
            split = "validation"
        self.prompt = "<TASK> {} <QUESTION> ".format(task)
        self.dataset = nlp.load_dataset(task, split="{}_eli5[:20]".format(split), cache_dir=cache_dir)
        self.tokenizer = tokenizer
        self.doc_mixing = doc_mixing
        self.max_docs = max_docs
        self.eos_token = eos_token
        logging.info("eos %s", eos_token)
        # Fix weird writing problem with urls. Only one particular one broke it, but IDK
        self.dataset = self.dataset.map(self.fix_urls)
        # Add EOS and grab the first answer in the list of answers as the "summary"
        #self.dataset = self.dataset.map(self.add_eos)
        # Using title and not selftext as query as there are a good number of "" selftext entries
        self.dataset = self.dataset.map(lambda example: {'title': self.prompt + example['title']})
        # Map all answers to 'document key', so my src documents are a list of answers which can be padded/mixed
        self.dataset = self.dataset = self.dataset.map(lambda example: {'document': example['answers']['text']})
        logging.info(self.dataset['document'])
        if self.doc_mixing:
            logging.info("Mixing in random documents")
            # Need to get answers because they're all nested in funny ways
            all_articles = [a for example in self.dataset for a in example['answers']['text']]
            logging.info("Number of all articles: %s", len(all_articles))
            # Article key required to keep pad_documents applicable to non-hf datasets
            doc_filling = functools.partial(sample_docs, max_docs=self.max_docs, all_articles=all_articles, article_key='document')
        else:
            logging.info("Padding documents with padding token")
            doc_filling = functools.partial(pad_docs, max_docs=self.max_docs, article_key='document')
        self.dataset = self.dataset.map(doc_filling)
        logging.info("doc after pad %s", self.dataset['document'])
        self.dataset = self.dataset.map(self.convert_to_features, batched=True)
        # Format dataset to outputs torch.Tensor to train a pytorch model
        columns_to_return = ['query_ids', 'query_mask', 'target_ids', 'target_mask', 'source_ids', 'source_mask']
        self.dataset.set_format(type='torch',
            columns=columns_to_return)

    def fix_urls(self, example):
        return {'selftext_urls': [], 'answers_urls': []}

    def add_eos(self, example):
        answer = example['answers']['text'][0]
        answer = answer + " " + self.eos_token
        return {'summary': answer}

    def get_dataset(self):
        return self.dataset

    def convert_to_features(self, example_batch):
        """
        Pad docs and tokenize iterable HF NLP dataset
        HF map provides input of datasets[:batch_size]
        """
        tgt_encodings = self.tokenizer(example_batch['summary'])
        query_encodings = self.tokenizer(example_batch['title'])
        src_encodings = [self.tokenizer(articles) for articles in example_batch['document']]
        input_ids = []
        attention_mask = []
        # Collate the tokenized sets of articles. For each example, there will
        # be a set of n articles, which looks like
        # {'input_ids': [article1 ids, ... , articlen ids]}. This needs to be collated
        # so the input_ids corresponds to the ids for all articles in a batch
        for i in src_encodings:
           input_ids.append(i['input_ids'])
           attention_mask.append(i['attention_mask'])

        return {
            "source_ids": input_ids,
            "source_mask": attention_mask,
            "query_ids": query_encodings['input_ids'],
            "query_mask": query_encodings['attention_mask'],
            "target_ids": tgt_encodings['input_ids'],
            "target_mask": tgt_encodings['attention_mask']}


class CnnDm():

    def __init__(self, tokenizer, max_seq_len, max_docs, eos_token, path=None, cache_dir=None, doc_mixing=False, split=None):
        logging.info("Loading CNN Dailymail dataset")
        task = "cnn_dailymail"
        if split == "val":
            split = "validation"
        self.dataset = nlp.load_dataset(task, '3.0.0', split="{}[:20]".format(split), cache_dir=cache_dir)
        self.tokenizer = tokenizer
        self.doc_mixing = doc_mixing
        self.max_docs = max_docs
        self.eos_token = eos_token
        print(self.dataset)
        #self.dataset = self.dataset.map(lambda example: {'highlights': example['highlights'] + " " + self.eos_token})
        # As CNN is a single-doc dataset, provide each article to pad or sample docs in a []
        if self.doc_mixing:
            logging.info("Mixing in random documents")
            # Article key required to keep pad_documents applicable to non-hf datasets
            doc_filling = functools.partial(sample_docs, max_docs=self.max_docs, all_articles=self.dataset['article'], article_key='article')
        else:
            logging.info("Padding documents with padding token")
            doc_filling = functools.partial(pad_docs, max_docs=self.max_docs, article_key='article')
        #self.dataset['article'][:] = [doc_filling(i) for i in self.dataset['article'][:]]
        self.dataset = self.dataset.map(doc_filling)
        self.dataset = self.dataset.map(self.convert_to_features, batched=True)
        print(len(self.dataset['source_ids'][:]))
        # Format dataset to outputs torch.Tensor to train a pytorch model
        columns_to_return = ['target_ids', 'target_mask', 'source_ids', 'source_mask']
        self.dataset.set_format(type='torch',
            columns=columns_to_return)

    def get_dataset(self):
        return self.dataset

    def convert_to_features(self, example_batch):
        """
        Pad docs and tokenize iterable HF NLP dataset
        HF map provides input of datasets[:batch_size]
        """
        tgt_encodings = self.tokenizer(example_batch['highlights'])
        src_encodings = [self.tokenizer(articles) for articles in example_batch['article']]
        #src_encodings = self.tokenizer(example_batch['article'])
        input_ids = []
        attention_mask = []
        # Collate the tokenized sets of articles. For each example, there will
        # be a set of n articles, which looks like
        # {'input_ids': [article1 ids, ... , articlen ids]}. This needs to be collated
        # so the input_ids corresponds to the ids for all articles in a batch
        for i in src_encodings:
           input_ids.append(i['input_ids'])
           attention_mask.append(i['attention_mask'])

        return {
            "source_ids": input_ids,
            "source_mask": attention_mask,
            # "source_ids": src_encodings['input_ids'],
            # "source_mask": src_encodings['attention_mask'],
            "target_ids": tgt_encodings['input_ids'],
            "target_mask": tgt_encodings['attention_mask']}


class Bioasq(Dataset):
    """Class for Bioasq pytorch dataset"""

    def __init__(self, tokenizer, max_seq_len, max_docs, eos_token, path, doc_mixing=False, split=None):
        file_name = "bioasq/bioasq_{}_collection.json".format(split)
        with open("{0}/{1}".format(path, file_name), "r", encoding="utf-8") as f:
            data = json.load(f)
        self.prompt = "<TASK> BIOASQ <QUESTION> "
        self.source = []
        self.queries = []
        self.summaries = []
        queries = []
        source = []
        summaries = []
        # If selecting from all articles first will need a list of all the articles
        if doc_mixing:
            all_articles = list(set([snippet['article'] for example in data for snippet in data[example]['snippets']]))
            logging.info("Number of all articles: %s", len(all_articles))
        # Parse out multi document data
        for example in data:
            articles = []
            pmids = []
            for snippet in data[example]['snippets']:
                if snippet['pmid'] not in pmids:
                    pmids.append(snippet['pmid'])
                    articles.append(snippet['article'])
            if doc_mixing:
                articles = sample_docs(articles, max_docs, all_articles=all_articles)
            else:
                articles = pad_docs(articles, max_docs)
            question = data[example]['question']
            summary = data[example]['ideal_answer']
            self.source.append(tokenizer(articles, return_tensors="pt"))
            # Add eos symbol (usually <\s>) to the end of the target. This will be shifted to the beginning
            # in the model
            self.summaries.append(tokenizer([summary], return_tensors="pt"))
            #self.summaries.append(tokenizer([summary + " " + eos_token], return_tensors="pt"))
            self.queries.append(tokenizer([self.prompt + question], return_tensors="pt"))

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        source_ids = self.source[index]["input_ids"]
        # Torch multihead attention expects mask to be bool and will convert it if not provided
        source_mask = self.source[index]["attention_mask"].to(torch.bool)
        target_ids = self.summaries[index]["input_ids"].squeeze()
        target_mask = self.summaries[index]["attention_mask"].squeeze().to(torch.bool)
        query_ids = self.queries[index]["input_ids"].squeeze()
        query_mask = self.queries[index]["attention_mask"].squeeze().to(torch.bool)

        return {
            "query_ids": query_ids,
            "source_ids": source_ids,
            "query_mask": query_mask,
            "source_mask": source_mask,
            "target_ids": target_ids,
            "target_mask": target_mask}

        return batch


class MedlineplusReviews(Dataset):
    """Class for Medlineplus multi document review dataset"""

    def __init__(self, tokenizer, max_seq_len, max_docs, eos_token, path, doc_mixing=False, split=None):

        with open("{0}/medlineplus_reviews/medlineplus_{1}_review_collection.json".format(split), "r", encoding="utf-8") as f:
            data = json.load(f)
        self.prompt = "<TASK> MEDLINEPLUS <QUESTION> "
        self.source = []
        self.queries = []
        self.summaries = []
        if doc_mixing:
            all_articles = [data[url]['reviews'][pmid] for url in data for pmid in data[url]['reviews']]
            logging.info("Number of all articles: %s", len(all_articles))
        for i, url in enumerate(data):
            articles = []
            for pmid in data[url]['reviews']:
                articles.append(data[url]['reviews'][pmid])
            if doc_mixing:
                articles = sample_docs(articles, max_docs, all_articles=all_articles)
            else:
                articles = pad_docs(articles, max_docs)
            summary = data[url]['summary']
            self.source.append(tokenizer(articles, return_tensors="pt"))
            self.summaries.append(tokenizer([summary], return_tensors="pt"))
            #self.summaries.append(tokenizer([summary + " " + eos_token], return_tensors="pt"))

    def __len__(self):
        return len(self.summaries)

    def __getitem__(self, index):
        source_ids = self.source[index]["input_ids"]
        # Torch multihead attention expects mask to be bool and will convert it if not provided
        source_mask = self.source[index]["attention_mask"].to(torch.bool)
        target_ids = self.summaries[index]["input_ids"].squeeze()
        target_mask = self.summaries[index]["attention_mask"].squeeze().to(torch.bool)

        return {
            "source_ids": source_ids,
            "source_mask": source_mask,
            "target_ids": target_ids,
            "target_mask": target_mask}


class EBM(Dataset):

    def __init__(self, tokenizer, max_seq_len, max_docs, eos_token, path, doc_mixing=False, split=None):
        """
        Parse and yield ebm_collection.json for multi-document summarization
        """
        file_name = "ebm/ebm_{}_collection.json".format(split)
        with open("{0}/{1}".format(path, file_name), "r", encoding="utf-8") as f:
            data = json.load(f)
            example_cnt = 0
        self.prompt = "<TASK> EBM <QUESTION> "
        self.source = []
        self.queries = []
        self.summaries = []
        if doc_mixing:
            all_articles = [justification[1][pmid] for ex in data for ans in data[ex]['answers'] for justification in ans['justifications'] for pmid in justification[1]]
            logging.info("Number of all articles: %s", len(all_articles))
        for example in data:
            question = data[example]['question']
            # Multiple answers, each answer with multiple justifications
            # Here the answer will be the summary, and the references of the multiple
            # justifications will be the source text. So one question will have
            # multiple answers in the dataset.
            # All questions will have at least one answer
            for answer in data[example]['answers']:
                answer_text = answer['answer_text']
                # Some answers will have no justifications
                if answer['justifications'] == []:
                    continue
                # For this task iterate through justifications, and take the abstract
                # the justication text was taken from, not the justification
                # text itself. Use the answer text as the summary
                articles = []
                for justification in answer['justifications']:
                    # Multiple references for each justification
                    for pmid in justification[1]:
                        articles.append(justification[1][pmid])
                # Some answers/justifications will have no reference texts
                if articles == []:
                    continue
                if doc_mixing:
                    articles = sample_docs(articles, max_docs, all_articles=all_articles)
                else:
                    articles = pad_docs(articles, max_docs)

                self.source.append(tokenizer(articles, return_tensors="pt"))
                # Add eos symbol (usually <\s>) to the end of the target. This will be shifted to the beginning
                # in the model
                self.summaries.append(tokenizer([answer_text], return_tensors="pt"))
                #self.summaries.append(tokenizer([answer_text + " " + eos_token], return_tensors="pt"))
                self.queries.append(tokenizer([self.prompt + question], return_tensors="pt"))

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        source_ids = self.source[index]["input_ids"]
        # Torch multihead attention expects mask to be bool and will convert it if not provided
        source_mask = self.source[index]["attention_mask"].to(torch.bool)
        target_ids = self.summaries[index]["input_ids"].squeeze()
        target_mask = self.summaries[index]["attention_mask"].squeeze().to(torch.bool)
        query_ids = self.queries[index]["input_ids"].squeeze()
        query_mask = self.queries[index]["attention_mask"].squeeze().to(torch.bool)

        return {
            "query_ids": query_ids,
            "source_ids": source_ids,
            "query_mask": query_mask,
            "source_mask": source_mask,
            "target_ids": target_ids,
            "target_mask": target_mask}


class MediqaAns(Dataset):
    """Class for MEDIQA AnS dataset"""

    def __init__(self, tokenizer, max_seq_len, max_docs, eos_token, path, doc_mixing=False, split=None):
        file_name = "chiqa/section2answer_multi_abstractive_summ.json"
        with open("{0}/{1}".format(path, file_name), "r", encoding="utf-8") as f:
            data = json.load(f)
            self.prompt = "<TASK> MEDIQA-AnS <QUESTION> "
            self.source = []
            self.queries = []
            self.summaries = []
            if doc_mixing:
                all_articles = [data[ex]['articles'][i][0] for ex in data for i in data[ex]['articles']]
                logging.info("Number of all articles: %s", len(all_articles))
            for example in data:
                question = data[example]['question']
                articles = []
                for answer_id in data[example]['articles']:
                    articles.append(data[example]['articles'][answer_id][0]),
                if doc_mixing:
                    articles = sample_docs(articles, max_docs, all_articles=all_articles)
                else:
                    articles = pad_docs(articles, max_docs)
                self.source.append(tokenizer(articles, return_tensors="pt"))
                # Add eos symbol (usually <\s>) to the end of the target. This will be shifted to the beginning
                # in the model
                #tokenized_summs = tokenizer([data[example]['summary']], return_tensors="pt")
                #tokenized_summs['input_ids'][0, -1] = eos_token_id
                self.summaries.append(tokenizer([data[example]['summary']], return_tensors="pt"))
                self.queries.append(tokenizer([self.prompt + question], return_tensors="pt"))

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        # Before squeeze size is (1, max_seq_len), because tokenizer returns tensors.
        source_ids = self.source[index]["input_ids"].squeeze()
        # Torch multihead attention expects mask to be bool and will convert it if not provided
        source_mask = self.source[index]["attention_mask"].to(torch.bool)
        target_ids = self.summaries[index]["input_ids"].squeeze()
        target_mask = self.summaries[index]["attention_mask"].squeeze().to(torch.bool)
        query_ids = self.queries[index]["input_ids"].squeeze()
        query_mask = self.queries[index]["attention_mask"].squeeze().to(torch.bool)

        return {
            "query_ids": query_ids,
            "source_ids": source_ids,
            "query_mask": query_mask,
            "source_mask": source_mask,
            "target_ids": target_ids,
            "target_mask": target_mask}
