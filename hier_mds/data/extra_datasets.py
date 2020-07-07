"""
Extra code for indexable pytorch datasets using huggingface cnn_dailymail and eli5 datasets.
"""


class ELI5(Dataset):
    """Indexable ELI5 dataset"""

    def __init__(self, tokenizer, max_seq_len, max_docs, eos_token, path=None, cache_dir=None, doc_mixing=False):
        task = "eli5"
        self.prompt = "<TASK> {} <QUESTION> ".format(task)
        dataset = nlp.load_dataset(task, split="train_eli5[:5%]", cache_dir=cache_dir)
        self.source = []
        self.summaries = []
        self.queries = []
        for ex in dataset:
            self.queries.append(tokenizer([self.prompt + ex['selftext']]))
            articles = [ex['document']]
            # Chose random articles to pad document set with, or just pad with 0
            if doc_mixing:
                sample_docs(articles)
            else:
                articles = pad_docs(articles, max_docs)
            self.source.append(tokenizer(articles))
            # Use the first answer
            self.summaries.append(tokenizer([ex['answers']['text'][0] + " " + eos_token]))

    def __len__(self):
        return len(self.source)

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


class CnnDm(Dataset):
    """Indexable HF NLP CNN/DM dataset"""

    def __init__(self, tokenizer, max_seq_len, max_docs, eos_token, path=None, cache_dir=None, doc_mixing=False):
        logging.info("Loading CNN Dailymail dataset")
        task = "cnn_dailymail"
        dataset = nlp.load_dataset(task, '3.0.0', split="train[:10%]", cache_dir=cache_dir)
        self.tokenizer = tokenizer
        self.doc_mixing = False
        self.source = []
        self.queries = []
        self.summaries = []
        self.prompt = ""
        # HuggingFace datasets allow you to map function to each example; however I want to be able to
        for ex in dataset:
            articles = [ex['article']]
            if doc_mixing:
                articles = sample_docs(articles)
            else:
                articles = pad_docs(articles, max_docs)
            self.source.append(tokenizer(articles))
            # Create blank query
            self.queries.append(tokenizer([""]))
            # Add eos symbol (usually <\s>) to the end of the target. This will be shifted to the beginning
            # in the model
            self.summaries.append(tokenizer([ex['highlights'] + " " + eos_token]))

    def __len__(self):
        return len(self.source)

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
