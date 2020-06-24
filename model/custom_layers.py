import logging
import numpy as np
import torch
from torch import nn

class MultiHeadPooling(nn.Module):
    """
    Multi head pooling layer
    This is used to aggregate token representations (input len * num_docs, batch size, hidden_dim)
    that are passed from the local encoder to the global encoder layer
    See https://github.com/nlpyang/hiersumm/blob/master/src/abstractive/attn.py for the implementation
    this code uses as reference
    """

    def __init__(self, max_seq_len, max_docs, batch_size, hidden_dim, n_heads, dropout):
        assert hidden_dim % n_heads == 0, "Hidden dimension is not divisible by number of heads"
        self.dim_per_head = int(hidden_dim / n_heads)
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.max_docs = max_docs
        self.batch_size = batch_size
        super(MultiHeadPooling, self).__init__()
        self.n_heads = n_heads
        self.linear_keys = nn.Linear(self.hidden_dim, self.n_heads)
        self.linear_values = nn.Linear(self.hidden_dim, self.n_heads * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, key, value, mask=None):
        """
        Forward pass for Multi Head Pooling used in the global encoder layer
        Expects inputs of shape (max_seq_len, batch_size * max_docs, hidden_dim)
        """
        logging.info("Computing multi-head pooling")
        # For each token representation in the input:
        # Run input through two linear layers, where the layer for the attn scores will be of hidden_dim x n_heads
        # This means that for the atttention scores for one head for one document
        # the output will be a 1 x max seq length score, one score for each token representation
        # in the document
        scores = self.linear_keys(key)
        logging.info(scores.size())
        # and the layer for the values will be of hidden dim x dim of heads
        values = self.linear_values(value)
        logging.info(value.size())
        # Reshape scores and values so that
        # shape(scores) == (seq len, score for each head for each doc for each sample)
        scores = scores.view(
            self.max_seq_len, self.max_docs * self.batch_size * self.n_heads
            ).unsqueeze(-1)
        logging.info("scores %s", scores.size())
        # Reshape values so that
        # shape(values) == (seq len, head for each doc for each sample, vector of size head dim)
        values = values.view(
            self.max_seq_len, self.max_docs * self.batch_size * self.n_heads, self.dim_per_head
            )
        logging.info("values %s", values.size())
        # Take softmax of scores to get a distribution for each head and apply dropout
        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        # Weight the embedding for each head with element-wise multiplication
        scores = scores * values
        logging.info("scores after weighting %s", scores.size())

        # TODO: Mask stuff
        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        # Sum the weights for each head in each doc in an example,
        # resulting in a context vector of size (max docs, n heads, dim of heads)
        context = torch.sum(scores, dim=0)
        logging.info("doc representation %s", context.size())
        # Then stack the heads so that the final tensor will be of size (max docs, batch size, hidden dimension)
        context = context.view(self.batch_size, self.max_docs, self.n_heads * self.dim_per_head).transpose(0,1)
        #attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        logging.info("doc representation %s", context.size())
        output = self.final_linear(context)
        logging.info("doc representation %s", output.size())
        assert output.size() == torch.Size([self.max_docs, self.batch_size, self.hidden_dim])
        return output


class MMR(nn.Module):
    """
    Class for computing Maximal Marginal Relevance.

    MMR for doc i = λ * Sim1(hidden state i, query state)
                    − (1 − λ) * max(Sim2(hidden state i, hidden state j)), j != i
    and the distance metric is PairwiseDistance.

    Code from https://github.com/Alex-Fabbri/Multi-News/blob/master/code/Hi_MAP/onmt/encoders/decoder.py
    """
    def __init__(self, hidden_dim, max_seq_len, lambda_m=.5):
        """Initialize lambda and weights for MMR"""
        super(MMR, self).__init__()
        # Labmda for mmr
        self.lambda_m = np.float32(lambda_m)
        # Weights for sentence and query distance for similarity 1
        self.query_W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query_attn = nn.Linear(hidden_dim, max_seq_len, bias=False)
        self.doc_W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Weights for attention between each doc represenation
        self.self_doc_W = nn.Linear(hidden_dim, hidden_dim, bias=False).cuda()
        self.attn_dist = nn.Softmax(dim=-1)
        self.measure = nn.CosineSimilarity(dim=-1)

    def forward(self, doc_emb, query):
        """
        Function for calculating mmr between query and documents
        Representations for each document in an example: shape == (max docs, batch size, hidden_dim)
        Representation for query: shape == (max seq len, batch size, hidden_dim)

        Arguments:
            doc_emb: Doc representations generated by multi head Pooling
            query_emb: Query embeddings from initial embedding layer
        """
        # The first similiary, between the query and each document
        # torch.bmm requries batch first
        logging.info("mmr!")
        logging.info("query size before linear: %s", query.size())
        logging.info("doc emb %s", doc_emb.size())
        doc_emb = doc_emb.transpose(0, 1)
        query = self.query_W(query.transpose(0, 1))
        # Apply linear for attention weights
        query_attn = self.query_attn(query)
        query_attn = self.attn_dist(query_attn)
        logging.info(query_attn)
        logging.info("query attn after linear: %s", query_attn.size())
        doc_values = self.doc_W(doc_emb)
        logging.info("doc emb %s", doc_emb.size())
        # First similiary measure will be of size (batch size, max docs, 1 score per doc)
        query_doc = torch.bmm(query, doc_values.transpose(1, 2))
        logging.info("query doc linear %s", query_doc.size())
        query_doc_attn = torch.bmm(query_attn, query_doc)
        logging.info("weighted query doc linear %s", query_doc.size())
        # Weighted summation
        # To do: Mybe just use this to weight document representations?, instead of mmr?
        sim1 = query_doc_attn.sum(dim=1)
        # shape == (batch_size x number of docs, where each doc has a mmr score)
        logging.info("first sim: %s", sim1.size())
        # Enumerate through docs
        doc_weights = self.self_doc_W(doc_emb)
        logging.info("self doc weights %s", doc_weights.size())
        sim2 = torch.bmm(doc_weights, doc_emb.transpose(1, 2))
        logging.info("sim 2 %s", sim2.size())
        logging.info("sim 2 %s", sim2)
        # TODO: Check this math
        mmr_scores = (self.lambda_m * sim1.unsqueeze(-1)) - ((1 - self.lambda_m) * sim2)
        logging.info("mmr_scores %s", mmr_scores.size())
        logging.info("mmr_scores %s", mmr_scores)
        mmr_scores = torch.max(self.attn_dist(mmr_scores), dim=-1).values
        logging.info("mmr_scores %s", mmr_scores.size())
        # Weight the doc representations by the mmr score for each document
        logging.info(doc_emb)
        doc_emb = doc_emb * mmr_scores.unsqueeze(-1)
        logging.info(doc_emb)
        logging.info("doc_emb %s", doc_emb.size())
        # Back to (max docs, batch size)
        doc_emb = doc_emb.transpose(0, 1)

        return doc_emb
