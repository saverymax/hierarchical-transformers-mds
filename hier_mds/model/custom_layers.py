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
        self.softmax = nn.Softmax(dim=0)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, key, value, src_key_padding_mask, doc_key_padding_mask=None):
        """
        Forward pass for Multi Head Pooling used in the global encoder layer
        Expects inputs of shape (max_seq_len, batch_size * max_docs, hidden_dim)
        """
        logging.info("Computing multi-head pooling")
        # Temporary fix to deal with nan issue caused by fully padded docs.
        # Could also always use doc mixing during any training
        src_key_padding_mask = None
        # For each token representation in the input:
        # Run input through two linear layers, where the layer for the attn scores will be of hidden_dim x n_heads
        # This means that for the atttention scores for one head for one document,
        # the output will be of shape max_seq_length x  one score for each token representation in the doc
        # i.e., shape == (max_seq_len, batch x max_docs, num_heads)
        assert self.dim_per_head * self.n_heads == self.hidden_dim, "Full hidden dimension (hidden_dim) must be divisible by n_heads"
        logging.info(key)
        scores = self.linear_keys(key)
        logging.info("attn scores %s", scores.size())
        # and the layer for the values will shape == (max_seq_len, batch x max_docs, hidden_dim)
        # which can be shaped into the number of heads
        values = self.linear_values(value)
        logging.info("values %s", value.size())

        # Mask the padded indices that shouldn't get a weight
        logging.info("scores before mask %s", scores)
        if isinstance(src_key_padding_mask, torch.Tensor):
            # Mask comes in with shape (batch_size x max_docs, max seq len)
            logging.info("mask for attn scores %s", src_key_padding_mask.size())
            src_key_padding_mask = src_key_padding_mask.transpose(0, 1).unsqueeze(-1)
            logging.info("mask for attn scores reshape %s", src_key_padding_mask.size())
            # Masked fill can Broadcast to last dim
            scores.masked_fill_(src_key_padding_mask, float('-inf'))

        logging.info("scores after masking %s", scores.size())
        logging.info("scores after masking %s", scores)
        # Take softmax of scores over tokens for a doc, to get a distribution for each head and apply dropout.
        # The masked values will be zeroed out
        attn = self.softmax(scores)
        logging.info("scores after softmax %s", attn.size())
        logging.info("scores after softmax %s", attn)
        # Reshape scores and values so that
        # shape(scores) == (seq len, score for each head for each doc for each sample)
        attn = attn.view(
            self.max_seq_len, self.max_docs * self.batch_size * self.n_heads
            ).unsqueeze(-1)
        logging.info("scores after reshape %s", attn.size())
        logging.info("scores after reshape %s", attn)
        logging.info("scores %s", attn.size())
        drop_attn = self.dropout(attn)
        # Reshape values so that
        # shape(values) == (seq len, head for each doc for each sample, vector of size head dim)
        values = values.view(
            self.max_seq_len, self.max_docs * self.batch_size * self.n_heads, self.dim_per_head
            )
        logging.info("values after shaping %s", values.size())
        logging.info(values)
        # Weight the embedding for each head with element-wise multiplication
        weighted_values = drop_attn * values
        logging.info("tokens after weightin %s", weighted_values.size())
        logging.info(weighted_values)
        # Sum the weights for each head in each sequence in an example,
        # resulting in context vectors of size (max docs x n heads, dim of heads)
        context = torch.sum(weighted_values, dim=0)
        logging.info("doc representation after sum%s", context.size())
        logging.info(context)
        # The documents will be contiguous,
        # so stack the heads so that the final tensor will be of size (max_docs, batch_size, hidden dimension)
        context = context.view(self.batch_size, self.max_docs, self.n_heads * self.dim_per_head).transpose(0,1)
        logging.info("doc representation reshpae %s", context.size())
        # Linear layer of full dimension
        output = self.final_linear(context)
        logging.info("doc representation after final linear %s", output.size())
        logging.info(output)
        assert output.size() == torch.Size([self.max_docs, self.batch_size, self.hidden_dim])

        return output


class MMR(nn.Module):
    """
    Class for computing Maximal Marginal Relevance.

    MMR for doc i = λ * Sim1(hidden state i, query state)
                    − (1 − λ) * max(Sim2(hidden state i, hidden state j)), j != i
    Reference code from https://github.com/Alex-Fabbri/Multi-News/blob/master/code/Hi_MAP/onmt/encoders/decoder.py
    However, the code here only roughly follows that implementation.
    """
    def __init__(self, hidden_dim, max_seq_len, lambda_m=.5):
        """Initialize lambda and weights for MMR"""
        super(MMR, self).__init__()
        # Labmda for mmr
        self.lambda_m = np.float32(lambda_m)
        # Weights for sentence and query distance for similarity 1
        self.query_W = nn.Linear(hidden_dim, hidden_dim)
        self.query_attn = nn.Linear(hidden_dim, 1)
        self.doc_W = nn.Linear(hidden_dim, hidden_dim)
        self.query_softmax = nn.Softmax(dim=1)
        self.mmr_softmax = nn.Softmax(dim=-1)
        self.measure = nn.CosineSimilarity(dim=-1)

    def forward(self, doc_emb, query, doc_key_padding_mask=None, query_padding_mask=None):
        """
        Function for calculating mmr between query and documents
        Representations for the documents: shape == (max docs, batch size, hidden_dim)
        Representations for the queries: shape == (max seq len, batch size, hidden_dim)

        Arguments:
            doc_emb: Doc representations generated by multi head Pooling
            query_emb: Query embeddings from initial embedding layer
        """
        # Temporary fix to deal with nan issue caused by fully padded docs.
        # Another possible fix would be to use single query representation, not token rep.
        # Or to always use doc mixing during any training
        doc_key_padding_mask = None
        # The first similiary, between the query and each document
        logging.info("mmr!")
        logging.info("query size before linear: %s", query.size())
        logging.info("doc emb %s", doc_emb.size())
        # torch.bmm requries batch first
        doc_emb = doc_emb.transpose(0, 1)
        # Apply weights with full dimension
        query = self.query_W(query.transpose(0, 1))
        logging.info("query after linear %s ", query.size())
        # Apply linear for attention weights to get single scores for each token
        query_attn = self.query_attn(query)
        # Mask the query attn scores where the query is padded,
        # so that any document representation multiplied with that token will result in 0
        if query_padding_mask is not None:
            # Mask comes in with batch_size x max_docs as 0th dim
            logging.info("query_mask %s", query_padding_mask.size())
            logging.info("query_mask %s", query_padding_mask)
            mask = query_padding_mask.unsqueeze(-1)
            # Masked fill can Broadcast to last dim
            # Fills tensor with -inf where true
            query_attn.masked_fill_(mask, float('-inf'))
        # Turn output of linear into distribution
        query_attn = self.query_softmax(query_attn)
        #logging.info(query_attn)
        logging.info("query attn after linear: %s", query_attn.size())
        # Then apply document weights with full dimension
        doc_values = self.doc_W(doc_emb)
        logging.info("doc emb %s", doc_emb.size())
        # For each token representation in query, multiply with document represenation
        query_doc = torch.bmm(query, doc_values.transpose(1, 2))
        logging.info("query doc linear %s", query_doc.size())
        logging.info("query doc linear %s", query_doc)

        query_doc_attn = query_attn * query_doc
        logging.info("weighted query doc linear %s", query_doc_attn.size())
        logging.info(query_doc_attn)
        # Weighted summation of columns, combining the attention scores for a document
        # over all the tokens in the query
        # sim1 shape will be (batch_size, number of docs)
        sim1 = query_doc_attn.sum(dim=1)
        logging.info("first sim: %s", sim1.size())
        logging.info(sim1)
        # Then compute similarity between documents, using the values previously computed for docs
        logging.info("doc weights %s", doc_values.size())
        logging.info("doc weights %s", doc_values)
        sim2 = torch.bmm(doc_emb, doc_values.transpose(1, 2))
        # Here we have the tensor
        #                   [[[a,b,c],
        #                   [x,y,z]],
        #                   ...]]]
        # where the last dimension i.e., [a,b,c], will be a document's score relative to every other doc.
        logging.info("sim 2 %s", sim2.size())
        logging.info("sim 2 %s", sim2)
        # Then take softmax and the max, which will be used as the final score,
        # where the shape will be
        if isinstance(doc_key_padding_mask, torch.Tensor):
            # Mask comes in with batch_size x max_docs as 0th dim
            logging.info("Doc padding mask %s", doc_key_padding_mask.size())
            logging.info("Doc padding mask %s", doc_key_padding_mask)
            mask = doc_key_padding_mask.unsqueeze(-1)
            # Masked fill can Broadcast to last dim
            sim2.masked_fill_(mask, float('-inf'))
            logging.info("sim2 after mask %s", sim2)

        sim2 = self.mmr_softmax(sim2)
        logging.info("sim 2 %s", sim2.size())
        logging.info("sim 2 %s", sim2)
        sim2 = torch.max(sim2, dim=-1).values
        logging.info("sim 2 %s", sim2.size())
        logging.info("sim 2 %s", sim2)
        # Using similarity measures, compute mmr!, where the doc level similiarity for doc_i
        # is subtracted from query-doc similiarity
        mmr_scores = (self.lambda_m * sim1) - ((1 - self.lambda_m) * sim2)
        logging.info("mmr_scores %s", mmr_scores.size())
        logging.info("mmr_scores %s", mmr_scores)
        # Weight the doc representations by the mmr score for each document
        logging.info("doc rep before mmr weights %s", doc_emb.size())
        logging.info(doc_emb)
        doc_emb = doc_emb * mmr_scores.unsqueeze(-1)
        logging.info("doc_emb after mmr weights %s", doc_emb.size())
        logging.info(doc_emb)
        # Back to (max docs, batch size)
        doc_emb = doc_emb.transpose(0, 1)

        return doc_emb
