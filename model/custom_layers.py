import logging
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

        logging.info("Contiguous: %s", value.is_contiguous())
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


class MMR():
    """
    Class for computing Maximal Marginal Relevance.
    Code from https://github.com/Alex-Fabbri/Multi-News/blob/master/code/Hi_MAP/onmt/encoders/decoder.py
    """
    def _init_mmr(self, hidden_dim):
        # for sentence and summary distance.. This is defined as sim 1
        self.mmr_W = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def _run_mmr(self, doc_emb, query_emb, src_sents, input_step):
        '''
        # : size (sent_len=9,batch=2,dim=512)
        # sent_decoder: size (sent_len=1,batch=2,dim=512)
        # src_sents: size (batch=2,sent_len=9)
        function to calculate mmr
        :param sent_encoder:
        :param sent_decoder:
        :param src_sents:
        :return:
        '''
        pdist = nn.PairwiseDistance(p=2)
        sent_decoder=sent_decoder.permute(1,0,2) # (2,1,512)

        scores =[]
        # define sent matrix and current vector distance as the Euclidean distance
        for d in doc_emb: # iterate over each batch sample
            # distance: https://pytorch.org/docs/stable/_modules/torch/nn/modules/distance.html
            sim1 = 1 - torch.mean(pdist(sent_encoder.permute(1, 0, 2), sent.unsqueeze(1)), 1).unsqueeze(1) # this is a similarity function
            # sim1 shape: (batch_size,1)
            sim2=torch.bmm(self.mmr_W(sent_decoder),sent.unsqueeze(2)).squeeze(2) # (2,1) -> this is sim1 on my equation
            # scores.append(sim1-sim2)
            scores.append(sim2 - sim1)

        sent_ranking_att = torch.t(torch.cat(scores,1)) #(sent_len=9,batch_size)
        sent_ranking_att = torch.softmax(sent_ranking_att, dim=0).permute(1,0)  #(sent_len=9,batch_size)
        # scores is a list of score (sent_len=9, tensor shape (batch_size, 1))
        mmr_among_words = [] # should be (batch=2,input_step=200)
        for batch_id in range(sent_ranking_att.size()[0]):
            # iterate each batch, create zero weight on the input steps
            # mmr= torch.zeros([input_step], dtype=torch.float32).cuda()

            tmp = []
            for id,position in enumerate(src_sents[batch_id]):

                for x in range(position):
                    tmp.append(sent_ranking_att[batch_id][id])

            mmr = torch.stack(tmp) # make to 1-d


            if len(mmr) < input_step: # pad with 0
                tmp = torch.zeros(input_step - len(mmr)).float().cuda()
                # for x in range(input_step-len(mmr)):
                mmr = torch.cat((mmr, tmp), 0)
            else:
                mmr = mmr[:input_step]

            mmr_among_words.append(mmr.unsqueeze(0))

        mmr_among_words = torch.cat(mmr_among_words,0)

        # shape: (batch=2, input_step=200)

        return mmr_among_words
