import math
import logging
import GPUtil

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

from transformers import AlbertConfig, AlbertModel
from transformers import AlbertModel

from custom_layers import MultiHeadPooling, MMR

class HierarchicalMDS(nn.Module):
    """
    The hierchical transformer model for multi-document classification.

    Example:
    from model import HierarchicalMDS

    hier_mds = HierarchicalMDS(args)
    hier_mds(input)
    ... and more example stuff
    """

    def __init__(
            self,
            args):
        super(HierarchicalMDS, self).__init__()
        self.max_docs = args.max_docs
        self.max_seq_len = args.max_seq_len
        self.batch_size = args.batch_size
        self.vocab_size = args.vocab_size

        # Model layers
        self.local_enc_layers = args.local_enc_layers
        self.global_enc_layers = args.global_enc_layers
        self.decoder_layers = args.dec_layers
        self.dropout = args.dropout
        self.enc_hidden_dim = args.enc_hidden_dim
        self.dec_hidden_dim = args.dec_hidden_dim
        self.n_att_heads = args.n_att_heads
        self.decoder_type = args.decoder_type
        self.decoder = self.get_decoder()
        if args.local_bert_enc:
            local_encoder = self.get_local_bert_encoder()
        else:
            local_encoder = self.get_local_encoder()
        # Add encoders in ModuleList
        self.pooling = args.multi_head_pooling
        # Option to use CLS token as document representation
        self.use_cls_token = False
        self.encoder = nn.ModuleList([local_encoder, self.get_global_encoder()])
        self.encoder_layers = ["local", "global"]
        logging.info("Encoder objects: {}".format(self.encoder))
        self.mmr = args.mmr
        self.mask = args.mask
        self.k = args.top_k


        # Shared embedding between encoder, decoder, and logits
        # Three-way weight tying: https://arxiv.org/abs/1706.03762
        if args.init_bert_weights:
            logging.info("Initializing embeddings with BERT")
            self.shared_embedding = self.get_bert_weights()
        else:
            logging.info("Initializing random embeddings")
            self.shared_embedding = nn.Embedding(args.vocab_size, args.enc_hidden_dim)
            self.init_weights(self.shared_embedding)

        # Initiate last layer for generating logits
        self.linear_layer = torch.nn.Linear(self.dec_hidden_dim, self.vocab_size)
        # Share shared decoder/encoder weights with last layer
        self.linear_layer.weight = self.shared_embedding.weight

    def set_device(self, device):
        """Set GPU or CPU for the class, and initialize any graphs that require it"""
        self.device = device
        self.pos_encoder = PositionalEncoding(self.enc_hidden_dim, self.dropout, max_len=self.max_seq_len, device=device)

    def get_local_encoder(self):
        """
        The Local transformer encoder layers, using native pytorch.
        See global encoder for custom encoder layer.
        """
        enc = TransformerEncoderLayer(d_model=self.enc_hidden_dim, nhead=self.n_att_heads)
        return TransformerEncoder(enc, num_layers=self.local_enc_layers)

    def get_global_encoder(self):
        """
        The global transformer encoder layers.
        This will be a custom layer for aggregating the local transformer layer output.
        """
        #enc = TransformerEncoderLayer(d_model=self.enc_hidden_dim, nhead=self.n_att_heads)
        #Using custom layer
        enc = TransformerGlobalEncoderLayer(
            d_model=self.enc_hidden_dim, max_seq_len=self.max_seq_len,
            max_docs=self.max_docs, batch_size=self.batch_size,
            nhead=self.n_att_heads, head_pooling=self.pooling)
        return TransformerEncoder(enc, num_layers=self.global_enc_layers)

    def get_decoder(self):
        """The decoder: RNN or Transformer"""
        if self.decoder_type == "RNN":
            dec = nnGRU(hidden_size=self.dec_hidden_dim)
        elif self.decoder_type == "transformer":
            dec = TransformerDecoderLayer(d_model=self.dec_hidden_dim, nhead=self.n_att_heads)
            dec = TransformerDecoder(dec, num_layers=self.decoder_layers)
        else:
            raise NotImplementedError("No decoder implementation of {}".format(self.decoder_type))

        return dec

    def init_weights(self, layer):
        """Weight initialization function for each layer"""
        initrange = 0.1
        nn.init.uniform_(layer.weight, -initrange, initrange)

    def get_bert_weights(self):
        """
        Get BERT (or BERT variant weights) for initialization
        Requires use of HuggingFace
        """
        logging.warning("Using Albert. Pad inputs on the right: https://huggingface.co/transformers/model_doc/albert.html")
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        model = AlbertModel.from_pretrained('albert-base-v2')
        embeddings =  model.get_input_embeddings()
        return embeddings

    def get_local_bert_encoder(self):
        """
        Initialize local BERT encoder from HuggingFace
        In this architecture, each document will be fed independently
        into the ALBERT encoder
        """
        albert_base_configuration = AlbertConfig(
            hidden_size=self.enc_hidden_dim,
            num_attention_heads=self.n_att_heads,
            intermediate_size=3072,
        )
        local_encoder = AlbertModel(albert_base_configuration)

        # Implementation not tested
        raise NotImplementedError()
        #return local_encoder

    def get_gpt_decoder():
        """Initialize GPT2 decoder from HuggingFace"""
        raise NotImplementedError

    def forward(self, doc_input, query_input, target_input):
        """
        Method for running forward pass through encoder and decoder
        Input dimensions: Batch, number of docs per example, number of tokens in each doc
        """
        # Shape for pass through document level encoders
        # In general, pytorch convention is to have inputs of (seq length, batch size, n features)
        # Apply shared embedding layer to inputs and targets
        doc_emb = self.shared_embedding(doc_input)
        logging.warning("Implement use of query emb!")
        query_emb = self.shared_embedding(doc_input)
        tgt_emb = self.shared_embedding(target_input)
        doc_emb = doc_emb.view(self.max_seq_len, self.batch_size * self.max_docs, self.enc_hidden_dim)
        tgt_emb = tgt_emb.view(self.max_seq_len, self.batch_size, self.enc_hidden_dim)
        logging.info("seq len: {0}, batch: {1}, docs: {2}, h: {3}".format(self.max_seq_len, self.batch_size, self.max_docs, self.enc_hidden_dim))
        logging.info(doc_emb.size())
        logging.info(tgt_emb.size())

        # How to mask? And do I need to mask for "fine-tuning"?
        if self.mask:
            device = self.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        # Attention is all you need:
        # In our model, we share the same weight matrix between the two embedding
        # layers and the pre-softmax linear transformation...
        # ...In the embedding layers, we multiply those weights by sqrt(model)"
        doc_emb = doc_emb * math.sqrt(self.enc_hidden_dim)
        tgt_emb = tgt_emb * math.sqrt(self.dec_hidden_dim)

        logging.debug("dob emb size: %s", doc_emb.size())
        # Using query and document embeddings, comput Maximal Marginal Relevance
        if self.mmr:
            self.compute_mmr = MMR()
            mmr_emb = self.compute_mmr(doc_emb, query_emb)

        doc_emb = self.pos_encoder(doc_emb)
        logging.debug("dob emb size after pos: %s", doc_emb.size())
        #query_emb = self.pos_encoder(query_emb)
        # Pass through each layer of encoder. This could be done with
        # the Sequential module in pytorch, but leaving the encoders in ModuleList
        # provides for flexibility and is more explicit as to what is going on.

        logging.info("Contiguous: %s", doc_emb.is_contiguous())
        logging.info("passing input through enc")
        logging.info(doc_emb.device)
        # Iterate through the local and global encoders
        for layer_type, layer in zip(self.encoder_layers, self.encoder):
            GPUtil.showUtilization()
            logging.debug("layer type: %s", layer_type)
            if layer_type == 'local':
                local_doc_emb = layer(doc_emb)
            elif layer_type == 'global':
                # Once input passed through local layers
                # reshape embeddings back to original shape:
                # (input length * max docs, batch siz, hidden dim)
                #local_doc_emb = local_doc_emb.view(self.max_seq_len * self.max_docs, self.batch_size, self.enc_hidden_dim)
                # TODO:
                # pool input max_seq_len * max_docs to just max_docs
                logging.debug("local doc emb size after reshaping %s:", local_doc_emb.size())
                local_doc_emb, global_doc_emb = layer(local_doc_emb)
        logging.debug("global doc emb size after encoder %s:", global_doc_emb.size())
        logging.debug("local doc emb size after encoder %s:", local_doc_emb.size())
        # TODO:
        # Use softmax to select the topk documents, and provide the local encoder embeddings
        doc_likelikehoods = torch.nn.functional.log_softmax(global_doc_emb, dim=-1)
        k_docs = torch.topk(doc_likelikehoods, k=self.k, dim=0)
        k_doc_emb = k_docs.values
        k_doc_indicies = k_docs.indices
        logging.info("Topk!")
        logging.info(k_doc_emb)
        logging.info(k_doc_emb.size())
        # Oh man what am I doing? Select docs from local_doc_emb, concatenate those representations
        # with the global topk, and then send them on their way to the decoder
        #logging.debug("doc emb size after after all encoding and reshaping: %s", doc_emb.size())
        #logging.debug("target emb size: %s", tgt_emb.size())
        output = self.decoder(tgt_emb, k_doc_emb, tgt_mask=None, memory_mask=None)
        #logging.debug("output emb size after decoding: %s", output.size())
        output = self.linear_layer(output)
        #logging.debug("output after linear layer: %s", output.size())
        # Final reshape for cross entropy (Soft max and NLLoss)
        output = output.view(self.batch_size, self.vocab_size, self.max_seq_len, )

        return output


class TransformerGlobalEncoderLayer(TransformerEncoderLayer):
    """
    TransformerGlobalEncoderLayer is a custom layer for this work.
    Boiler plate transformer code from
    https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer

    The layer is composed of self-attn and feedforward network, using pytorch's implementation.
    It is intended to be used with a hierarchical model, and should receive input
    from multiple local encoders.

    It diverges from the standard pytorch transformer in that it first uses multi-head pooling
    to generate document level representations.  It then uses those representations to compute MMR
    and multi head attention between documents.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        pooling: True or False, for multi head pooling


    Example:
        >>> src = torch.rand(10, 32, 512)
        >>> encoder_layer = nn.TransformerGlobalEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> out = transformer_encoder(src)
    """

    def __init__(
        self, d_model, max_seq_len, max_docs,
        batch_size, nhead,
        head_pooling=False, dim_feedforward=2048,
        dropout=0.1, activation="relu"):

        super(TransformerGlobalEncoderLayer, self).__init__(
            d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"
            )
        self.max_seq_len = max_seq_len
        self.max_docs = max_docs
        self.batch_size = batch_size
        self.head_pooling = head_pooling
        # TODO: Add option for using cls token as doc representation
        if head_pooling:
            self.multi_head_pooling = MultiHeadPooling(
                max_seq_len, max_docs, batch_size, d_model, nhead, dropout=dropout
                )
        else:
            self.max_pooling = nn.MaxPool1d(d_model)

    def forward(self, input, mmr=None, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        """Pass the input through the global encoder layer and return document representation

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        logging.info("Using global transformer")
        logging.info("input_size %s", input.size())
        if self.head_pooling:
            x_pooled = self.multi_head_pooling(input, input)

        else:
            x_pooled = self.max_pooling(input).view(self.max_docs, self.batch_size, self.d_model)
        logging.info("Pooling size %s", x_pooled.size())

        # Once the token representations have been pooled, we can compute attention between documents
        doc_emb = self.self_attn(x_pooled, x_pooled, x_pooled, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        logging.info("self attn %s", doc_emb.size())
        if mmr:
            logging.info("Placeholder for mmr computation")
            mmr_weights = self.mmr(x_att, query_emb)
            doc_emb = mmr_weights * mmr_weights

        # TODO: Compute this correctly in light of multihead pooling
        logging.info("doc attn size %s", doc_emb.size())
        doc_emb = doc_emb.view()
        word_emb = input + self.dropout1(doc_emb)
        word_emb_1 = self.norm1(word_emb)
        logging.info("norm1 src %s", word_emb.size())
        word_emb_2 = self.linear2(self.dropout(self.activation(self.linear1(word_emb))))
        logging.info("src2 linr attn %s", word_emb_2.size())
        word_emb = word_emb_1 + self.dropout2(word_emb_2)
        word_emb = self.norm2(word_emb)
        logging.info("src norm attn %s", word_emb.size())
        #assert src.size() == torch.Size([self.max_docs, self.batch_size, self.enc])
        return word_emb, doc_emb

class PositionalEncoding(nn.Module):
    """
    Class for computing position embeddings from
    https://github.com/pytorch/tutorials/blob/master/beginner_source/transformer_tutorial.py
    No fancy additions made, EXCEPT adding device to init and where tensor is created,
    in order to run on GPU.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000, device=None):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
