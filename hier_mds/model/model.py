import math
import logging
import GPUtil

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer, MultiheadAttention

from transformers import PreTrainedModel, AlbertConfig, AlbertModel

from .custom_layers import MultiHeadPooling, MMR

#class HierarchicalTransformer(nn.Module):
class HierarchicalTransformer(PreTrainedModel):
    """
    The hierchical transformer model for multi-document classification.

    Example:
    from model import HierarchicalMDS

    hier_mds = HierarchicalMDS(args)
    hier_mds(input)
    ... and more example stuff
    """

    def __init__(self, config):
        # Model inherits self.config.all-the-params-you-desire from hf PreTrainedModel
        super(HierarchicalTransformer, self).__init__(config)
        logging.info("Hierarchical config: %s", self.config)
        self.decoder = self.get_decoder()
        self.pos_encoder = PositionalEncoding(self.config.enc_hidden_dim, self.config.dropout, max_len=self.config.max_seq_len)
        local_encoder = self.get_local_encoder()
        self.encoder = nn.ModuleList([local_encoder, self.get_global_encoder()])
        self.encoder_layers = ["local", "global"]
        #logging.info("Encoder objects: {}".format(self.encoder))
        # Softmax for selecting documents
        self.doc_softmax = nn.Softmax(dim=-1)

        # Shared embedding between encoder, decoder, and logits
        # Three-way weight tying: https://arxiv.org/abs/1706.03762
        if self.config.init_bert_weights:
            logging.info("Initializing embeddings with BERT")
            self.shared_embedding = self.get_bert_weights()
        else:
            logging.info("Initializing random embeddings")
            self.shared_embedding = nn.Embedding(self.config.vocab_size, self.config.enc_hidden_dim)
            self.init_weights(self.shared_embedding)

        # Initiate last layer for generating logits
        self.final_linear_layer = torch.nn.Linear(self.config.dec_hidden_dim, self.config.vocab_size)
        # Share shared decoder/encoder weights with last layer
        self.final_linear_layer.weight = self.shared_embedding.weight

    def set_device(self, device):
        """Set GPU or CPU for instance"""
        self.compute_device = device
        #self.pos_encoder = PositionalEncoding(self.enc_hidden_dim, self.dropout, max_len=self.max_seq_len, device=device)

    def get_local_encoder(self):
        """
        The Local transformer encoder layers, using native pytorch.
        See global encoder for custom encoder layer.
        """
        enc = TransformerEncoderLayer(d_model=self.config.enc_hidden_dim, nhead=self.config.n_att_heads)
        return TransformerEncoder(enc, num_layers=self.config.local_enc_layers)

    def get_global_encoder(self):
        """
        The global transformer encoder layers.
        This will be a custom layer for aggregating the local transformer layer output.
        """
        # Custom transformer encoder layer
        # This could just inherit the config but I'd like to keep it independent of this code.
        enc = TransformerGlobalEncoderLayer(
            d_model=self.config.enc_hidden_dim, max_seq_len=self.config.max_seq_len,
            max_docs=self.config.max_docs, batch_size=self.config.batch_size,
            nhead=self.config.n_att_heads, mmr=self.config.mmr, query_doc_attn=self.config.query_doc_attn,
            head_pooling=self.config.multi_head_pooling)
        return enc

    def get_decoder(self):
        """The decoder: RNN or Transformer"""
        if self.config.decoder_type == "transformer":
            dec = TransformerDecoderLayer(d_model=self.config.dec_hidden_dim, nhead=self.config.n_att_heads)
            dec = TransformerDecoder(dec, num_layers=self.config.decoder_layers)
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
        model = AutoModelWithLMHead.from_pretrained(self.config.hf_model)
        embeddings =  model.get_input_embeddings()
        return embeddings

    def get_output_embeddings(self):
        """Method required by hugging face to use self.generate()"""
        vocab_size, emb_size = self.shared_embedding.weight.shape
        lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
        lin_layer.weight.data = self.shared_embedding.weight.data
        return lin_layer

    def generate_square_subsequent_mask(self, sz):
        """Pytorch nn.Transformer implementation to generate causal mask"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(self.device)
        return mask

    def forward(
            self, src_input, target_input, query_input=None,
            src_padding_mask=None, tgt_padding_mask=None, query_padding_mask=None):
        """
        Method for running forward pass through encoder and decoder
        Input dimensions: Batch, number of docs per example, number of tokens in each doc
        """
        # Shape for pass through document level encoders
        # In general, pytorch convention is to have inputs of (seq length, batch size, n features)
        # The inputs come in shape (batch size, max docs, input length) so once embedded, there is much
        # ado to shape them for pytorch transformers
        # Masking prep:
        if not isinstance(src_padding_mask, torch.Tensor) or not isinstance(tgt_padding_mask, torch.Tensor):
            raise IOError("Src or tgt mask is not of type torch.Tensor. Please provide masks from hf tokenizer with return_tensors='pt'")
            # Masks should be of shape (batch_size, seq_len)
            # Simple way to mask out documents is to sum booleans along seq len dimension.
            # If greater than zero, there must be at least one token in the doc
            # Will be of shape (batch_size, max_docs)
        else:
            doc_padding_mask = src_padding_mask.sum(-1) > 0
            logging.info("doc_mask size %s" , doc_padding_mask.size())
            logging.info("doc_mask %s" , doc_padding_mask)
            logging.info("TOken mask! %s", src_padding_mask.size())
            if isinstance(query_padding_mask, torch.Tensor):
                logging.info("query mask! %s", query_padding_mask.size())
            logging.info("target mask! %s", tgt_padding_mask.size())

        # Apply shared embedding layer to inputs and targets
        token_emb = self.shared_embedding(src_input)
        logging.debug("dob emb size for input: %s", token_emb.size())
        # Take transpose of target sequences for pytorch transformers, putting seq length first
        tgt_emb = self.shared_embedding(target_input).transpose(0, 1)
        logging.debug("tgt emb size: %s", tgt_emb.size())
        if isinstance(query_padding_mask, torch.Tensor):
            query_emb = self.shared_embedding(query_input).transpose(0, 1)
            logging.debug("query emb size: %s", query_emb.size())
        # Reshape and flatten batch size x max docs for local transformers, where each document is independent
        # For example, the first element of the 0th dimension will contain the first token represenation from each of the
        # documents in the same example, followed by the first token from the documents in the next example
        token_emb = token_emb.permute(2, 0, 1, 3).contiguous().view(
            self.config.max_seq_len,
            self.config.batch_size * self.config.max_docs,
            self.config.enc_hidden_dim)
        # Great, pytorch expects mask tensor of shape (batch_size, max_docs, max_seq_len)
        src_padding_mask = src_padding_mask.view(
            self.config.batch_size * self.config.max_docs,
            self.config.max_seq_len
            )
        logging.debug("dob emb size for input after initial reshape: %s", token_emb.size())
        logging.debug("token mask emb size for input after initial reshape: %s", src_padding_mask.size())

        # Attention is all you need:
        # In our model, we share the same weight matrix between the two embedding
        # layers and the pre-softmax linear transformation...
        # ...In the embedding layers, we multiply those weights by sqrt(model)"
        token_emb = token_emb * math.sqrt(self.config.enc_hidden_dim)
        tgt_emb = tgt_emb * math.sqrt(self.config.dec_hidden_dim)
        logging.info("initial doc emb %s", token_emb)
        logging.info("src mask %s", src_padding_mask)
        # Add positional embeddings
        # Will return same shape as input
        token_emb = self.pos_encoder(token_emb)
        tgt_emb = self.pos_encoder(tgt_emb)

        if isinstance(query_padding_mask, torch.Tensor):
            query_emb = query_emb * math.sqrt(self.config.dec_hidden_dim)
            query_emb = self.pos_encoder(query_emb)
        else:
            query_emb = None

        # Iterate through the local and global encoders
        # Global doc encoding will not be generated until first pass
        global_doc_emb = None
        # Negate query mask if provided
        if isinstance(query_padding_mask, torch.Tensor):
            query_padding_mask = ~query_padding_mask

        for layer_type, layer in zip(self.encoder_layers, self.encoder):
            logging.debug("layer type: %s", layer_type)
            if layer_type == 'local':
                # Automatic layer forwarding is done here by pytorch, so no need to iterate through local_enc_layers
                # Cannot use src_key_padding_mask with fully padded docs. Why? b/c
                # softmax + vectors completely full of -inf = :(
                # src_key_padding_mask should be of shape (batch_size, max_seq_len)
                # and will pad the full hidden dimnension of a token position.
                # With doc sampling this won't be a problem, but that means there has to be a option for it in the model
                # local_doc_emb = layer(doc_emb, src_key_padding_mask=~src_padding_mask)
                local_doc_emb = layer(token_emb, src_key_padding_mask=None)
                logging.info("local layer emb: %s", local_doc_emb)
            elif layer_type == 'global':
                logging.debug("local doc emb size after reshaping %s:", local_doc_emb.size())
                # Pytorch TransformerEncoder cannot handle custom arguments
                # so can't wrap the encoder layer with it to handle layer iteration
                for global_layer in range(self.config.global_enc_layers):
                    local_doc_emb, global_doc_emb, query_emb = layer(
                                                                [local_doc_emb, global_doc_emb, query_emb],
                                                                query_padding_mask=query_padding_mask,
                                                                doc_key_padding_mask=~doc_padding_mask,
                                                                src_key_padding_mask=~src_padding_mask)
                    logging.debug("local doc emb size after global enc %s:", local_doc_emb.size())
                    logging.debug("global doc emb size after global_enc %s:", global_doc_emb.size())
                    if isinstance(query_padding_mask, torch.Tensor):
                        logging.debug("query emb size after global_enc %s:", query_emb.size())
                    local_doc_emb = local_doc_emb.view(
                        self.config.max_seq_len,
                        self.config.batch_size * self.config.max_docs,
                        self.config.enc_hidden_dim)
        logging.debug("global doc emb size after encoder %s:", global_doc_emb.size())
        logging.debug("local doc emb size after encoder %s:", local_doc_emb.size())

        # Probably not going to use top k unless I come up with a better way to do it.
        # Maybe just use at inference?
        if self.config.k_docs is not None:
            # Use softmax to select the topk documents, and provide the local encoder embeddings
            doc_likelikehoods = self.doc_softmax(global_doc_emb)
            # Select the topk k docs from the attention representations weighted with mmr or other measures
            # such as contradiction
            k_docs = torch.topk(doc_likelikehoods, k=self.config.k_docs, dim=0)
            k_doc_emb = k_docs.values
            # Don't mask any documents if using topk
            doc_padding_mask = None

        if self.config.decoder_attn_mask:
            # Causal mask to prevent decoder from seeing the fuuuture
            tgt_attn_mask = self.generate_square_subsequent_mask(self.config.max_seq_len)
            logging.info("tgt attn mask %s", tgt_attn_mask.size())
        else:
            tgt_attn_mask = None

        logging.warning("Use global or local representations for decoder?")
        if doc_padding_mask is not None:
            doc_padding_mask = ~doc_padding_mask

        # Don't need to negate attn mask as it is correctly generated as is
        output = self.decoder(
            tgt_emb, global_doc_emb,
            tgt_mask=tgt_attn_mask,
            tgt_key_padding_mask=~tgt_padding_mask,
            memory_key_padding_mask=doc_padding_mask,
            memory_mask=None)

        #logging.debug("output emb size after decoding: %s", output.size())
        output = self.final_linear_layer(output)
        logging.debug("output after linear layer: %s", output.size())
        # Final reshape for cross entropy (Soft max and NLLoss), where the number of classes
        # is dim == 1
        output = output.permute(1, 2, 0)
        logging.debug("output %s", output.size())

        return output


class TransformerGlobalEncoderLayer(nn.Module):
    """
    TransformerGlobalEncoderLayer is a custom layer for this work.
    Boiler plate transformer code from
    https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer

    The layer is composed of self-attn and feedforward network, using pytorch's implementation.
    It is intended to be used with a hierarchical model, and should receive input
    from multiple local encoders.

    It diverges from the standard pytorch transformer in that it first uses multi-head pooling
    to generate document level representations. It then uses those representations to compute MMR
    and multi head attention between documents, and adds that context between documents back into
    the token representations.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        pooling: True or False, for multi head pooling

    Example:
        >>> from torch import nn
        >>> from model import TransformerGlobalEncoderLayer
        >>> src = torch.rand(10, 32, 512)
        >>> encoder_layer = TransformerGlobalEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> out = transformer_encoder(src)
    """

    def __init__(
        self, d_model, max_seq_len, max_docs,
        batch_size, nhead, mmr=False, query_doc_attn=False,
        head_pooling=False, dim_feedforward=2048,
        dropout=0.1, activation="relu"):

        super().__init__()

        # Definintions from pytorch encoder layer
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        if activation == "relu":
            self.activation = nn.functional.relu
        else:
            raise IOError("Please specify 'relu' activation")
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.d_model = d_model

        # New variables
        self.max_seq_len = max_seq_len
        self.max_docs = max_docs
        self.batch_size = batch_size
        self.head_pooling = head_pooling
        self.mmr = mmr
        self.query_doc_attn = query_doc_attn

        if mmr:
            self.mmr_attention = MMR(d_model, max_seq_len)
        # TODO: Add option for using cls token as doc representation
        if head_pooling:
            self.head_pooling = MultiHeadPooling(
                max_seq_len, max_docs, batch_size, d_model, nhead, dropout=dropout
                )
        logging.warning("Register buffer?")
        #self.register_buffer('', stuff)

    def forward(self, input, src_mask=None, src_key_padding_mask=None, query_padding_mask=None, doc_key_padding_mask=None):
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
        # Take 0th element, in order to be able to output a tuple with multiple layers
        # Query is in 2nd index. Global doc is in the 1st
        if not isinstance(query_padding_mask, torch.Tensor) and (self.mmr or self.query_doc_attn):
            raise IOError("Please provide query and query mask if mmr or query_doc_attn is set to true.")

        query_emb = input[2]
        input = input[0]
        logging.info("initial input %s", input)
        logging.info("input_size %s", input.size())
        logging.info("query size %s", query_emb.size())
        logging.info("src padding_mask size %s", src_key_padding_mask.size())
        if self.head_pooling:
            x_pooled = self.head_pooling(input, input, src_key_padding_mask)
            #probably should normalize
        else:
            x_pooled = input.sum(0).div(self.max_seq_len)
            logging.info(x_pooled.size())
            # Reshape to match output shape of multihead pooling layer
            x_pooled = x_pooled.view(self.batch_size, self.max_docs, self.d_model).transpose(0, 1).contiguous()
        logging.info("Pooling size %s", x_pooled.size())
        logging.info("x pooled %s", x_pooled)
        # Once the token representations have been pooled, we can compute attention between documents
        # using the parent encoder layer method
        doc_attn = self.self_attn(x_pooled, x_pooled, x_pooled, attn_mask=src_mask,
                              key_padding_mask=doc_key_padding_mask)[0]
        logging.info("doc attn size %s", doc_attn.size())
        logging.info("inter doc attn %s", doc_attn)
        # Two ways to inject query knowledge into context embeddings:
        # Either compute mmr attention b/w docs and query, or
        # compute multi-headed self attention with pytorch module.
        # Additionally, it is possible to directly use the document
        # vectors without query information
        if self.mmr:
            doc_attn = self.mmr_attention(doc_attn, query_emb, doc_key_padding_mask, query_padding_mask)
            logging.info("doc mmr weighted size %s", doc_attn.size())
        elif self.query_doc_attn:
            doc_attn = self.self_attn(doc_attn, query_emb, query_emb, attn_mask=src_mask,
                              key_padding_mask=query_padding_mask)[0]
            logging.info("doc query attn size %s", doc_attn.size())

        # Shape last three dimensions of input to match dimension of doc rep: (max_docs, batch_size, hidden dim)
        input = input.contiguous().view(-1, self.batch_size, self.max_docs, self.d_model).transpose(1, 2)
        logging.info("input for adding context %s", input.size())
        # Broadcast doc embeddings to dim 0 (seq length) of word emb
        word_emb = input + self.dropout1(doc_attn)
        # Undo reshaping
        word_emb = word_emb.transpose(1, 2).view(self.max_seq_len, self.batch_size * self.max_docs, self.d_model)
        # Apply the rest of the transformer layer as in the Pytorch encoder layer
        word_emb_1 = self.norm1(word_emb)
        word_emb_2 = self.linear2(self.dropout(self.activation(self.linear1(word_emb))))
        word_emb = word_emb_1 + self.dropout2(word_emb_2)
        word_emb = self.norm2(word_emb)

        return word_emb, doc_attn, query_emb


class PositionalEncoding(nn.Module):
    """
    Class for computing position embeddings from
    https://github.com/pytorch/tutorials/blob/master/beginner_source/transformer_tutorial.py
    No fancy additions were made.

    x: [sequence length, batch size, embed dim]
    output: [sequence length, batch size, embed dim]
    """
    def __init__(self, d_model, dropout=0.1, max_len=128, device=None):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
