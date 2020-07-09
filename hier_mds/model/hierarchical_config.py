"""
Configuration for Transformer for Multi-Document Summarization
HuggingFace BART configuration used as reference
"""


import logging

from transformers import PretrainedConfig

logger = logging.getLogger(__name__)

class HierarchicalTransformerConfig(PretrainedConfig):
    """
    Configuration for Hierarchical Transformer
    """
    model_type = "hier_transformer"

    def __init__(
        self,
        max_docs=10,
        max_seq_len=128,
        batch_size=32,
        vocab_size=30000,
        local_enc_layers=2,
        global_enc_layers=2,
        decoder_layers=2,
        dropout=.1,
        enc_hidden_dim=128,
        dec_hidden_dim=128,
        ffw_dim=512,
        n_att_heads=8,
        decoder_type="transformer",
        multi_head_pooling=True,
        use_cls_token=False,
        mmr=False,
        query_doc_attn=False,
        padding_mask=True,
        decoder_attn_mask=False,
        k_docs=None,
        init_bert_weights=True,
        hf_model=None,
        eos_token=None,
        eos_token_id=None,
        is_encoder_decoder=True,
        **common_kwargs
    ):
        r"""
            :class:`~hierarchical_mds.HierarchicalTransformerConfig` is the configuration class
            for `HierarchicalTransformerModel`.
            Examples:
                config = HierarchicalTransformerConfig.from_pretrained('hier-transform')
                model = HierarchicalTransformerModel(config)
        """
        # TODO:
        # Which hyperparameters need to be initiated from base class?
        super().__init__(
            **common_kwargs,
        )
        #  For hugging face generate method:
        self.is_encoder_decoder = is_encoder_decoder
        # Hierachical params
        self.max_docs = max_docs
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.local_enc_layers = local_enc_layers
        self.global_enc_layers = global_enc_layers
        self.decoder_layers = decoder_layers
        self.dropout = dropout
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.ffw_dim = ffw_dim
        self.n_att_heads = n_att_heads
        self.decoder_type = decoder_type
        self.multi_head_pooling = multi_head_pooling
        self.use_cls_token = use_cls_token
        self.mmr = mmr
        self.query_doc_attn = query_doc_attn
        self.padding_mask = padding_mask
        self.decoder_attn_mask = decoder_attn_mask
        self.k_docs = k_docs
        # Specify which hugging face model to use to load embeddings.
        # The same cli argument for this will also load the tokenizer,
        # so you don't have to worry about matching them
        self.hf_model = hf_model
        self.init_bert_weights = init_bert_weights
        # Using EOS as BOS for teacher forcing, meaning EOS is wrapped to the beginning of of a seq,
        # and pad token goes at the end
        self.bos_token_id = eos_token_id
        self.eos_token_id = eos_token_id
        self.eos_token = eos_token
        assert self.eos_token is not None, "Please specify eos token used by tokenizer, as this is used for shifting inputs right "
