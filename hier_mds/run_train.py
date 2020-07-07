import os
import sys
import argparse
import logging
import functools
import json

import GPUtil
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoTokenizer

from hier_mds.data import DatasetRegistry
from hier_mds.model import HierarchicalTransformer
from hier_mds.model.hierarchical_config import HierarchicalTransformerConfig

def get_args():
    """
    Argument defnitions
    """
    parser = argparse.ArgumentParser(description="Arguments for training")

    parser.add_argument("--init_checkpoint", dest="init_checkpoint", default=None, help="Weights to use to initialize model")
    parser.add_argument("--checkpoint_dir", dest="checkpoint_dir", default=None, help="Location to save model")
    parser.add_argument("--prediction_dir", dest="prediction_dir", default=None, help="Location to save predictions")
    parser.add_argument("--data_dir", dest="data_dir", default="", help="Base directory containing data")
    parser.add_argument("--cache_dir", dest="cache_dir", default="", help="Directory to check for cached data")
    parser.add_argument("--tasks", dest="tasks", default="", help="String of task datasets")
    parser.add_argument("--tensorboard", dest="tensorboard", action="store_true", help="Log with tensorboard")
    parser.add_argument("--hf_model", dest="hf_model", default="", help="Specify hugging face tokenizer and model to use for initializing embeddings")

    # Training
    parser.add_argument("--train", dest="train", action="store_true", help="Run training")
    parser.add_argument("--validate", dest="validate", action="store_true", help="Include validation during training")
    parser.add_argument("--test", dest="test", action="store_true", help="Evaluate model on test set after training")
    parser.add_argument("--multitask", dest="multitask", action="store_true", help="Use multiple datasets for multitask training")
    parser.add_argument("--doc_mixing", dest="doc_mixing", action="store_true", help= "Randomly select documents to pad examples with n_docs < max_docs")
    parser.add_argument("--epochs", dest="epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=16, help="Number of batches per epoch")
    parser.add_argument("--eval_batches", dest="eval_batches", type=int, default=16, help="Number of batches for validation")
    parser.add_argument("--eval_batch_size", dest="eval_batch_size", type=int, default=16, help="Size of batches for validation")
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--gpu", type=int, default=0, help="gpu id to use")
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=2, help="Set learning rate")
    parser.add_argument("--label_smoothing", default=None, type=float)
    parser.add_argument("--beta_1", dest="beta_1", type=float, default=.9, help="Beta 1 for optimizer")
    parser.add_argument("--beta_2", dest="beta_2", type=float, default=.998, help="Beta 2 for optimzer")

    # Model hyperparams
    parser.add_argument("--dropout", dest="dropout", type=float, default=.1, help="Dropout")
    parser.add_argument("--max_docs", dest="max_docs", type=int, default=2, help="Max number of documents to consider")
    parser.add_argument("--max_seq_len", dest="max_seq_len", type=int, default=256, help="Max length of one document")
    parser.add_argument("--decoder_type", dest="decoder_type", default="transformer", help="Model to use for decoder: GRU or transformer")
    parser.add_argument("--enc_hidden_dim", dest="enc_hidden_dim", type=int, default=128, help="Dimensions for hidden dim of transformer encoder")
    parser.add_argument("--dec_hidden_dim", dest="dec_hidden_dim", type=int, default=128, help="Dimensions for hidden dim of transformer decoder")
    parser.add_argument("--ffw_dim", dest="ffw_dim", type=int, default=1024, help="Dimensions for all linear feed forward layers")
    parser.add_argument("--local_enc_layers", dest="local_enc_layers", type=int, default=2, help="Local encoder layers")
    parser.add_argument("--global_enc_layers", dest="global_enc_layers", type=int, default=2, help="Global encoder layers")
    parser.add_argument("--dec_layers", dest="dec_layers", type=int, default=2, help="Decoder layers")
    parser.add_argument("--n_att_heads", dest="n_att_heads", type=int, default=4, help="Number of attention heads for any given layer")
    parser.add_argument("--topk_docs", dest="topk_docs", type=int, default=None, help="Top k documents to select")
    parser.add_argument("--padding_mask", dest="padding_mask", action="store_true", help="Use mask padding for src, query and target")
    parser.add_argument("--decoder_attn_mask", dest="decoder_attn_mask", action="store_true", help="Causal mask for decoder")
    parser.add_argument("--mmr", dest="mmr", action="store_true", help="Compute MMR embeddings")
    parser.add_argument("--query_doc_attn", dest="query_doc_attn", action="store_true", help="Inject query into documents with multihead attention")
    parser.add_argument("--init_bert_weights", dest="init_bert_weights", action="store_true", help="Initiate encoder embeddings with BERT weights")
    parser.add_argument("--use_cls_token", dest="use_cls_token", action="store_true", help="Use CLS token in pretrained model weights as document representation")
    parser.add_argument("--multi_head_pooling", dest="multi_head_pooling", action="store_true", help="Use multi head pooling for global encoder layer")

    return parser

def save_model(model, path):
    model.save_pretrained(path)


def shift_input_right(input_ids, pad_token_id):
    """
    Following BART finetuning and HuggingFace implementation, shift input ids
    one token to the right by wrapping the last non pad token (such as <\s> or <eos>) to the beginning.
    See https://github.com/pytorch/fairseq/issues/1389 for a good elaboration about ways to do this.
    """
    shifted_tokens = input_ids.clone()
    index_of_pad = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    shifted_tokens[:, 0] = input_ids.gather(1, index_of_pad).squeeze()
    shifted_tokens[:, 1:] = input_ids[:, :-1]
    # Replace the end of sentence tag with the pad token
    shifted_tokens[:, index_of_pad] = pad_token_id

    return shifted_tokens


def main():
    """
    Main function for training Hierarchical Multi Document summarization transformer.
    While this function contains code for training the model, the model itself should be easy to import
    into any training script
    """

    args = get_args().parse_args()
    torch.manual_seed(args.seed)
    logging.basicConfig(filename="logs/hier_mds.log", filemode="w", level=logging.DEBUG)

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    eos_token = tokenizer.eos_token
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    logging.info("EOS token from %s tokenizer %s, %s", args.hf_model, eos_token, eos_token_id)
    token_encoder = functools.partial(tokenizer.batch_encode_plus,
                                            add_special_tokens=False,
                                            add_space_before_punct_symbol=True,
                                            max_length=args.max_seq_len,
                                            pad_to_max_length=True,
                                            truncation_strategy="only_first",
                                            return_token_type_ids=False,
                                            return_attention_mask=True,
                                            )

    # Set up model config for training
    config = HierarchicalTransformerConfig(
        max_docs=args.max_docs,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        vocab_size=tokenizer.vocab_size,
        local_enc_layers=args.local_enc_layers,
        global_enc_layers=args.global_enc_layers,
        decoder_layers=args.dec_layers,
        dropout=args.dropout,
        enc_hidden_dim=args.enc_hidden_dim,
        dec_hidden_dim=args.dec_hidden_dim,
        ffw_dim=args.ffw_dim,
        n_att_heads=args.n_att_heads,
        decoder_type=args.decoder_type,
        multi_head_pooling=args.multi_head_pooling,
        use_cls_token=args.use_cls_token,
        mmr=args.mmr,
        query_doc_attn=args.query_doc_attn,
        padding_mask=args.padding_mask,
        decoder_attn_mask=args.decoder_attn_mask,
        eos_token=eos_token,
        k_docs=args.topk_docs,
        init_bert_weights=args.init_bert_weights,
    )
    model = HierarchicalTransformer(config)

    if args.tensorboard:
        tb_writer = SummaryWriter()
        tb_writer.add_graph(model)
        tb_writer.close()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda:{}".format(args.gpu))
        logging.info("Using {}".format(device))
        model.to(device)
    else:
        raise NotImplementedError("No implementation for CPU available")
    model.set_device(device)

    # Run training
    if args.train:
        if args.checkpoint_dir is not None:
            os.makedirs(args.checkpoint_dir, exist_ok=True)

        # Map tasks to dataset classes
        data_registry = DatasetRegistry().get_tasks()
        tasks = [task for task in args.tasks.split()]
        logging.info("Tasks provided: %s", tasks)
        task_dict = {}
        for task in tasks:
            if task in data_registry:
                task_dict[task] = data_registry[task]
            else:
                logging.info("Task not found in registry: %s", task)
        if len(task_dict) == 0:
            raise IOError("No valid tasks provided. Make sure task name is one of %s ", data_registry.keys())

        # Call the data builder in task dict, passing the tokenizer, then load with DataLoader
        datasets = []
        for task in task_dict:
            # Huggingface datasets have to be retrieved first, as opposed to just initiated
            if task in ["cnn_dailymail", "eli5"]:
                dataset = task_dict[task](
                    token_encoder, args.max_seq_len, args.max_docs, eos_token,
                    path=args.data_dir, cache_dir=args.cache_dir, doc_mixing=args.doc_mixing).get_dataset()
            else:
                dataset = task_dict[task](
                   token_encoder, args.max_seq_len, args.max_docs, eos_token,
                   path=args.data_dir, doc_mixing=args.doc_mixing)
            datasets.append(dataset)
        concat_data = ConcatDataset(datasets)
        # concat_data = task_dict[task](
        #             token_encoder, args.max_seq_len, args.max_docs, eos_token,
        #             path=args.data_dir, cache_dir=args.cache_dir, doc_mixing=args.doc_mixing)

        training_data = DataLoader(concat_data, batch_size=args.batch_size, shuffle=True)

        # Set up model for training
        model.train()
        optimizer = torch.optim.Adam(
            model.parameters(),
            betas=[args.beta_1,args.beta_2],
            lr=args.learning_rate)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
        if args.label_smoothing is not None:
            criterion = build_loss_compute(
            model.generator, symbols, vocab_size, device, train=True, label_smoothing=args.label_smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        logging.info("Beginning training for {e} epochs, batch size {b}".format(e=args.epochs, b=args.batch_size))
        for epoch in range(args.epochs):
            running_loss = 0
            for i, batch in enumerate(training_data):
                GPUtil.showUtilization()
                # If not query is provided...
                if 'query_ids' in batch:
                    query_batch = batch['query_ids'].to(device)
                    query_mask = batch['query_mask'].to(device)
                else:
                    query_batch = None
                    query_mask = None
                source_batch = batch['source_ids'].to(device)
                source_mask = batch['source_mask'].to(device)
                # Add to device after shift
                target_batch = batch['target_ids'].to(device)
                target_mask = batch['target_mask'].to(device)
                # This check is to deal with hugging face datasets that don't let me convert to tensors
                # before they are called in a batch
                if source_mask.dtype != torch.bool:
                    source_mask = source_mask.to(torch.bool)
                if target_mask.dtype != torch.bool:
                    target_mask = target_mask.to(torch.bool)
                if query_mask.dtype != torch.bool:
                    query_mask = query_mask.to(torch.bool)

                logging.info("source mask: %s", type(source_mask))
                logging.info(target_batch.size())
                # Shift inputs for the decoder to the right so it learns to predict next word, not copy
                decoder_tgt_input = shift_input_right(target_batch, pad_token_id)

                if i == 0 and epoch == 0:
                    # Shape checking
                    assert source_batch.size() == torch.Size([args.batch_size, args.max_docs, args.max_seq_len]), source_batch.size()
                    assert target_batch.size() == torch.Size([args.batch_size, args.max_seq_len]), target_batch.size()
                    idx = 20
                    if query_batch is not None:
                        assert query_batch.size() == torch.Size([args.batch_size, args.max_seq_len]), query_batch.size()
                        logging.info("Encoded query: %s", query_batch[0, :idx])
                        logging.info("Decoded query: %s", tokenizer.decode(
                            query_batch[0, :idx], skip_special_tokens=False))
                    logging.info("Encoded source: %s", source_batch[0, 0, :idx])
                    logging.info("Decoded source: %s", tokenizer.decode(
                        source_batch[0, 0, :idx], skip_special_tokens=False))
                    logging.info("Encoded decoder input:\n%s", decoder_tgt_input[0, :idx])
                    logging.info("Decoded shifted summary:\n%s", tokenizer.decode(
                        decoder_tgt_input[0, :idx], skip_special_tokens=False))

                output = model(
                    source_batch, decoder_tgt_input, query_batch,
                    src_padding_mask=source_mask, tgt_padding_mask=target_mask, query_padding_mask=query_mask)
                logging.info(output)
                # Output will be of dimension (batch size, max seq lenth of summaries, dimension)
                loss = criterion(output, target_batch)
                loss.backward()
                running_loss += loss.item()
                optimizer.step() # Update weights
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scheduler.step() # Update learning rate if appropriate
                optimizer.zero_grad() # zero the gradient buffers
                logging.info("Batch # %s", i)
                logging.warning("FORCED EXIT!")
                break
                #sys.exit()
            model.generate()

            # Run validation if specified
            if args.validate:
                running_valid_loss = 0
                with torch.no_grad():
                    for batch in validation_data:
                        model.eval()
                        input = next(IterableDataset)
                        output = model(input)
                        valid_step_loss = criterion(output, target)
                        running_valid_loss += valid_step_loss.item()
                        valid_steps += 1

                avg_valid_loss = running_valid_loss / valid_steps
                if avg_valid_loss < best_valid_loss:
                    if args.checkpoint_dir is None:
                        raise IOError("Please provide path to save best performing checkpoint")
                    else:
                        save_model(model, "{0}_best".format(checkpoint_dir))

        # Save final model
        if args.checkpoint_dir is not None:
            save_model(model, args.checkpoint_dir)


if __name__ == "__main__":
    main()
