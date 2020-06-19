import os
import argparse
import logging

import torch
from torch.utils.data import DataLoader

from data import RandomDataset
from model import HierarchicalMDS

def get_args():
    """
    Argument defnitions
    """
    parser = argparse.ArgumentParser(description="Arguments for training")

    parser.add_argument("--init_checkpoint", dest="init_checkpoint", default=None, help="Weights to use to initialize model")
    parser.add_argument("--checkpoint_dir", dest="checkpoint_dir", default=None, help="Location to save model")
    parser.add_argument("--prediction_dir", dest="prediction_dir", default=None, help="Location to save predictions")
    parser.add_argument("--data_path", dest="data_path", default=None, help="Location of data")

    # Training
    parser.add_argument("--train_tok", dest="train_tok", action="store_true", help="Train sentencepiece tokenizer on training data")
    parser.add_argument("--vocab_size", dest="vocab_size", type=int, default=10, help="Size of vocab")
    parser.add_argument("--train", dest="train", action="store_true", help="Run training")
    parser.add_argument("--validate", dest="validate", action="store_true", help="Include validation during training")
    parser.add_argument("--test", dest="test", action="store_true", help="Evaluate model on test set after training")
    parser.add_argument("--epochs", dest="epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=16, help="Number of batches per epoch")
    parser.add_argument("--eval_batches", dest="eval_batches", type=int, default=16, help="Number of batches for validation")
    parser.add_argument("--eval_batch_size", dest="eval_batch_size", type=int, default=16, help="Size of batches for validation")
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--gpu", type=int, default=0, help="gpu id to use")

    # Model hyperparams
    parser.add_argument("--beta_1", dest="beta_1", type=float, default=.9, help="Set learning rate")
    parser.add_argument("--beta_2", dest="beta_2", type=float, default=.998, help="Set learning rate")
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=2, help="Set learning rate")
    parser.add_argument("--dropout", dest="dropout", type=float, default=.1, help="Dropout")
    parser.add_argument("--max_docs", dest="max_docs", type=int, default=2, help="Max number of documents to consider")
    parser.add_argument("--max_seq_len", dest="max_seq_len", type=int, default=256, help="Max length of one document")
    parser.add_argument("--decoder_type", dest="decoder_type", default="transformer", help="Model to use for decoder: GRU or transformer")
    parser.add_argument("--enc_hidden_dim", dest="enc_hidden_dim", type=int, default=128, help="Dimensions for encoder")
    parser.add_argument("--dec_hidden_dim", dest="dec_hidden_dim", type=int, default=128, help="Dimensions for decoder")
    parser.add_argument("--local_enc_layers", dest="local_enc_layers", type=int, default=2, help="Local encoder layers")
    parser.add_argument("--global_enc_layers", dest="global_enc_layers", type=int, default=2, help="Global encoder layers")
    parser.add_argument("--dec_layers", dest="dec_layers", type=int, default=2, help="Global encoder layers")
    parser.add_argument("--n_att_heads", dest="n_att_heads", type=int, default=4, help="Number of attention heads for any given layer")
    parser.add_argument("--top_k", dest="top_k", type=int, default=5, help="Top k documents to select")
    parser.add_argument("--label_smoothing", default=None, type=float)
    parser.add_argument("--mask", dest="mask", action="store_true", help="Boolean to use mask")
    parser.add_argument("--mmr", dest="mmr", action="store_true", help="Compute MMR embeddings")
    parser.add_argument("--init_bert_weights", dest="init_bert_weights", action="store_true", help="Initiate encoders with BERT weights")
    parser.add_argument("--local_bert_enc", dest="local_bert_enc", action="store_true", help="Initiate encoders with BERT weights")
    parser.add_argument("--multi_head_pooling", dest="multi_head_pooling", action="store_true", help="Use multi head pooling for global encoder layer")

    return parser


def main():
    """Main function for training Hierarchical Multi Document summarization model.
    While this function contains code for training the model, the model itself should be easy to import
    into any training script
    """

    torch.manual_seed(args.seed)
    logging.basicConfig(filename="logs/hier_mds.log", filemode="w", level=logging.DEBUG)

    doc_emb = DataLoader(dataset=RandomDataset((args.batch_size * args.epochs, args.max_docs, args.max_seq_len)),
                             batch_size=args.batch_size, shuffle=True)

    query_emb = DataLoader(dataset=RandomDataset((args.batch_size * args.epochs, args.max_seq_len)),
                             batch_size=args.batch_size, shuffle=True)

    summary_emb = DataLoader(dataset=RandomDataset((args.batch_size * args.epochs, args.max_seq_len)),
                             batch_size=args.batch_size, shuffle=True)

    model = HierarchicalMDS(args)
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
        model.train()
        optimizer = torch.optim.Adam(
            model.parameters(),
            betas=[args.beta_1,args.beta_2],
            lr=args.learning_rate)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
        # TODO: Add smoothed loss from open NMT
        # Smoothed cross entropy loss
        # train_criterion = build_loss_compute(
        #    model.generator, symbols, vocab_size, device, train=True, label_smoothing=args.label_smoothing)
        # Simpler cross entropy
        criterion = torch.nn.CrossEntropyLoss()

        logging.info("Beginning training for {e} epochs, batch size {b}".format(e=args.epochs, b=args.batch_size))
        for epoch in range(args.epochs):
            running_loss = 0
            #for step in args.steps_per_epoch:
            for doc_batch, query_batch, summary_batch in zip(doc_emb, query_emb, summary_emb):
                doc_batch, query_batch, summary_batch = doc_batch.to(device), query_batch.to(device), summary_batch.to(device)
                #print(doc_batch)
                assert doc_batch.size() == torch.Size([args.batch_size, args.max_docs, args.max_seq_len])
                assert summary_batch.size() == torch.Size([args.batch_size, args.max_seq_len])
                #input = next(IterableDataset)
                output = model(doc_batch, query_batch, summary_batch)
                logging.info(output)
                logging.info(summary_batch)
                # Output will be of dimension (batch size, max seq lenth of summaries, dimension)
                loss = criterion(output, summary_batch)
                loss.backward()
                running_loss += loss.item()
                optimizer.step() # Update weights
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scheduler.step() # Update learning rate if appropriate
                optimizer.zero_grad() # zero the gradient buffers

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
                    torch.save(model.state_dict(), args.checkpoint_dir)


if __name__ == "__main__":
    global args
    args = get_args().parse_args()
    main()
