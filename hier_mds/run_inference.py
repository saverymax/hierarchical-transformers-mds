import argparse

import torch
from transformes import AutoTokenizer
from hier_mds.model import HierarchicalTransformer
from hier_mds.model import HierarchicalTransformerConfig

def get_args():
    parser = argparse.ArgumentParser(description="Arguments for inference with trained model")
    parser.add_argument("--checkpoint_dir", dest="checkpoint_dir", default=None, help="Location to save model")
    parser.add_argument("--prediction_dir", dest="prediction_dir", default=None, help="Location to save predictions")
    parser.add_argument("--data_path", dest="data_path", default=None, help="Location of data")
    parser.add_argument("tokenizer", dest="tokenizer", default=None, help="Name of hugging face model to select tokenizer")

    return parser


def data_loader(data_path):
    data = None
    return data


def main():
    """
    Function for running inference with finetuned model
    """
    args = get_args().parse_args()
    data = data_loader(args.data_path)

    with torch.no_grad():
        # Set config with settings used for finetuning
        config = HierarchicalTransformerConfig()
        model = HierarchicalTransformer(config)
        model.load_state_dict(torch.load(args.checkpoint_dir))
        model.eval()
        total = 0
        predictions = []
        token = AutoTokenizer.from_pretrained(model)
        for batch in testloader:
            outputs = model.generate(
                                batch,
                                attention_mask=forward_params['attention_mask'],
                                do_sample=False,
                                max_length=max_len,
                                min_length=min_len,
                                num_beams=4,
                                length_penalty=2.,
                                no_repeat_ngram_size=3,
                                early_stopping=True)
            tokens = token_decoder(outputs)

            predictions.append(tokens)

        pred_dict = {'predictions': predictions}

    with open(prediction_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f)

if __name__ == "__main__"
    main()
