import numpy as np
import os
import pickle
from hf_api import get_model_and_tokenizer
from parse_args import parse_command_line
from compute_nll import compute_nll
from sample import sample
from prepare_data import prepare_data
from helpers import compute_classification_metrics


def run_jolt(args, model, tokenizer):
    np.random.seed(args.seed)

    # create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    results = prepare_data(args)

    if 'sample' in args.mode:
        results = sample(args, tokenizer, model, results)
    
    if 'logpy' in args.mode:
        results = compute_nll(args, tokenizer, model, results)
        for i, type in enumerate(args.y_column_types):
            if type == 'categorical':
                results = compute_classification_metrics(results=results, column_index=i)

    with open(os.path.join(args.output_dir, args.experiment_name + '.pkl'), "wb") as f:
        pickle.dump(results, f)

    return results


def main():
    # parse the command line arguments
    args = parse_command_line()

    # get the llm and asociated tokenizer
    model, tokenizer = get_model_and_tokenizer(args)
    run_jolt(args=args, model=model, tokenizer=tokenizer)


if __name__ == '__main__':
    main()
