import torch
import os
import pickle
from tqdm import tqdm
from helpers import construct_prompts, process_generated_results
from hf_api import hf_generate
from helpers import get_predicted_values_from_generated_sample


def sample(args, tokenizer, model, results):   
    with torch.no_grad():
        # generate
        results['gen'] = [[] for _ in range(len(results['data']['x_test']))]
        # generate the prompts from the data
        prompts = construct_prompts(
            x_train=results['data']['x_train'],
            y_train=results['data']['y_train'],
            x_test=results['data']['x_test'],
            args=args,
            dim_y=results['dim_y']  
        )

        num_prompts = len(prompts)
        for idx in tqdm(range(num_prompts), desc='Sampling'):
            prompt = prompts[idx]
            samples = []
            num_samples = args.num_samples
            while num_samples > 0:
                bs = min(args.batch_size, num_samples)
                res = hf_generate(
                    model=model,
                    tokenizer=tokenizer,
                    input_str=prompt,
                    batch_size=bs,
                    temp=args.temperature, 
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_new_tokens=args.max_generated_length
                )
                for j in range(len(res)):
                    gen_sample = get_predicted_values_from_generated_sample(
                        generated_input=res[j],
                        args=args,
                        category_names=results['categories']
                        )
                    if gen_sample is not None:
                        samples.append(gen_sample)
                        num_samples -= 1
                del res
            results['gen'][idx] += samples

        # Print out the first sample.
        if args.print_prompts:
            for prompt, gen in zip(prompts, results['gen']):
                print(prompt, flush=True)
                print(f"> {gen[0]}", flush=True)
                print("\n==================================\n", flush=True)

    results['prompts'] = prompts

    results = process_generated_results(gen_results=results, args=args)

    # save off the results
    with open(os.path.join(args.output_dir, args.experiment_name + '.pkl'), "wb") as f:
        pickle.dump(results, f)

    return results