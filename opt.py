import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from quant import *
from sparsegpt import *
from modelutils import *
from graph import *
from datautils import *

try:
    import wandb
    has_wandb = True
except:
    has_wandb = False

# Lists to store time and error
time_list = []
error_list = []

# Function to get the model
def get_opt(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    return model

# The dataloader function (assuming it's part of datautils or imported properly)
def get_loaders(dataset, seed, model, seqlen):
    # Modify this function to return the actual dataloader and testloader
    dataloader, testloader = DataLoader(...), DataLoader(...)  # Replace with your logic
    return dataloader, testloader

# Mocked opt_eval (performs evaluation and calculates perplexity)
@torch.no_grad()
def opt_eval(model, testenc, dev, dataset: str, log_wandb: bool = False):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    
    layers[0] = layers[0].to(dev)
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    
    # Time tracking
    start_time = time.time()

    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    
    elapsed_time = time.time() - start_time
    time_list.append(elapsed_time)

    # Compute the error (loss or perplexity)
    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[ :, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    
    # Append the error (perplexity) to the list
    error_list.append(ppl.item())
    
    # Restore cache
    model.config.use_cache = use_cache

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str, 
        help='OPT model to load; pass facebook/opt-X.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed', type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--log_wandb', action='store_true',
        help='Whether to log results to Weights and Biases'
    )

    args = parser.parse_args()

    if args.log_wandb:
        assert has_wandb, "wandb not installed, try pip install wandb"
        wandb.init(config=args)

    model = get_opt(args.model)
    model.eval()

    datasets = ['wikitext2', 'ptb', 'c4']
    perplexities = []
    dataset_names = ['WikiText-2', 'PTB', 'C4']

    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset, "Here it is!")
        opt_eval(model, testloader, 'cuda' if torch.cuda.is_available() else 'cpu', dataset, args.log_wandb)

    # Create a bar graph for perplexity
    plt.figure(figsize=(8, 6))
    plt.bar(dataset_names, error_list, color=['blue', 'green', 'orange'])

    # Add labels and title
    plt.xlabel('Datasets')
    plt.ylabel('Perplexity')
    plt.title('Perplexity Comparison Across Datasets')

    # Display the values on top of the bars
    for i, value in enumerate(error_list):
        plt.text(i, value + 1, f'{value:.2f}', ha='center')

    # Show plot
    plt.show()

    # Optionally, plot time vs error
    if len(time_list) > 0 and len(error_list) > 0:
        plt.figure(figsize=(8, 6))
        plt.plot(time_list, error_list)
        plt.xlabel('Time (s)')
        plt.ylabel('Error (Perplexity)')
        plt.title('Time vs Error Plot')
        plt.show()
    else:
        print("No data available to plot.")
