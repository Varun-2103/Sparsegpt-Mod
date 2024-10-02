import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from quant import *
from sparsegpt import *
from modelutils import *

try:
    import wandb
    has_wandb = True
except:
    has_wandb = False

# Lists to store time and error
time_list = []
error_list = []

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

@torch.no_grad()
def opt_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev) 
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
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
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        
        gpts = {}
        for name in subset:
            if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (not args.invert):
              continue
            gpts[name] = SparseGPT(subset[name])
            if args.wbits < 16:
                gpts[name].quantizer = Quantizer()
                gpts[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=False
                )

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')
            sparsity = args.sparsity
            gpts[name].fasterprune(
                sparsity, prunen=args.prunen, prunem=args.prunem, percdamp=args.percdamp, blocksize=args.blocksize
            )
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

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
    
    # Append the error to the list
    error_list.append(ppl.item())
    
    # Restore cache
    model.config.use_cache = use_cache

if __name__ == '__main__':
    import argparse
    from datautils import *

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
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--sparsity', type=float, default=0,
        help='Target sparsity'
    )
    parser.add_argument(
        '--prunen', type=int, default=0,
        help='N for N:M pruning.'
    )
    parser.add_argument(
        '--prunem', type=int, default=0,
        help='M for N:M pruning.'
    )
    parser.add_argument(
        '--blocksize', type=int, default=128,
        help='Blocksize to use for adaptive mask selection.'
    )
    parser.add_argument(
        '--gmp', action='store_true',
        help='Whether to run the GMP baseline.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16,
        help='Whether to quantize as well.'
    )
    parser.add_argument(
        '--minlayer', type=int, default=-1,
        help='Prune all layers with id >= this.'
    )
    parser.add_argument(
        '--maxlayer', type=int, default=1000,
        help='Prune all layers with id < this.'
    )
    parser.add_argument(
        '--prune_only', type=str, default='self_attn.k_proj',
        help='Prune only this part of the model.'
    )
    parser.add_argument(
        '--invert', action='store_true',
        help='Prune everything except the specified layers.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save the quantized model under this name.'
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

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if (args.sparsity or args.prunen) and not args.gmp:
        tick = time.time()
        opt_sequential(model, dataloader, 'cuda' if torch.cuda.is_available() else 'cpu')
        print(time.time() - tick)

    for dataset in ['wikitext2', 'ptb', 'c4']:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen)
        print(dataset,"Here it is!")
        opt_eval(model, testloader, 'cuda' if torch.cuda.is_available() else 'cpu', dataset, args.log_wandb)

    if args.save:
        model.save_pretrained(args.save)

# Plot the time vs error
if len(time_list) > 0 and len(error_list) > 0:
    plt.plot(time_list, error_list)
    plt.xlabel('Time (s)')
    plt.ylabel('Error (Perplexity)')
    plt.title('Time vs Error Plot')
    plt.show()
else:
    print("No data available to plot.")
