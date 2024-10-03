import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Assuming these are defined elsewhere or imported
from quant import *
from sparsegpt import *
from modelutils import *
from opt import *

# Mocking the get_loaders function
def get_loaders(dataset, seed, model, seqlen):
    # This function should return a dataloader and a testloader based on the dataset.
    # For now, return mock loaders
    return None, None  # Replace with actual dataloader and testloader

# Mocking the opt_eval function
def opt_eval(model, testloader, device, dataset, log_wandb):
    # For testing purposes, let's print what this function would do
    print(f"Evaluating {dataset} on {device}...")
    return random.uniform(10, 30)  # Returning random perplexity as a placeholder

# Assuming args and model are defined, mock them if necessary
class Args:
    seed = 42
    model = 'some_model'
    log_wandb = False

args = Args()

class Model:
    seqlen = 512

model = Model()

# Data
perplexities = []
datasets_display = ['WikiText-2', 'PTB', 'C4']  # For displaying the dataset names

# Loop through datasets, evaluating model and capturing perplexities
for dataset in ['wikitext2', 'ptb', 'c4']:
    dataloader, testloader = get_loaders(dataset, seed=args.seed, model=args.model, seqlen=model.seqlen)
    
    # Log evaluation process
    print(f"{dataset} - Here it is!")
    
    # Evaluate the model using opt_eval
    perplexity = opt_eval(model, testloader, 'cuda' if torch.cuda.is_available() else 'cpu', dataset, args.log_wandb)
    
    # Store the perplexity value
    perplexities.append(perplexity)

# Create bar graph for the perplexities
plt.figure(figsize=(8, 6))
plt.bar(datasets_display, perplexities, color=['blue', 'green', 'orange'])

# Add labels and title
plt.xlabel('Datasets')
plt.ylabel('Perplexity')
plt.title('Perplexity Comparison Across Datasets')

# Display the values on top of the bars
for i, value in enumerate(perplexities):
    plt.text(i, value + 1, f'{value:.2f}', ha='center')

# Show plot
plt.show()
