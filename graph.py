import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from quant import *
from sparsegpt import *
from modelutils import *
from opt import *

# Data
perplexities = []
for dataset in ['wikitext2', 'ptb', 'c4']:
 dataloader, testloader = get_loaders(dataset, seed=args.seed, model=args.model, seqlen=model.seqlen)
for in range(0,3):
{
 perplexities[i]=dataset;}
datasets = ['PTB', 'C4', 'WikiText-2']
# Create bar graph
plt.figure(figsize=(8, 6))
plt.bar(datasets, perplexities, color=['blue', 'green', 'orange'])

# Add labels and title
plt.xlabel('Datasets')
plt.ylabel('Perplexity')
plt.title('Perplexity Comparison Across Datasets')

# Display the values on top of the bars
for i, value in enumerate(perplexities):
    plt.text(i, value + 1, f'{value:.2f}', ha='center')

# Show plot
plt.show()
