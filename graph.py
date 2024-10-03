import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from quant import *
from sparsegpt import *
from modelutils import *
from opt import *

# Assuming args and model variables are defined elsewhere

# Data
perplexities = []
datasets = ['WikiText-2', 'PTB', 'C4']  # Correct names for display

for dataset in datasets:
    dataloader, testloader = get_loaders(dataset.lower(), seed=args.seed, model=args.model, seqlen=model.seqlen)
    # Assuming you have a function to calculate perplexity for the dataset
    perplexity = calculate_perplexity(model, testloader)  # This function needs to exist
    perplexities.append(perplexity)

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
