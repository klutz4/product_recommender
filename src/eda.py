import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from clean_data import get_data

products = get_data()

def plot_nulls():
    sns.heatmap(products.isnull(),yticklabels=False,cbar=False,cmap='viridis')
    plt.title('Null Values of Product Dataset')
    plt.xticks()
    plt.tight_layout()
    plt.savefig('images/nullplot.png')
