###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score, accuracy_score


def distribution(data, transformed = False):
    """
    Visualization code for displaying skewed distributions of features
    """
    
    # Create figure
    fig = pl.figure(figsize = (11,5));

    # Skewed feature plotting
    for i, feature in enumerate(['capital-gain','capital-loss']):
        ax = fig.add_subplot(1, 2, i+1)
        ax.hist(data[feature], bins = 25, color = '#00A0A0')
        ax.set_title("'%s' Feature Distribution"%(feature), fontsize = 14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
        ax.set_ylim((0, 2000))
        ax.set_yticks([0, 500, 1000, 1500, 2000])
        ax.set_yticklabels([0, 500, 1000, 1500, ">2000"])

    # Plot aesthetics
    if transformed:
        fig.suptitle("Log-transformed Distributions of Continuous Census Data Features", \
            fontsize = 16, y = 1.03)
    else:
        fig.suptitle("Skewed Distributions of Continuous Census Data Features", \
            fontsize = 16, y = 1.03)

    fig.tight_layout()
    fig.show()


def evaluate(results, accuracy, f1):
    """
    Visualization code to display results of various learners.

    inputs:
      - results: dict of model_name -> {idx -> metrics dict} as returned by train_predict()
      - accuracy: The score for the naive predictor
      - f1: The score for the naive predictor
    """

    # Create figure
    fig, ax = pl.subplots(2, 3, figsize=(13, 7))

    learners = list(results.keys())
    n_learners = len(learners)

    # Infer which training-size indices exist (supports 1 or many)
    train_sizes = sorted(next(iter(results.values())).keys())
    x = np.arange(len(train_sizes))

    # Labels (default assumes legacy 1%,10%,100%; falls back gracefully)
    legacy_labels = {0: "1%", 1: "10%", 2: "100%"}
    xticklabels = [legacy_labels.get(i, str(i)) for i in train_sizes]

    # If we only have one training size, make it explicit
    if len(train_sizes) == 1:
        xticklabels = ["100%"]

    # Dynamic bar width so groups fit nicely
    bar_width = min(0.8 / max(n_learners, 1), 0.25)

    # Use a colormap for many models (keeps it simple and scalable)
    cmap = pl.get_cmap('tab20')
    colors = [cmap(i % 20) for i in range(n_learners)]

    metrics = ['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']

    for k, learner in enumerate(learners):
        offset = (k - (n_learners - 1) / 2) * bar_width
        for j, metric in enumerate(metrics):
            values = [results[learner][i][metric] for i in train_sizes]
            ax[j // 3, j % 3].bar(x + offset, values, width=bar_width, color=colors[k])

    # Common x formatting
    for j in range(len(metrics)):
        ax[j // 3, j % 3].set_xticks(x)
        ax[j // 3, j % 3].set_xticklabels(xticklabels)
        ax[j // 3, j % 3].set_xlabel("Training Set Size")

    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")

    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")

    # Add horizontal lines for naive predictors
    ax[0, 1].axhline(y=accuracy, linewidth=1, color='k', linestyle='dashed')
    ax[1, 1].axhline(y=accuracy, linewidth=1, color='k', linestyle='dashed')
    ax[0, 2].axhline(y=f1, linewidth=1, color='k', linestyle='dashed')
    ax[1, 2].axhline(y=f1, linewidth=1, color='k', linestyle='dashed')

    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Legend (moved to bottom, global to figure)
    patches = [mpatches.Patch(color=colors[i], label=learners[i]) for i in range(n_learners)]
    fig.legend(
        handles=patches,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.06),
        ncol=min(4, max(1, n_learners)),
        fontsize='medium',
        frameon=False
    )

    pl.suptitle(f"Performance Metrics for {n_learners} Supervised Learning Models", fontsize=16, y=1.02)
    pl.tight_layout(rect=[0, 0.08, 1, 0.96])
    pl.show()
def feature_plot(importances, X_train, y_train):
    
    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:5]]
    values = importances[indices][:5]

    # Creat the plot
    fig = pl.figure(figsize = (9,5))
    pl.title("Normalized Weights for First Five Most Predictive Features", fontsize = 16)
    pl.bar(np.arange(5), values, width = 0.6, align="center", color = '#00A000', \
          label = "Feature Weight")
    pl.bar(np.arange(5) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#00A0A0', \
          label = "Cumulative Feature Weight")
    pl.xticks(np.arange(5), columns)
    pl.xlim((-0.5, 4.5))
    pl.ylabel("Weight", fontsize = 12)
    pl.xlabel("Feature", fontsize = 12)
    
 #  pl.legend(loc = 'lower center')
    pl.tight_layout(rect=[0, 0.08, 1, 0.95])
    pl.show()  
