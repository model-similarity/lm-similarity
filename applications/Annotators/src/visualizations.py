import sys
import os
import matplotlib.pyplot as plt
from networkx import weakly_connected_components
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import json
from scipy.cluster.hierarchy import linkage, leaves_list
import seaborn as sns
from statsmodels.stats.proportion import proportion_confint  # Import for confidence intervals
from tqdm import tqdm, trange
import scienceplots

from metrics import get_acc, get_jsd, get_kappa_mcqs, get_probkappa_mcqs, get_diffp, populate_preds, precomputed_accs, precomputed_diffs, acc_buckets

#########################################
# Plotting Functions
#########################################

def grouped_bar_plot_accs(accs, split='test', metric_name="Accuracy", savepath=None, datasets_list=None, average_across_datasets=False):
    plt.style.use(['science', 'no-latex'])
    fig, ax = plt.subplots(figsize=(10, 6))
    # colors = ['#f28e2b', '#e15759', '#000000', '#76b7b2', '#59a14f', '#4e79a7']

    # Convert {model: {dset: val}} to {model: [vals]} if needed
    if len(accs) > 0 and isinstance(list(accs.values())[0], dict):
        if datasets_list is None:
            datasets_list = list(list(accs.values())[0].keys())
        for m in accs:
            accs[m] = [accs[m][d] for d in datasets_list]
    else:
        if datasets_list is None:
            raise ValueError("datasets_list must be provided if accs are already lists.")
    
    model_names = list(accs.keys())
    mname_dict = {"weak_ft": "Weak Supervisor", "strong_base2": "Initial Strong Student", "w2s": "Weak-to-Strong", "strong2" : "Elicitation Ceiling", "diff_ceil" : "Elicitation + Complementary Knowledge Ceiling"}


    if average_across_datasets:
        # Calculate average for each model across datasets
        avg_accs = {model: np.mean(vals) for model, vals in accs.items()}
        
        # For averaged plot, use wider bars with spacing
        barWidth = 0.6
        spacing = 1  # Space between bars
        positions = np.arange(len(model_names)) * spacing

        # Create bars
        bars = ax.bar(positions, 
                     [avg_accs[model] for model in model_names],
                     width=barWidth, 
                    #  color=colors[:len(model_names)]
                    )
        
        # Set x-ticks at bar positions
        ax.set_xticks(positions)
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=16)
    else:
        # Original grouped bar plot logic
        barWidth = 0.12
        num_datasets = len(datasets_list)
        r = range(num_datasets)
        
        for i, mname in enumerate(model_names):
            vals = accs[mname]
            ax.bar([x + barWidth * i for x in r], vals, width=barWidth, 
                #   color=colors[i % len(colors)],
                 label=mname_dict.get(mname, mname))
        
        ax.set_xticks([x + barWidth * (len(model_names) - 1) / 2 for x in r])
        dataset_labels = [d.replace("_", "-") for d in datasets_list]
        ax.set_xticklabels(dataset_labels, rotation=45, ha='right', fontsize=16)
        ax.legend(
            title='Models',
            loc='upper center', 
            bbox_to_anchor=(0.5, -0.3),  # Move legend below plot
            fontsize=14, 
            title_fontsize=16,
            ncol=len(model_names)//2 + 1
        )
        
    
    ax.set_yticks(range(0, 110, 10))
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylabel(metric_name, fontsize=18)
    # title_prefix = "Average " if average_across_datasets else ""
    # ax.set_title(f"{title_prefix}{metric_name} of Models on {split.capitalize()} Split", 
                # fontsize=18, fontweight='bold')
    ax.set_ylim(50, 100)
    # ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, bbox_inches='tight')
        print(f"Saved grouped bar plot to {savepath}")

def plot_conf_mat_from_buckets(
    buckets_dict, 
    datasets_list,
    mref1="weak_ft",
    mref2="strong_base",
    macc="w2s",
    savepath=None,
    confidence_level=0.95,
    average_across_datasets=False
):
    """
    Plots 2x2 confusion matrices from precomputed buckets_dict.
    If average_across_datasets is True, combines data across all datasets into a single plot.
    """
    if average_across_datasets:
        # Combine buckets across datasets
        combined_buckets = {
            'cc': {'correct': 0, 'total': 0},
            'cw': {'correct': 0, 'total': 0},
            'wc': {'correct': 0, 'total': 0},
            'ww': {'correct': 0, 'total': 0}
        }
        
        for dset in datasets_list:
            for bucket in ['cc', 'cw', 'wc', 'ww']:
                combined_buckets[bucket]['correct'] += buckets_dict[dset][bucket]['correct']
                combined_buckets[bucket]['total'] += buckets_dict[dset][bucket]['total']
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        plot_single_conf_mat(combined_buckets, ax, mref1, mref2, macc, 
                           "Average across datasets", confidence_level)
    else:
        num_datasets = len(datasets_list)
        fig, axes = plt.subplots(1, num_datasets, figsize=(5 * num_datasets, 5))
        if num_datasets == 1:
            axes = [axes]
        
        for dset, ax in zip(datasets_list, axes):
            plot_single_conf_mat(buckets_dict[dset], ax, mref1, mref2, macc, 
                               dset, confidence_level)
    
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
        plt.close()
        print(f"Saved confusion matrices to {savepath}")
    else:
        plt.show()

def plot_single_conf_mat(buckets, ax, mref1, mref2, macc, title_prefix, confidence_level):
    """Helper function to plot a single confusion matrix."""
    correct_cc, total_cc = buckets['cc']['correct'], buckets['cc']['total']
    correct_cw, total_cw = buckets['cw']['correct'], buckets['cw']['total']
    correct_wc, total_wc = buckets['wc']['correct'], buckets['wc']['total']
    correct_ww, total_ww = buckets['ww']['correct'], buckets['ww']['total']

    total = total_cc + total_cw + total_wc + total_ww

    # Calculate confidence intervals
    ci_cc = proportion_confint(correct_cc, total_cc, alpha=1-confidence_level, method='wilson') if total_cc > 0 else (0, 0)
    ci_cw = proportion_confint(correct_cw, total_cw, alpha=1-confidence_level, method='wilson') if total_cw > 0 else (0, 0)
    ci_wc = proportion_confint(correct_wc, total_wc, alpha=1-confidence_level, method='wilson') if total_wc > 0 else (0, 0)
    ci_ww = proportion_confint(correct_ww, total_ww, alpha=1-confidence_level, method='wilson') if total_ww > 0 else (0, 0)

    cm = np.array([
        [correct_cc/total_cc if total_cc > 0 else 0.0, correct_cw/total_cw if total_cw > 0 else 0.0],
        [correct_wc/total_wc if total_wc > 0 else 0.0, correct_ww/total_ww if total_ww > 0 else 0.0]
    ])

    cm_text = np.array([
        [
            f"\n\n +-{(ci_cc[1]-ci_cc[0])/2:.2f} (n:{total_cc/total*100:.2f}%)" if total_cc > 0 else "0.00\n(0, 0)",
            f"\n\n +-{(ci_cw[1]-ci_cw[0])/2:.2f} (n:{total_cw/total*100:.2f}%)" if total_cw > 0 else "0.00\n(0, 0)"
        ],
        [
            f"\n\n +-{(ci_wc[1]-ci_wc[0])/2:.2f} (n:{total_wc/total*100:.2f}%)" if total_wc > 0 else "0.00\n(0, 0)",
            f"\n\n +-{(ci_ww[1]-ci_ww[0])/2:.2f} (n:{total_ww/total*100:.2f}%)" if total_ww > 0 else "0.00\n(0, 0)"
        ]
    ])

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Correct", "Wrong"])
    disp.plot(ax=ax, cmap="Greens", colorbar=False, values_format=".2f")
    
    for text in disp.text_.ravel():
        text.set_fontsize(14)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm_text[i, j], ha="center", va="center", fontsize=12, color="orange")

    total_correct = correct_cc + correct_cw + correct_wc + correct_ww
    overall_acc = total_correct/total if total > 0 else 0.0
    ci_overall = proportion_confint(total_correct, total, alpha=1-confidence_level, method='wilson') if total > 0 else (0, 0)

    ax.set_title(
        f"{title_prefix}\nTotal: {total}\n{macc} Acc: {overall_acc*100:.1f}% "
        f"({ci_overall[0]*100:.1f}%, {ci_overall[1]*100:.1f}%)",
        fontsize=14
    )
    ax.set_xlabel(f"{mref2} Prediction", fontsize=14)
    ax.set_ylabel(f"{mref1} Prediction", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)

import numpy as np
import matplotlib.pyplot as plt

def plot_quadrants_with_subrects(
    quadrant_acc,
    weak_models,
    strong_models,
    mref1="weak_ft",
    mref2="strong_base",
    macc="w2s",
    title_prefix="Disentangling Accuracy of Weak to Strong Models",
    savepath=None,
):
    """
    Plots a 2x2 grid of quadrants (cc, cw, wc, ww), each a *square* subplot,
    subdivided into a W×S grid of cells for (weak_model, strong_model) pairs.
    If W != S, sub-cells become rectangles, but each entire quadrant is square.

    The 4 quadrants are arranged like a confusion matrix:
      top-left = cc  (both correct)
      top-right = cw (weak correct, strong wrong)
      bot-left = wc  (weak wrong, strong correct)
      bot-right= ww  (both wrong)

    One colorbar is placed on the right; "Correct"/"Wrong" ticks appear on
    both axes, and there's a single label for each axis:
      y-axis: "Weak Supervisor Prediction"
      x-axis: "Strong Elicited Prediction"

    The function also prints per-pair accuracies and summary stats for each quadrant.
    """

    W = len(weak_models)
    S = len(strong_models)

    # Prepare four arrays of shape (W, S)
    cc_data = np.zeros((W, S))
    cw_data = np.zeros((W, S))
    wc_data = np.zeros((W, S))
    ww_data = np.zeros((W, S))

    quadrant_values = {
        'cc': [],
        'cw': [],
        'wc': [],
        'ww': []
    }

    # Fill from quadrant_acc
    for i, w_model in enumerate(weak_models):
        for j, s_model in enumerate(strong_models):
            stats = quadrant_acc[(w_model, s_model)]
            for quad_name, arr in zip(('cc','cw','wc','ww'),
                                      (cc_data, cw_data, wc_data, ww_data)):
                c = stats[quad_name]['correct']
                t = stats[quad_name]['total']
                acc = (c / t) if t > 0 else 0.0
                arr[i, j] = acc
                quadrant_values[quad_name].append(acc)

    # Create 2x2 subplots, each forced to be a square
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    # Remove spacing between subplots so they appear joined
    plt.subplots_adjust(wspace=0, hspace=0)

    # For reference, we define quadrant -> (row, col):
    #   cc -> (0,0), cw -> (0,1),
    #   wc -> (1,0), ww -> (1,1)

    # Plot each quadrant
    # Top-left = cc
    ax_cc = axes[0,0]
    im_cc = ax_cc.pcolormesh(cc_data, cmap='Greens', vmin=0.0, vmax=1.0,
                             edgecolors='gray', linewidth=0.1, shading='auto')
    ax_cc.set_box_aspect(1)

    # Top-right = cw
    ax_cw = axes[0,1]
    im_cw = ax_cw.pcolormesh(cw_data, cmap='Greens', vmin=0.0, vmax=1.0,
                             edgecolors='gray', linewidth=0.1, shading='auto')
    ax_cw.set_box_aspect(1)

    # Bottom-left = wc
    ax_wc = axes[1,0]
    im_wc = ax_wc.pcolormesh(wc_data, cmap='Greens', vmin=0.0, vmax=1.0,
                             edgecolors='gray', linewidth=0.1, shading='auto')
    ax_wc.set_box_aspect(1)

    # Bottom-right = ww
    ax_ww = axes[1,1]
    im_ww = ax_ww.pcolormesh(ww_data, cmap='Greens', vmin=0.0, vmax=1.0,
                             edgecolors='gray', linewidth=0.1, shading='auto')
    ax_ww.set_box_aspect(1)

    # Hide standard ticks/labels inside each axis
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # Put short quadrant titles *inside* each quadrant
    # Position them a bit away from edges so they don't overlap
    ax_cc.text(0.5, 0.5, "Both",
               ha='center', va='center', transform=ax_cc.transAxes,
               fontsize=17, fontweight='bold')
    ax_cw.text(0.5, 0.5, "Complementary",
               ha='center', va='center', transform=ax_cw.transAxes,
               fontsize=17, fontweight='bold')
    ax_wc.text(0.5, 0.5, "Elicitable",
               ha='center', va='center', transform=ax_wc.transAxes,
               fontsize=17, fontweight='bold')
    ax_ww.text(0.5, 0.5, "Neither",
               ha='center', va='center', transform=ax_ww.transAxes,
               fontsize=17, fontweight='bold')

    # We want "Correct"/"Wrong" on each axis, like a confusion matrix.
    # We'll place text near the midpoints of each outer axis.
    # Top row => y in [0..1], x in [0..1] for each subplot
    # For "Correct" on top-left x-axis, we place text below that axis
    # But let's do a simpler approach: place them on the edges with fig.text

    # Label row=0 with "Correct" for left subplot, "Wrong" for right subplot
    fig.text(0.32, 0.07, "Correct", ha='center', va='center', fontsize=18)  # bottom center of left subplots
    fig.text(0.64, 0.07, "Wrong",   ha='center', va='center', fontsize=18)  # bottom center of right subplots
    # Label row=1 with "Correct" for top subplot, "Wrong" for bottom subplot
    # But we need them on the left side, so we do:
    fig.text(0.05, 0.32, "Wrong",   ha='center', va='center', rotation='vertical', fontsize=18)
    fig.text(0.05, 0.64, "Correct", ha='center', va='center', rotation='vertical', fontsize=18)

    # Also label the entire figure's axes
    fig.text(0.5, 0.0, f"{mref2} Prediction", ha='center', va='center', fontsize=20)
    fig.text(0.0, 0.5, f"{mref1} Prediction", ha='center', va='center',
             rotation='vertical', fontsize=20)

    # Title at top
    fig.suptitle(
        f"{title_prefix}\n(Sub-rectangles for each of {W}×{S} model pairs)",
        fontsize=20, y=0.98
    )

    # Single colorbar to the right
    # We can pick one of the images (im_cc) or a list of them
    cbar = fig.colorbar(im_cc, ax=axes.ravel(), fraction=0.04, pad=0.07)
    cbar.set_label(f"{macc} Accuracy", fontsize=18)

    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {savepath}")
    else:
        plt.show()

    # --------------- Print per-pair and summary stats ---------------
    for qname in ['cc','cw','wc','ww']:
        print(f"Quadrant: {qname}")
        acc_list = quadrant_values[qname]
        idx = 0
        for w_model in weak_models:
            for s_model in strong_models:
                print(f"  Pair=({w_model}, {s_model}): {acc_list[idx]:.3f}")
                idx += 1
        arr = np.array(acc_list)
        mean_ = arr.mean() if len(arr) > 0 else 0.0
        std_ = arr.std(ddof=1) if len(arr) > 1 else 0.0
        n_ = len(arr)
        print(f"  => Quadrant {qname} summary: mean={mean_:.3f}, stdev={std_:.3f}, n={n_}\n")


def plot_diff_metric_from_values(metrics_per_pair, datasets_list, metric_name, savepath=None):
    pair_labels = list(metrics_per_pair.keys())
    x = np.arange(len(datasets_list))
    width = 0.25
    plt.style.use(['science', 'no-latex'])
    
    colors = ['skyblue', 'salmon', 'lightgreen']
    
    fig, ax = plt.subplots(figsize=(18, 10))
    for i, label in enumerate(pair_labels):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, metrics_per_pair[label], width, label=label, color=colors[i % len(colors)])
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + 0.005,
                    f'{height:.2f}', 
                    ha='center', va='bottom', fontsize=16,
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

    ax.set_xlabel('Datasets', fontsize=18)
    ax.set_ylabel(metric_name, fontsize=18)
    # ax.set_title(f'{metric_name} Between Model Pairs Across Datasets', fontsize=18, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets_list, rotation=45, ha='right', fontsize=12)
    ax.legend(title='Model Pairs', fontsize=12, title_fontsize=14)
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
    plt.tight_layout()
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, bbox_inches='tight')


def plot_multiple_triangles_from_values(jsd_values, datasets_list, m1name="weak_ft", m2name="strong_base", m3name="w2s", padding=0.5, savepath=None):
    fig, ax = plt.subplots(figsize=(15, 8))
    current_x = 0
    max_y = 0

    for dset in datasets_list:
        jsd_1_2, jsd_2_3, jsd_1_3 = jsd_values[dset]
        sides = [jsd_1_2, jsd_1_3, jsd_2_3]
        a, b, c = jsd_1_3, jsd_2_3, jsd_1_2

        if any(s <= 0 for s in sides):
            continue
        if not (a+b>c and b+c>a and a+c>b):
            continue

        p1 = np.array([current_x, 0])
        p2 = np.array([current_x + c, 0])
        x3 = (a**2 - b**2 + c**2) / (2*c)
        y3_squared = a**2 - x3**2
        if y3_squared < 0:
            continue
        y3 = np.sqrt(y3_squared)
        p3 = np.array([current_x + x3, y3])
        max_y = max(max_y, y3)
        triangle = np.array([p1, p2, p3, p1])
        ax.plot(triangle[:, 0], triangle[:, 1], 'k-', linewidth=2)
        ax.scatter([p1[0], p2[0], p3[0]],
                   [p1[1], p2[1], p3[1]],
                   color=['blue', 'blue', 'orange'], zorder=5, s=80)

        # Imaginary equilateral apex
        center_x_eq = (p1[0] + p2[0]) / 2
        h_eq = (np.sqrt(3) / 2) * c
        p3_eq = np.array([center_x_eq, h_eq])
        ax.plot([p1[0], p3_eq[0]], [p1[1], p3_eq[1]], 'k--', linewidth=1)
        ax.plot([p2[0], p3_eq[0]], [p2[1], p3_eq[1]], 'k--', linewidth=1)

        # Annotate sides
        ax.text((p1[0]+p2[0])/2, (p1[1]+p2[1])/2 + 0.1*max_y, f"{c:.2f}",
                ha='center', va='top', fontsize=10,
                bbox=dict(facecolor='white', edgecolor='none', pad=2))
        ax.text((p1[0]+p3[0])/2 - 0.5*c, (p1[1]+p3[1])/2 - 0.1*max_y, f"{a:.2f}",
                ha='left', va='bottom', fontsize=10, color='red',
                bbox=dict(facecolor='white', edgecolor='none', pad=2))
        ax.text((p2[0]+p3[0])/2 + 0.5*c, (p2[1]+p3[1])/2 - 0.1*max_y, f"{b:.2f}",
                ha='right', va='bottom', fontsize=10, color='blue',
                bbox=dict(facecolor='white', edgecolor='none', pad=2))

        # Model labels
        ax.text(p1[0], p1[1] - 0.12 * max_y, m1name, ha='center', va='top', fontsize=10, color='blue',
                bbox=dict(facecolor='white', edgecolor='none', pad=2))
        ax.text(p2[0], p2[1] - 0.12 * max_y, m2name, ha='center', va='top', fontsize=10, color='blue',
                bbox=dict(facecolor='white', edgecolor='none', pad=2))
        ax.text(p3[0], p3[1] + 0.5 * max_y, m3name, ha='center', va='top', fontsize=10, color='blue',
                bbox=dict(facecolor='white', edgecolor='none', pad=2))

        # Dataset label
        center_x = (p1[0] + p2[0]) / 2
        ax.text(center_x, -0.6 * max_y, dset, ha='center', va='top', fontsize=12,
                bbox=dict(facecolor='white', edgecolor='none', pad=2))

        current_x = p2[0] + padding

    ax.set_xlim(-padding, current_x)
    ax.set_ylim(-0.25 * max_y, max_y * 1.4)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title("JSD Triangles for Multiple Datasets", fontsize=16, pad=20)
    plt.tight_layout()
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, bbox_inches='tight')
        print(f"Saved JSD triangles plot to {savepath}")

def reorder_models_by_similarity(diff_matrix):
    # Symmetrize for clustering
    sym_matrix = (diff_matrix + diff_matrix.T) / 2.0
    linkage_matrix = linkage(sym_matrix, method='average')
    idx = leaves_list(linkage_matrix)
    return idx

def plot_diff_matrices_subplots(diff_matrices, model_names, metric_name, savepath=None):
    """
    Plots subplots of diff_matrices (one per dataset) in a 5 x 3 grid
    with consistent model ordering and larger fonts.

    diff_matrices: dict { dataset_name: NxN diff_matrix }
    model_names: list of models corresponding to the rows/cols of diff_matrix
    metric_name: for plot titles (e.g. "JSD", "Kappa", "Prediction Diff%")
    savepath: optional path to save the figure
    """
    # List of dataset names
    datasets_list = list(diff_matrices.keys())
    num_datasets = len(datasets_list)

    # If no data, just return
    if num_datasets == 0:
        print("No diff matrices to plot.")
        return

    # Font settings for larger text
    plt.rcParams.update({
        'font.size': 14,      # Base font size
        'axes.titlesize': 16, # Title font size
        'axes.labelsize': 16, # X/Y label font size
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })

    # Create a 5x3 grid of subplots (15 total)
    fig, axes = plt.subplots(5, 3, figsize=(20, 30))  # Adjust as needed
    axes = axes.flatten()  # Flatten so we can iterate easily

    def format_no_leading_zero(val):
        """Formats float to 2 decimals and strips leading zero if < 1."""
        return f"{val:.2f}".lstrip('0') if abs(val) < 1 else f"{val:.2f}"

    # Plot each dataset's diff_matrix in a separate subplot
    for i in range(15):
        if i < num_datasets:
            dset = datasets_list[i]
            diff_matrix = diff_matrices[dset]

            sns.heatmap(
                diff_matrix,
                xticklabels=model_names,
                yticklabels=model_names,
                cmap="YlOrRd",
                annot=np.array([
                    [format_no_leading_zero(val) for val in row] 
                    for row in diff_matrix
                ]),
                fmt="",       # Already providing strings for annotations
                square=True,
                cbar=False,
                ax=axes[i],
                annot_kws={"size": 14}  # Larger font for annotation
            )
            axes[i].set_title(dset, fontsize=16)
            axes[i].set_xlabel("Models", fontsize=16)
            axes[i].set_ylabel("Models", fontsize=16)
        else:
            # Hide unused subplots if there are fewer than 15 datasets
            axes[i].set_visible(False)

    # Add a main title for the whole figure
    plt.suptitle(f"{metric_name} Differences Across Datasets", fontsize=20, y=0.92)

    # Make layout tight so labels don't overlap
    plt.tight_layout()

    # Save or show
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
        plt.close()
        print(f"Saved multi-dataset {metric_name} heatmaps to {savepath}")
    else:
        plt.show()

#########################################
# Helper Functions Using Precomputed Values
#########################################


def compute_confusion_buckets_for_single_pair(preds, datasets_list, mref1, mref2, macc, split):
    """
    Computes confusion buckets for a single (mref1, mref2, macc) triple in 'preds'.

    Args:
        preds (dict): A single dictionary containing predictions, structured as:
            preds[dataset][model][split] = list of dicts, each dict has {"labels", "pred"}.
        datasets_list (list): List of dataset names.
        mref1 (str): Key for the first reference model.
        mref2 (str): Key for the second reference model.
        macc (str): Key for the main model.
        split (str): Data split to use (e.g. "test").

    Returns:
        dict: A dictionary keyed by dataset, each value is the 2x2 confusion buckets structure.
    """
    buckets_dict = {}
    for dset in datasets_list:
        dsref1 = preds[dset][mref1][split]  # list of {"labels", "pred"}
        dsref2 = preds[dset][mref2][split]
        dsmain = preds[dset][macc][split]

        # Compute confusion buckets using your acc_buckets function
        buckets_dict[dset] = acc_buckets(dsref1, dsref2, dsmain)

    return buckets_dict


def compute_confusion_buckets_for_average(preds_dict, datasets_list, mref1, mref2, macc, split, skip_list=None):
    """
    Aggregates confusion buckets across multiple (weak_model, strong_model) pairs.

    Args:
        preds_dict (dict): Dictionary keyed by (weak_model, strong_model), 
                           each storing a 'preds' structure:
                           preds_dict[(weak_model, strong_model)][dataset][model][split].
        datasets_list (list): List of dataset names.
        mref1 (str): Key for the first reference model (e.g., "weak_ft").
        mref2 (str): Key for the second reference model (e.g., "strong_base").
        macc (str): Key for the main model (e.g., "w2s").
        split (str): Data split to use (e.g. "test").
        skip_list (list): List of (weak_model, strong_model) pairs to exclude.

    Returns:
        dict: A dictionary keyed by dataset with aggregated confusion buckets.
    """
    if skip_list is None:
        skip_list = []

    # Initialize aggregated buckets (raw counts, not normalized)
    agg_buckets_dict = {
        dset: {
            'cc': {'correct': 0, 'total': 0},
            'cw': {'correct': 0, 'total': 0},
            'wc': {'correct': 0, 'total': 0},
            'ww': {'correct': 0, 'total': 0}
        }
        for dset in datasets_list
    }

    count_valid_pairs = 0

    # Iterate over all (weak_model, strong_model) pairs
    for (weak_model, strong_model), single_preds in preds_dict.items():
        if (weak_model, strong_model) in skip_list:
            continue

        for dset in datasets_list:
            dsref1 = single_preds[dset][mref1][split]  # list of {"labels", "pred"}
            dsref2 = single_preds[dset][mref2][split]
            dsmain = single_preds[dset][macc][split]

            # Compute confusion buckets for this pair/dataset
            buckets_pair = acc_buckets(dsref1, dsref2, dsmain)

            # Accumulate the raw counts in the aggregate
            for bucket_type, bucket_vals in buckets_pair.items():
                agg_buckets_dict[dset][bucket_type]['correct'] += bucket_vals['correct']
                agg_buckets_dict[dset][bucket_type]['total']   += bucket_vals['total']

        count_valid_pairs += 1

    if count_valid_pairs == 0:
        print("Warning: No valid model pairs found after applying skip_list.")

    return agg_buckets_dict

def compute_per_pair_quadrant_accuracies(
    preds_dict,
    datasets_list,
    weak_models,
    strong_models,
    mref1="weak_ft",
    mref2="strong_base",
    macc="w2s",
    split="test",
):
    """
    Computes w2s accuracy in each quadrant (cc, cw, wc, ww) for every (weak_model, strong_model) pair,
    aggregated across all datasets in datasets_list.

    Returns:
        quadrant_acc[(weak_model, strong_model)] = {
            'cc': (correct_count, total_count),
            'cw': (correct_count, total_count),
            'wc': (correct_count, total_count),
            'ww': (correct_count, total_count)
        }
    """

    quadrant_acc = {}

    for w_model in weak_models:
        for s_model in strong_models:
            # Initialize per-quadrant aggregator
            quadrant_acc[(w_model, s_model)] = {
                'cc': {'correct': 0, 'total': 0},
                'cw': {'correct': 0, 'total': 0},
                'wc': {'correct': 0, 'total': 0},
                'ww': {'correct': 0, 'total': 0}
            }

            # Go through each dataset and accumulate
            for dset in datasets_list:
                dsref1 = preds_dict[(w_model, s_model)][dset][mref1][split]  # list of {"labels", "pred"}
                dsref2 = preds_dict[(w_model, s_model)][dset][mref2][split]
                dsmain = preds_dict[(w_model, s_model)][dset][macc][split]

                # We assume acc_buckets returns something like:
                # {
                #   'cc': {'correct': X, 'total': Y},
                #   'cw': {'correct': X, 'total': Y},
                #   'wc': {'correct': X, 'total': Y},
                #   'ww': {'correct': X, 'total': Y}
                # }
                bckt = acc_buckets(dsref1, dsref2, dsmain)
                
                # Accumulate
                for q in ['cc','cw','wc','ww']:
                    quadrant_acc[(w_model, s_model)][q]['correct'] += bckt[q]['correct']
                    quadrant_acc[(w_model, s_model)][q]['total']   += bckt[q]['total']

    return quadrant_acc

def compute_diff_matrix_for_dataset_single_pair(preds, folder_path, dataset, model_names, split, diff_func, diff_func_name):
    """
    Compute the N x N difference matrix for a single dataset and single (weak_model, strong_model) scenario.
    """
    N = len(model_names)
    diff_matrix = np.zeros((N, N))
    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            # if i == j:
            #     continue
            val = precomputed_diffs(preds, folder_path, dataset, split, diff_func, diff_func_name, m1, m2)
            diff_matrix[i, j] = val
    return diff_matrix

def compute_diff_matrix_for_dataset_average(preds_dict, folder_name, dataset, model_names, split, diff_func, diff_func_name, skip_list):
    """
    Compute the NxN difference matrix for one dataset, averaged over all (weak_model, strong_model) pairs.
    """
    N = len(model_names)
    diff_matrix_sum = np.zeros((N, N))
    count = 0
    for (weak_model, strong_model), preds in preds_dict.items():
        if (weak_model, strong_model) in skip_list:
            continue
        pair_folder = f"{folder_name}/{weak_model}___{strong_model}"
        single_diff = compute_diff_matrix_for_dataset_single_pair(preds, pair_folder, dataset, model_names, split, diff_func, diff_func_name)
        diff_matrix_sum += single_diff
        count += 1
    if count == 0:
        return None
    return diff_matrix_sum / count

def compute_average_accuracies(preds_dict, folder_name, datasets, model_names, skip_list, split='test'):
    """
    Compute average accuracies using precomputed_accs.
    """
    acc_list = {m : [] for m in model_names}
    acc_sums = {m: {d:0 for d in datasets} for m in model_names}
    count = 0
    for (weak_model, strong_model), preds in preds_dict.items():
        if (weak_model, strong_model) in skip_list:
            continue
        # For each dataset, load accuracies from precomputed_accs
        for d in datasets:
            accs = precomputed_accs(preds, f"{folder_name}/{weak_model}___{strong_model}", d, model_names, split)
            for m in model_names:
                acc_sums[m][d] += accs[m]
                acc_list[m].append(accs[m])
        count += 1

    if count == 0:
        return {m: {d:0 for d in datasets} for m in model_names}

    for m in model_names:
        for d in datasets:
            acc_sums[m][d] /= count
        print(f"{m}: {np.mean(acc_list[m]):.4f} (n={len(acc_list[m])})")
    
    return acc_sums

def compute_average_metric_pairs(preds_dict, folder_name, datasets, diff_func, diff_func_name, skip_list, split='test', m1="weak_ft", m2="strong_base", m3="w2s"):
    """
    Compute average of w-s, s-w2s, w-w2s differences using precomputed_diffs.
    """
    pair_metrics_sums = {f"{m1}-{m2}": [0]*len(datasets), f"{m2}-{m3}": [0]*len(datasets), f"{m1}-{m3}": [0]*len(datasets)}
    count = 0
    for (weak_model, strong_model), preds in preds_dict.items():
        if (weak_model, strong_model) in skip_list:
            continue
        for i, d in enumerate(datasets):
            val_w_s = precomputed_diffs(preds, f"{folder_name}/{weak_model}___{strong_model}", d, split, diff_func, diff_func_name, m1, m2)
            val_s_w2s = precomputed_diffs(preds, f"{folder_name}/{weak_model}___{strong_model}", d, split, diff_func, diff_func_name, m2, m3)
            val_w_w2s = precomputed_diffs(preds, f"{folder_name}/{weak_model}___{strong_model}", d, split, diff_func, diff_func_name, m1, m3)
            pair_metrics_sums[f"{m1}-{m2}"][i] += val_w_s
            pair_metrics_sums[f"{m2}-{m3}"][i] += val_s_w2s
            pair_metrics_sums[f"{m1}-{m3}"][i] += val_w_w2s
        count += 1

    if count == 0:
        return {k:[0]*len(datasets) for k in pair_metrics_sums.keys()}
    
    for k in pair_metrics_sums.keys():
        pair_metrics_sums[k] = [val/count for val in pair_metrics_sums[k]]
    return pair_metrics_sums


def compute_average_jsd_triangles(preds_dict, folder_name, datasets, skip_list, split='test', m1="weak_ft", m2="strong_base", m3="w2s"):
    """
    Compute average JSD values for the triangle plot using precomputed_diffs.
    """
    sums = {d: [0,0,0] for d in datasets}
    count = 0
    for (weak_model, strong_model), preds in preds_dict.items():
        if (weak_model, strong_model) in skip_list:
            continue
        for d in datasets:
            jsd_w_s = precomputed_diffs(preds, f"{folder_name}/{weak_model}___{strong_model}", d, split, get_jsd, "JSD", m1, m2)
            jsd_s_w2s = precomputed_diffs(preds, f"{folder_name}/{weak_model}___{strong_model}", d, split, get_jsd, "JSD", m2, m3)
            jsd_w_w2s = precomputed_diffs(preds, f"{folder_name}/{weak_model}___{strong_model}", d, split, get_jsd, "JSD", m1, m3)
            sums[d][0] += jsd_w_s
            sums[d][1] += jsd_s_w2s
            sums[d][2] += jsd_w_w2s
        count += 1

    if count == 0:
        return {d:(0,0,0) for d in datasets}

    for d in datasets:
        sums[d] = (sums[d][0]/count, sums[d][1]/count, sums[d][2]/count)
    return sums


def get_preds_all_modelpairs(folder_name, model_names, datasplits=["test"], datasets=None, skip_list=[]):
    preds_dict = {}
    for dir in os.listdir(f"results/{folder_name}"):
        dir_path = f"results/{folder_name}/{dir}"
        if os.path.isdir(dir_path) and "___" in dir:
            weak_model, strong_model = dir.split("___")
            if (weak_model, strong_model) in skip_list:
                continue
            if datasets is None:
                datasets = os.listdir(dir_path)
                datasets = [d for d in datasets if d != "plots"]
            preds = {}
            preds = populate_preds(preds, datasets, model_names, datasplits, folder_name, weak_model, strong_model)
            preds_dict[(weak_model, strong_model)] = preds
    return preds_dict


#########################################
# Main Functionality
#########################################

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "average"  # default "average"
    
    folder_name = "epochs_3"
    # model_names = ["weak_ft", "weak2", "strong_base", "strong_base2", "mean", "w2s", "w2snaive", "strong_ft", "strong2", "diff_ceil"]
    model_names = ["strong_base2", "weak_ft", "w2s", "strong2", "diff_ceil"]
    m1_name, m2_name, m3_name = "weak_ft", "strong2", "w2s"
    mtriplet_name = "strft2"
    # If you only want a subset, uncomment below:
    # model_names = ["weak_ft", "strong_base", "w2s", "strong_ft"]
    split = "train"
    datasplits = [split]
    datasets = ["anli-r2", "boolq", "cola", "ethics-utilitarianism", "sciq", "sst2", "twitter-sentiment", "dream", "mc_taco", "multirc", "quail", "quartz", "social_i_qa", "wic", "cosmos_qa"]
    skip_list = [("gemma-2-9b", "Llama-3.1-8B"), ("Llama-3.1-8B", "gemma-2-9b"), ("Llama-3.1-8B", "Llama-3.1-8B"), ("Qwen2.5-0.5B", "OLMo-2-1124-7B"), ("Qwen2.5-0.5B", "Qwen2.5-14B"), ("Qwen2.5-0.5B", "gemma-2-27b"), ("Qwen2.5-0.5B", "granite-3.0-8b-base"), ("gemma-2-2b", "OLMo-2-1124-7B"), ("SmolLM-1.7B", "gemma-2-9b"), ("SmolLM-1.7B", "Qwen2.5-7B"), ("SmolLM-1.7B", "Llama-3.1-8B")]
    average_across_datasets = False
    weak_models = ["gemma-2-2b", "Qwen2.5-1.5B", "Llama-3.2-3B", "phi-2"]
    strong_models = ["gemma-2-9b", "Qwen2.5-7B", "Llama-3.1-8B"]
    preds_dict = get_preds_all_modelpairs(folder_name, model_names, datasplits=datasplits, datasets=datasets, skip_list=skip_list)

    if mode == "average":
        os.makedirs(f"results/{folder_name}/plots/{mtriplet_name}", exist_ok=True)

        avg_acc = compute_average_accuracies(preds_dict, folder_name, datasets, model_names, skip_list, split=split)
        grouped_bar_plot_accs(avg_acc, split=split, metric_name="Accuracy", savepath=f"results/{folder_name}/grouped_bar_plot_accs_{split}.pdf", datasets_list=datasets, average_across_datasets=average_across_datasets)

        quadrant_acc = compute_per_pair_quadrant_accuracies(
            preds_dict,
            datasets_list=datasets, 
            weak_models=weak_models,
            strong_models=strong_models,
            mref1=m1_name,
            mref2=m2_name,
            macc=m3_name,
            split="test"
        )
        
        plot_quadrants_with_subrects(
            quadrant_acc,
            weak_models=weak_models,
            strong_models=strong_models,
            mref1="Weak Supervisor",
            mref2="Strong Elicited",
            macc="Weak to Strong Trained Model",
            title_prefix="Disentangling Accuracy of Weak to Strong Trained Model",
            savepath=f"results/{folder_name}/plots/{mtriplet_name}/new_conf_mat_accs_{split}.png",
        )


        # Compute averages across model pairs using precomputed values

        averaged_buckets = compute_confusion_buckets_for_average(
            preds_dict,
            datasets,
            mref1=m1_name,
            mref2=m2_name,
            macc=m3_name,
            split=split,
            skip_list=skip_list
        )

        # Plot averaged confusion matrices
        plot_conf_mat_from_buckets(
            averaged_buckets,
            datasets,
            mref1=m1_name,
            mref2=m2_name,
            macc=m3_name,
            savepath=f"results/{folder_name}/plots/{mtriplet_name}/conf_mat_accs_{split}.png",
            average_across_datasets=average_across_datasets
        )

        JSD
        avg_jsd_pairs = compute_average_metric_pairs(preds_dict, folder_name, datasets, get_jsd, "JSD", skip_list, split=split, m1=m1_name, m2=m2_name, m3=m3_name)
        plot_diff_metric_from_values(avg_jsd_pairs, datasets, "JSD", savepath=f"results/{folder_name}/plots/{mtriplet_name}/averaged_jsd_{split}.pdf")

        avg_kappa_pairs = compute_average_metric_pairs(preds_dict, folder_name, datasets, get_probkappa_mcqs, "ProbKappa", skip_list, split=split, m1=m1_name, m2=m2_name, m3=m3_name)
        plot_diff_metric_from_values(avg_kappa_pairs, datasets, "ProbKappa", savepath=f"results/{folder_name}/plots/{mtriplet_name}/averaged_probkappa_{split}.pdf")

        Kappa
        avg_kappa_pairs = compute_average_metric_pairs(preds_dict, folder_name, datasets, get_kappa_mcqs, "Kappa", skip_list, split=split, m1=m1_name, m2=m2_name, m3=m3_name)
        plot_diff_metric_from_values(avg_kappa_pairs, datasets, "Kappa", savepath=f"results/{folder_name}/plots/{mtriplet_name}/averaged_kappa_{split}.png")

        Prediction Diff%
        avg_diffp_pairs = compute_average_metric_pairs(preds_dict, folder_name, datasets, get_diffp, "Prediction Diff%", skip_list, split=split, m1=m1_name, m2=m2_name, m3=m3_name)
        plot_diff_metric_from_values(avg_diffp_pairs, datasets, "Prediction Diff%", savepath=f"results/{folder_name}/plots/{mtriplet_name}/averaged_diffp_{split}.png")

        JSD Triangles
        avg_jsd_triangles = compute_average_jsd_triangles(preds_dict, folder_name, datasets, skip_list, split=split, m1=m1_name, m2=m2_name, m3=m3_name)
        plot_multiple_triangles_from_values(avg_jsd_triangles, datasets, padding=0.3, savepath=f"results/{folder_name}/plots/{mtriplet_name}/averaged_jsdtriangles_{split}.pdf", m1name=m1_name, m2name=m2_name, m3name=m3_name)

        for diff_func, diff_func_name in [(get_probkappa_mcqs, "Similarity")]:
            diff_matrices = {}
            for d in datasets:
                mat = compute_diff_matrix_for_dataset_average(preds_dict, folder_name, d, model_names, split, diff_func, diff_func_name, skip_list)
                if mat is not None:
                    diff_matrices[d] = mat
            
            if diff_matrices:
                heatmap_savepath = f"results/{folder_name}/plots/{mtriplet_name}/diffmatrix_averaged_{diff_func_name.lower().replace(' ','-')}_{split}.pdf"
                plot_diff_matrices_subplots(diff_matrices, model_names, diff_func_name, savepath=heatmap_savepath)
            else:
                print(f"No data to plot for {diff_func_name} in average mode.")

    else:
        # mode == "all": Plot separately for each model pair as originally implemented, but using precomputed values now
        for (weak_model, strong_model), preds in preds_dict.items():
            if (weak_model, strong_model) in skip_list:
                continue
            pair_folder = f"{folder_name}/{weak_model}___{strong_model}"
            savepath = f"results/{pair_folder}/plots/{mtriplet_name}"
            os.makedirs(savepath, exist_ok=True)

            for split in datasplits:
                # Compute accuracies from precomputed
                accs = {}
                for d in datasets:
                    acc_dict = precomputed_accs(preds, pair_folder, d, model_names, split)
                    for m in acc_dict:
                        if m not in accs:
                            accs[m] = []
                        accs[m].append(acc_dict[m])

                grouped_bar_plot_accs(accs, split=split, metric_name="acc", savepath=f"{savepath}/grouped_bar_plot_accs_{split}.png", datasets_list=datasets)
                # conf_mat_accs(preds, split=split, savepath=f"{savepath}/conf_mat_accs_{split}.png", datasets_list=datasets)

                single_buckets = compute_confusion_buckets_for_single_pair(
                    preds,
                    datasets,
                    mref1=m1_name,
                    mref2=m2_name,
                    macc=m3_name,
                    split=split
                )

                # Plot confusion matrices for this pair
                plot_conf_mat_from_buckets(
                    single_buckets,
                    datasets,
                    mref1=m1_name,
                    mref2=m2_name,
                    macc=m3_name,
                    savepath=f"{savepath}/conf_mat_accs_{split}.png"
                )

                def plot_single_pair_diff_metric_from_precomputed(diff_func, diff_func_name, metric_name, savepath, m1_name="weak_ft", m2_name="strong_base", m3_name="w2s"):
                    pair_metrics = {f"{m1_name}-{m2_name}":[], f"{m2_name}-{m3_name}":[], f"{m1_name}-{m3_name}":[]}
                    for d in datasets:
                        val_w_s = precomputed_diffs(preds, pair_folder, d, split, diff_func, diff_func_name, m1_name, m2_name)
                        val_s_w2s = precomputed_diffs(preds, pair_folder, d, split, diff_func, diff_func_name, m2_name, m3_name)
                        val_w_w2s = precomputed_diffs(preds, pair_folder, d, split, diff_func, diff_func_name, m1_name, m3_name)
                        pair_metrics[f"{m1_name}-{m2_name}"].append(val_w_s)
                        pair_metrics[f"{m2_name}-{m3_name}"].append(val_s_w2s)
                        pair_metrics[f"{m1_name}-{m3_name}"].append(val_w_w2s)
                    plot_diff_metric_from_values(pair_metrics, datasets, metric_name, savepath=savepath)

                # JSD
                plot_single_pair_diff_metric_from_precomputed(get_jsd, "JSD", "JSD", f"{savepath}/jsd_{split}.png", m1_name=m1_name, m2_name=m2_name, m3_name=m3_name)
                # Kappa
                plot_single_pair_diff_metric_from_precomputed(get_kappa_mcqs, "Kappa", "Kappa", f"{savepath}/kappa_{split}.png", m1_name=m1_name, m2_name=m2_name, m3_name=m3_name)
                # Prediction Diff%
                plot_single_pair_diff_metric_from_precomputed(get_diffp, "Prediction Diff%", "Prediction Diff%", f"{savepath}/diffp_{split}.png", m1_name=m1_name, m2_name=m2_name, m3_name=m3_name)

                # Triangles for single pair using precomputed diffs for JSD
                jsd_values = {}
                for d in datasets:
                    jsd_w_s = precomputed_diffs(preds, pair_folder, d, split, get_jsd, "JSD", m1_name, m2_name)
                    jsd_s_w2s = precomputed_diffs(preds, pair_folder, d, split, get_jsd, "JSD", m2_name, m3_name)
                    jsd_w_w2s = precomputed_diffs(preds, pair_folder, d, split, get_jsd, "JSD", m1_name, m3_name)
                    jsd_values[d] = (jsd_w_s, jsd_s_w2s, jsd_w_w2s)

                plot_multiple_triangles_from_values(jsd_values, datasets, padding=0.3, savepath=f"{savepath}/jsdtriangles_{split}.png", m1name=m1_name, m2name=m2_name, m3name=m3_name)

                for diff_func, diff_func_name in [(get_jsd, "JSD"), (get_kappa_mcqs, "Kappa"), (get_diffp, "Prediction Diff%")]:
                    diff_matrices = {}
                    for d in datasets:
                        diff_matrix = compute_diff_matrix_for_dataset_single_pair(preds, pair_folder, d, model_names, split, diff_func, diff_func_name)
                        diff_matrices[d] = diff_matrix
                    heatmap_savepath = f"{savepath}/diffmatrix_{diff_func_name.lower().replace(' ','_')}_{split}.png"
                    plot_diff_matrices_subplots(diff_matrices, model_names, diff_func_name, savepath=heatmap_savepath)
