import os
from datasets import Dataset
from regex import F
from scipy.spatial.distance import jensenshannon as jsd
import numpy as np
from w2s.roc_auc import roc_auc
import torch
import json
import pyarrow as pa
import pandas as pd



def get_logits(ds):
    return np.array([1-np.array(ds["soft_pred"]), np.array(ds["soft_pred"])]).T

def get_brier(ds):
    labels = np.array([d["labels"] for d in ds])
    preds = np.array([d["soft_pred"] for d in ds])
    return np.mean((labels - preds)**2)

def get_jsd(ds1, ds2):
    return np.mean(jsd(get_logits(ds1), get_logits(ds2)))

def get_kappa_mcqs(m1, m2, n_options=2):
    same, total, m1_corr, m2_corr, samewrong = 0, 0, 0, 0, 0
    for s1, s2 in zip(m1, m2):
        total += 1
        if s1["pred"] == s2["pred"]:
            same += 1
        if s1["pred"] == s1["labels"]:
            m1_corr += 1
        if s2["pred"] == s2["labels"]:
            m2_corr += 1
        if s1["pred"] == s2["pred"] and s1["pred"] != s1["labels"]:
            samewrong += 1
    cobs = same/total
    p1, p2 = (m1_corr/total), (m2_corr/total)
    expsamewrong = (1 - p1) * (1 - p2) * 1/(n_options - 1)
    cexp = p1 * p2 + expsamewrong
    kappa = (cobs - cexp) / (1 - cexp)
    return kappa

def get_maxagreement(a1, a2):
    tmp1, tmp2 = a1, a2
    if a2 > a1:
        tmp1, tmp2 = a2, a1
    return tmp2 + 1-tmp1

def get_minagreement(a1, a2):
    return max(0, a1 + a2 - 1)

def get_adjkappa_mcqs(m1, m2, n_options=2):
    same, total, m1_corr, m2_corr, samewrong = 0, 0, 0, 0, 0
    for s1, s2 in zip(m1, m2):
        total += 1
        if s1["pred"] == s2["pred"]:
            same += 1
        if s1["pred"] == s1["labels"]:
            m1_corr += 1
        if s2["pred"] == s2["labels"]:
            m2_corr += 1
        if s1["pred"] == s2["pred"] and s1["pred"] != s1["labels"]:
            samewrong += 1
    cobs = same/total
    p1, p2 = (m1_corr/total), (m2_corr/total)
    expsamewrong = (1 - p1) * (1 - p2) * 1/(n_options - 1)
    cexp = p1 * p2 + expsamewrong
    if cobs > cexp:
        cobs_max = get_maxagreement(p1, p2)
        kappa = (cobs - cexp) / (cobs_max - cexp)
    else:
        cobs_min = get_minagreement(p1, p2)
        kappa = (cobs - cexp) / (cexp - cobs_min)
    return kappa

def get_probkappa_mcqs(m1, m2, n_options=2):
    cobs, total = 0, 0
    p1corr, p2corr = 0, 0
    
    for s1, s2 in zip(m1, m2):
        total += 1
        cobs += s1["soft_pred"] * s2["soft_pred"] + (1 - s1["soft_pred"]) * (1 - s2["soft_pred"])
        p1corr += s1["soft_pred"] if s1["labels"] == 1 else 1 - s1["soft_pred"]
        p2corr += s2["soft_pred"] if s2["labels"] == 1 else 1 - s2["soft_pred"]

    cobs/=total
    p1corr/=total
    p2corr/=total
    cexp = p1corr * p2corr + (1 - p1corr) * (1 - p2corr)
    kappa = (cobs - cexp) / (1 - cexp)
    return kappa

def get_diffp(ds1, ds2):
    #s1[pred] != s2['pred']
    diffp = 0
    for s1, s2 in zip(ds1, ds2):
        if s1["pred"] != s2["pred"]:
            diffp += 1
    return diffp/len(ds1)

def get_acc(ds):
    labels = np.array([d["labels"] for d in ds])
    preds = np.array([d["soft_pred"] for d in ds])
    return np.mean(labels == (preds > 0.5))*100

def compute_mean_preds(preds, dname, split):
    weak_split = preds[dname]["weak_ft"][split]
    strong_split = preds[dname]["strong_base2"][split]

    # Convert columns to numpy arrays for vectorized operations
    weak_soft = np.array(weak_split["soft_pred"], dtype=np.float32)
    strong_soft = np.array(strong_split["soft_pred"], dtype=np.float32)

    # Vectorized average soft predictions
    avg_soft_preds = (weak_soft + strong_soft) / 2.0
    # Vectorized binary predictions
    avg_preds = avg_soft_preds > 0.5

    # Create mean_split dataset
    mean_split = weak_split.remove_columns(["soft_pred", "pred"])
    mean_split = mean_split.add_column("soft_pred", avg_soft_preds.tolist())
    mean_split = mean_split.add_column("pred", avg_preds.tolist())

    preds[dname]["mean"] = {split: mean_split}
    return preds[dname]["mean"]


def compute_diff_ceil(preds, dname, split):
    weak_split = preds[dname]["weak_ft"][split]
    strong_split = preds[dname]["strong_base2"][split]

    # Convert columns to numpy arrays
    weak_pred = np.array(weak_split["pred"], dtype=np.bool_)
    weak_soft = np.array(weak_split["soft_pred"], dtype=np.float32)
    strong_pred = np.array(strong_split["pred"], dtype=np.bool_)
    strong_soft = np.array(strong_split["soft_pred"], dtype=np.float32)
    labels = np.array(weak_split["labels"])

    # Construct mask where strong predictions match the labels
    mask = (strong_pred == labels)

    # Vectorized selection of oracle predictions and soft predictions
    oracle_preds = np.where(mask, strong_pred, weak_pred)
    oracle_soft_preds = np.where(mask, strong_soft, weak_soft)

    # Create diff_ceil_split dataset
    diff_ceil_split = weak_split.remove_columns(["soft_pred", "pred"])
    diff_ceil_split = diff_ceil_split.add_column("soft_pred", oracle_soft_preds.tolist())
    diff_ceil_split = diff_ceil_split.add_column("pred", oracle_preds.tolist())

    preds[dname]["diff_ceil"] = {split: diff_ceil_split}
    return preds[dname]["diff_ceil"]

def populate_preds(preds, datasets, model_names, datasplits, folder_name, weak_model, strong_model, verbose=False):
    if not verbose:
        from datasets.utils.logging import set_verbosity_error
        set_verbosity_error()
    for dname in datasets:
        if verbose: print(f"Dataset: {dname}")
        preds[dname] = {}
        for mname in model_names:
            if verbose: print(f"Model: {mname}")
            mfilename = f"{weak_model}___{strong_model}"
            preds[dname][mname] = {}
            mtype = mname
            if mname.endswith("ft"):
                mtype = mname[:-3]
            
            if "weak" in mname:
                mfilename = weak_model
            elif "strong" in mname:
                mfilename = strong_model                                    
            for split in datasplits:
                preds_path = f"results/{folder_name}/{mfilename}/{dname}/{mtype}/predictions/{split}/data-00000-of-00001.arrow"
                
                if mname == "mean" and not os.path.exists(preds_path):
                    compute_mean_preds(preds, dname, split)
                    continue
                elif mname == "diff_ceil":
                    compute_diff_ceil(preds, dname, split)
                    continue
                try:
                    preds[dname][mname][split] = add_preds(Dataset.from_file(preds_path))
                    if verbose: print(f"Loaded {len(preds[dname][mname][split])} examples for {split} split")
                except:
                    print(f"Error loading {dname} predictions for mtype {mname} in model pair ({weak_model}, {strong_model})")
    return preds


def precomputed_accs(preds, foldername, dname, mnames, split):
    """
    Computes or loads cached accuracies for all requested models `mnames` on a given dataset `dname` and `split`.

    The results are stored in:
       results/{foldername}/{dname}/{split}/accs.json

    Format of accs.json:
    {
       "weak_ft": 0.85,
       "strong_base": 0.90,
       "w2s": 0.88,
       ...
    }

    If a model's accuracy is not present, compute it, add it to the dict, and rewrite the file.
    """
    base_path = f"results/{foldername}/{dname}/{split}"
    os.makedirs(base_path, exist_ok=True)
    acc_path = os.path.join(base_path, "accs.json")

    # Load existing accuracies if available
    if os.path.exists(acc_path):
        with open(acc_path, "r") as f:
            accs = json.load(f)
    else:
        accs = {}

    # Compute missing accuracies
    updated = False
    for mname in mnames:
        if mname not in accs:
            acc_val = get_acc(preds[dname][mname][split])
            accs[mname] = acc_val
            updated = True

    # Save if updated
    if updated:
        with open(acc_path, "w") as f:
            json.dump(accs, f, indent=2)

    return accs

def precomputed_diffs(preds, foldername, dname, split, diff_func, diff_func_name, m1, m2):
    """
    Computes or loads cached differences for a given metric function and a pair of models (m1, m2).
    Results are stored in:
       results/{foldername}/{dname}/{split}/diffs.json

    Format of diffs.json:
    {
       "JSD_weak_ft_strong_base": 0.05,
       "Kappa_strong_base_w2s": 0.10,
       ...
    }

    The key format is: f"{diff_func_name}_{m1}_{m2}"
    If this key doesn't exist, compute, store, and return it.
    If it does, just return the cached value.
    """
    base_path = f"results/{foldername}/{dname}/{split}"
    os.makedirs(base_path, exist_ok=True)
    diff_path = os.path.join(base_path, "diffs.json")

    # Load existing diffs if available
    if os.path.exists(diff_path):
        with open(diff_path, "r") as f:
            diffs = json.load(f)
    else:
        diffs = {}

    diff_key = f"{diff_func_name}_{m1}_{m2}"
    if diff_key not in diffs:
        # Compute the difference metric for these two models
        vals_m1 = preds[dname][m1][split]
        vals_m2 = preds[dname][m2][split]
        diff_val = diff_func(vals_m1, vals_m2)
        diffs[diff_key] = diff_val

        # Save updated diffs.json
        with open(diff_path, "w") as f:
            json.dump(diffs, f, indent=2)
    else:
        diff_val = diffs[diff_key]

    return diff_val

def get_auc(ds):
    labels = torch.from_numpy(np.array([d["labels"] for d in ds])).float()
    preds = torch.from_numpy(np.array([d["soft_pred"] for d in ds])).float()
    return roc_auc(labels, preds)

def add_preds(ds):
    # Initialize predictions as a list
    ds = ds.map(lambda x: {'pred': x['soft_pred'] > 0.5})
    return ds

def acc_buckets(dsref1, dsref2, dsmain, verbose=False):
    # Convert lists of dictionaries to Pandas DataFrames
    df1 = pd.DataFrame(dsref1)
    df2 = pd.DataFrame(dsref2)
    df_main = pd.DataFrame(dsmain)
    
    # Compute correctness for each dataset
    df1['correct'] = df1['labels'] == df1['pred']
    df2['correct'] = df2['labels'] == df2['pred']
    df_main['correct'] = df_main['labels'] == df_main['pred']
    
    # Assign bucket labels based on correctness of df1 and df2
    # 'c' for correct, 'w' for wrong
    bucket_labels = (
        df1['correct'].map({True: 'c', False: 'w'}) +
        df2['correct'].map({True: 'c', False: 'w'})
    )
    
    # Add bucket labels to the main DataFrame
    df_main['bucket'] = bucket_labels
    
    # Initialize buckets
    buckets = {
        "cc": {"correct": 0, "total": 0},
        "cw": {"correct": 0, "total": 0},
        "wc": {"correct": 0, "total": 0},
        "ww": {"correct": 0, "total": 0}
    }
    
    # Group by bucket and calculate total and correct counts
    grouped = df_main.groupby('bucket')['correct']
    
    # Update buckets with aggregated counts
    for bucket in ['cc', 'cw', 'wc', 'ww']:
        if bucket in grouped.groups:
            total = grouped.size()[bucket]
            correct = grouped.sum()[bucket]
            buckets[bucket]['total'] = total
            buckets[bucket]['correct'] = correct
        else:
            # If a bucket has no entries, it remains zero
            pass
    
    # Print the final bucket counts
    if verbose: print(buckets)
    return buckets

def pretty_print(dsref1, dsref2, dsmain):
    dsref1, dsref2, dsmain = add_preds(dsref1), add_preds(dsref2), add_preds(dsmain)
    buckets = acc_buckets(dsref1, dsref2, dsmain)
    for cat in buckets:
        print(f"Category {cat}: Accuracy - {buckets[cat]['correct']/buckets[cat]['total']*100:.2f}%, Total - {buckets[cat]['total']}")

    print(f"JSD between {m1name} and {m2name}: {get_jsd(dsref1, dsref2):.4f}")
    print(f"JSD between {m1name} and {m3name}: {get_jsd(dsref1, dsmain):.4f}")
    print(f"JSD between {m2name} and {m3name}: {get_jsd(dsref2, dsmain):.4f}")

    print(f"Kappa between {m1name} and {m2name}: {get_kappa_mcqs(dsref1, dsref2, n_options=2):.4f}")
    print(f"Kappa between {m1name} and {m3name}: {get_kappa_mcqs(dsref1, dsmain, n_options=2):.4f}")
    print(f"Kappa between {m2name} and {m3name}: {get_kappa_mcqs(dsref2, dsmain, n_options=2):.4f}")

if __name__ == '__main__':
    datasets = ['anli-r2', 'boolq', 'cola', 'ethics-utilitarianism', 'hellaswag', 'sciq', 'piqa', 'sst2', 'twitter-sentiment']
    shared_folder_name = 'shared3'
    test_folder_name = 'test3'
    for dset in datasets:
        m1name, m2name, m3name, m4name = "weak", "strong_base", "w2s", "strong"
        print("Dataset:", dset)
        dsref1 = Dataset.from_file(f"results/{shared_folder_name}/{dset}/{m1name}/predictions/test/data-00000-of-00001.arrow")
        print(f"Loaded {len(dsref1)} examples for {m1name}, accuracy: {get_acc(dsref1):.2f}%, auc: {get_auc(dsref1):.2f}")
        dsref2 = Dataset.from_file(f"results/{shared_folder_name}/{dset}/{m2name}/predictions/test/data-00000-of-00001.arrow")
        print(f"Loaded {len(dsref2)} examples for {m2name}, accuracy: {get_acc(dsref2):.2f}%, auc: {get_auc(dsref2):.2f}")
        dsmain = Dataset.from_file(f"results/{test_folder_name}/{dset}/{m3name}/predictions/test/data-00000-of-00001.arrow")
        print(f"Loaded {len(dsmain)} examples for {m3name}, accuracy: {get_acc(dsmain):.2f}%, auc: {get_auc(dsmain):.2f}")
        dsref3 = Dataset.from_file(f"results/{shared_folder_name}/{dset}/{m4name}/predictions/test/data-00000-of-00001.arrow")
        print(f"Loaded {len(dsref1)} examples for {m4name}, accuracy: {get_acc(dsref3):.2f}%, auc: {get_auc(dsref3):.2f}")
        
        pretty_print(dsref1, dsref2, dsmain)
    

