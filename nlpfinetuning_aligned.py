"""Yelp Review Full fine-tuning aligned with the PCam protocol of the paper: the BERT and XLM-RoBERTa runs of
"Crossing the Validation Crisis" (Eve, Varoquaux and Moreau, 2026).

Differences from nlpfinetuning_full_adapted.py, which ran the Qwen3-0.6B runs of the paper (and earlier BERT and
XLM-RoBERTa runs that the paper no longer uses) and is kept unchanged:

* one fixed benchmarking set, the same for every seed and model: 600,000 of the 650,000 training reviews
  (stratified split, random_state 42). It is scored IN FULL after every split, length-sorted and padded per batch,
  which gives the same predictions with far fewer padded tokens than padding every review to 512 tokens;
* the remaining 50,000 reviews are the pool of each seed's study set (stratified draw) and of its outer set;
* --train-sizes gives n_tr, the training and validation subsets together, as in the paper; the study set has
  n_tr / (1 - test fraction) reviews (1,250 for n_tr = 1,000);
* the predictor of the first split also scores the outer set, in chunks of the test-fold size (at most 200), so the
  single-split side of the sample gain comes from the same runs as the K-split side (paired, as for PCam);
* per-sample losses on the test folds (0-1 error, negative log-likelihood, multiclass Brier score) give the
  study-only redundancy statistics of the PCam script, cumulative after each split;
* accuracy, NLL and Brier score are recorded for every evaluated set.

Two steps:
    python nlpfinetuning_aligned.py prepare --model-choices roberta          # login node, CPU: tokenize once
    python nlpfinetuning_aligned.py run --model-choices roberta --train-sizes 1000 --n-splits 20 --seeds 0 1
The prepare step writes <--data-dir>/<model>/ (tokenized training split, fixed split, benchmark order; --data-dir
defaults to $SCRATCH/yelp_aligned, or ./yelp_aligned without $SCRATCH); run jobs read it and never tokenize the
600,000 benchmark reviews again. Each run writes one Parquet file per (model, n_tr) to --out-dir.
"""
import argparse
import gc
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, train_test_split

os.environ["TOKENIZERS_PARALLELISM"] = "false"

NUM_TRAIN_EPOCHS = 3
BENCH_SIZE = 600_000
SPLIT_SEED = 42
MAX_LENGTH = 512
MODEL_CONFIGS = {
    "bert":       {"hf_name": "google-bert/bert-base-cased", "is_decoder": False, "per_device_train_batch_size": 32},
    "roberta":    {"hf_name": "FacebookAI/xlm-roberta-base", "is_decoder": False, "per_device_train_batch_size": 32},
    "qwen3-0.6b": {"hf_name": "Qwen/Qwen3-0.6B", "is_decoder": True, "per_device_train_batch_size": 32},
}


def parse_arguments():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prepare", "run"])
    ap.add_argument("--model-choices", nargs="+", default=["roberta"], choices=list(MODEL_CONFIGS))
    ap.add_argument("--train-sizes", type=int, nargs="+", default=[1000], help="n_tr: training + validation subsets")
    ap.add_argument("--n-splits", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 1], metavar=("START", "END_EXCLUSIVE"))
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--max-outer-chunks", type=int, default=200)
    ap.add_argument("--eval-batch-size", type=int, default=512)
    ap.add_argument("--data-dir", default=os.path.join(os.environ.get("SCRATCH", "."), "yelp_aligned"))
    ap.add_argument("--out-dir", default="results_aligned")
    ap.add_argument("--check-padding", type=int, default=0,
                    help="smoke test: also score the first N benchmark reviews padded to the longest of each block "
                         "of 1,000 (the earlier protocol) and compare predictions and time with the sorted scoring")
    return ap.parse_args()


# --------------------------------------------------------------------------------------------------------------
# Redundancy statistics: copied from benchmark_pcam/pcam_deep_training_adapted.py so that both datasets use the
# same estimator (pairwise OOF-intersection loss correlation, and the per-sample OOF ANOVA).
# --------------------------------------------------------------------------------------------------------------
def _pairwise_oof_icc(loss_by_fold, ids_by_fold):
    id_to_loss = [dict(zip(np.asarray(ids).tolist(), np.asarray(loss)))
                  for ids, loss in zip(ids_by_fold, loss_by_fold)]
    covs, vars_, sizes = [], [], []
    n_folds = len(ids_by_fold)
    for k in range(n_folds):
        for ell in range(k + 1, n_folds):
            shared = np.intersect1d(np.asarray(ids_by_fold[k]), np.asarray(ids_by_fold[ell]), assume_unique=True)
            if shared.size < 2:
                continue
            qk = np.array([id_to_loss[k][int(i)] for i in shared])
            ql = np.array([id_to_loss[ell][int(i)] for i in shared])
            pair_cov = np.cov(qk, ql)
            cov = float(pair_cov[0, 1])
            var = float(0.5 * (pair_cov[0, 0] + pair_cov[1, 1]))
            if not (np.isfinite(cov) and np.isfinite(var)):
                continue
            covs.append(cov)
            vars_.append(var)
            sizes.append(int(shared.size))
    if not covs:
        return np.nan, 0, np.nan
    var_mean = float(np.mean(vars_))
    rho = float(np.mean(covs) / var_mean) if var_mean > 0 else 0.0
    return rho, int(len(covs)), float(np.mean(sizes))


def _oof_anova(loss_by_fold, ids_by_fold):
    sums, sumsq, count = {}, {}, {}
    for ids, loss in zip(ids_by_fold, loss_by_fold):
        for i, v in zip(np.asarray(ids).tolist(), np.asarray(loss, dtype=float)):
            sums[i] = sums.get(i, 0.0) + v
            sumsq[i] = sumsq.get(i, 0.0) + v * v
            count[i] = count.get(i, 0) + 1
    if not count:
        return np.nan, np.nan, np.nan, 0, np.nan
    c_all = np.array(list(count.values()), dtype=float)
    keys = [k for k, n in count.items() if n >= 2]
    if len(keys) < 2:
        return np.nan, np.nan, np.nan, len(keys), float(np.mean(c_all))
    cc = np.array([count[k] for k in keys], dtype=float)
    s = np.array([sums[k] for k in keys])
    ss = np.array([sumsq[k] for k in keys])
    means = s / cc
    within = np.clip((ss - cc * means ** 2) / (cc - 1), 0.0, None)
    fold_var = float(np.mean(within))
    sample_var = max(float(np.var(means, ddof=1) - np.mean(within / cc)), 0.0)
    total = sample_var + fold_var
    icc = float(sample_var / total) if total > 0 else np.nan
    return sample_var, fold_var, icc, int(len(keys)), float(np.mean(c_all))


def per_sample_losses(probs, labels):
    eps = 1e-12
    preds = probs.argmax(axis=1)
    true_p = probs[np.arange(len(labels)), labels]
    onehot = np.eye(probs.shape[1])[labels]
    return {"err01": (preds != labels).astype(float), "nll": -np.log(true_p + eps),
            "brier": np.sum((probs - onehot) ** 2, axis=1)}


def summary(losses):
    return {"accuracy": float(1.0 - losses["err01"].mean()), "nll": float(losses["nll"].mean()),
            "brier": float(losses["brier"].mean())}


# --------------------------------------------------------------------------------------------------------------
def load_tokenizer(cfg):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["hf_name"])
    if cfg["is_decoder"]:
        tok.pad_token = tok.eos_token
        tok.padding_side = "left"
    return tok


def prepare(args):
    """Login node, CPU: tokenize the 650,000 training reviews once per model (no padding), fix the
    benchmark/pool split and the length order of the benchmark set."""
    from datasets import load_dataset
    raw = load_dataset("yelp_review_full", "yelp_review_full", download_mode="reuse_cache_if_exists")["train"]
    labels = np.asarray(raw["label"])
    bench_idx, pool_idx = train_test_split(np.arange(len(raw)), train_size=BENCH_SIZE, stratify=labels,
                                           random_state=SPLIT_SEED)
    for name in args.model_choices:
        out = Path(args.data_dir) / name
        out.mkdir(parents=True, exist_ok=True)
        tok = load_tokenizer(MODEL_CONFIGS[name])
        t0 = time.time()
        def _tok(b):
            enc = tok(b["text"], truncation=True, max_length=MAX_LENGTH)
            enc["length"] = [len(x) for x in enc["input_ids"]]
            return enc
        ds = raw.map(_tok, batched=True, remove_columns=["text"], num_proc=8, desc=f"tokenize {name}")
        lengths = np.asarray(ds["length"], dtype=np.int32)
        keep = [c for c in ("input_ids", "attention_mask", "label") if c in ds.column_names]
        ds = ds.select_columns(keep).rename_column("label", "labels")
        bench_order = bench_idx[np.argsort(lengths[bench_idx], kind="stable")]
        ds.save_to_disk(str(out / "tokenized"))
        np.savez(out / "split.npz", bench_idx=bench_idx, pool_idx=pool_idx, bench_order=bench_order,
                 lengths=lengths, labels=labels)
        print(f"{name}: tokenized {len(ds):,} reviews in {time.time() - t0:.0f} s; benchmark {len(bench_idx):,}, "
              f"pool {len(pool_idx):,}; mean length {lengths.mean():.0f}, at {MAX_LENGTH}: {np.mean(lengths == MAX_LENGTH):.3f}")


class Scorer:
    """Scores a model on given rows of the tokenized split, in length order, padding each batch to its own
    longest review. Returns class probabilities aligned with the requested rows."""

    def __init__(self, ds, lengths, tokenizer, batch_size, device):
        self.ds, self.lengths, self.tok, self.bs, self.device = ds, lengths, tokenizer, batch_size, device

    def probs(self, model, rows, presorted=False):
        import torch
        rows = np.asarray(rows)
        order = np.arange(len(rows)) if presorted else np.argsort(self.lengths[rows], kind="stable")
        out = np.empty((len(rows), model.config.num_labels), dtype=np.float32)
        model.eval()
        with torch.inference_mode():
            for s in range(0, len(order), self.bs):
                pos = order[s:s + self.bs]
                batch = self.ds[rows[pos]]
                enc = self.tok.pad({"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]},
                                   return_tensors="pt")
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    logits = model(input_ids=enc["input_ids"].to(self.device),
                                   attention_mask=enc["attention_mask"].to(self.device)).logits
                out[pos] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        return out


def check_padding(model, scorer, rows, labels):
    """Smoke test: the earlier protocol padded every review to the longest of its block of 1,000 (in practice to
    512 tokens); compare predictions and time with the length-sorted scoring on the same rows."""
    import torch
    t0 = time.time()
    p_new = scorer.probs(model, rows)
    t_new = time.time() - t0
    t0 = time.time()
    p_old = np.empty_like(p_new)
    model.eval()
    with torch.inference_mode():
        for b0 in range(0, len(rows), 1000):
            blk = rows[b0:b0 + 1000]
            batch = scorer.ds[blk]
            enc = scorer.tok.pad({"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]},
                                 padding="longest", return_tensors="pt")
            for s in range(0, len(blk), scorer.bs):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=scorer.device.type == "cuda"):
                    logits = model(input_ids=enc["input_ids"][s:s + scorer.bs].to(scorer.device),
                                   attention_mask=enc["attention_mask"][s:s + scorer.bs].to(scorer.device)).logits
                p_old[b0 + s:b0 + s + len(logits)] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    t_old = time.time() - t0
    lab = labels[rows]
    return {"rows": int(len(rows)), "time_sorted_s": t_new, "time_block_padded_s": t_old, "speedup": t_old / t_new,
            "argmax_agreement": float(np.mean(p_new.argmax(1) == p_old.argmax(1))),
            "max_abs_prob_diff": float(np.abs(p_new - p_old).max()),
            "accuracy_sorted": float(np.mean(p_new.argmax(1) == lab)), "accuracy_block_padded": float(np.mean(p_old.argmax(1) == lab))}


def run(args):
    import torch
    from datasets import load_from_disk
    from transformers import (AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer,
                              TrainingArguments)
    from transformers import logging as hf_logging

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.model_choices:
        cfg = MODEL_CONFIGS[name]
        src = Path(args.data_dir) / name
        ds = load_from_disk(str(src / "tokenized"))
        sp = np.load(src / "split.npz")
        pool_idx, bench_order, lengths, labels = sp["pool_idx"], sp["bench_order"], sp["lengths"], sp["labels"]
        bench_idx = sp["bench_idx"]
        tok = load_tokenizer(cfg)
        scorer = Scorer(ds, lengths, tok, args.eval_batch_size, device)
        collator = DataCollatorWithPadding(tok)

        def metrics_fn(eval_pred):
            logits, y = eval_pred
            return {"accuracy": float(np.mean(np.argmax(logits, axis=-1) == y))}

        for n_tr in args.train_sizes:
            study_n = int(round(n_tr / (1.0 - args.test_size)))
            rows_out = []
            for seed in range(*args.seeds):
                torch.manual_seed(seed)
                np.random.seed(seed)
                study_idx, rest_idx = train_test_split(pool_idx, train_size=study_n, stratify=labels[pool_idx],
                                                       random_state=seed)
                outer_idx = np.random.default_rng(seed).permutation(rest_idx)
                ss = ShuffleSplit(n_splits=args.n_splits, test_size=args.test_size, random_state=seed)
                fold_ids, fold_loss = [], {"brier": [], "nll": [], "err01": []}
                group = []
                for fold, (tv_pos, te_pos) in enumerate(ss.split(np.zeros(study_n))):
                    tr_pos, va_pos = train_test_split(tv_pos, test_size=0.25, random_state=seed + fold)
                    tr_rows, va_rows, te_rows = study_idx[tr_pos], study_idx[va_pos], study_idx[te_pos]
                    hf_logging.set_verbosity_error()
                    model = AutoModelForSequenceClassification.from_pretrained(cfg["hf_name"], num_labels=5)
                    if cfg["is_decoder"]:
                        model.config.pad_token_id = tok.pad_token_id
                    model.to(device)
                    tmp = tempfile.mkdtemp(dir=os.environ.get("JOBSCRATCH", os.environ.get("SCRATCH", None)))
                    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                    targs = TrainingArguments(
                        output_dir=tmp, eval_strategy="epoch", save_strategy="best", save_total_limit=1,
                        save_only_model=True, load_best_model_at_end=True, metric_for_best_model="eval_accuracy",
                        num_train_epochs=NUM_TRAIN_EPOCHS, per_device_train_batch_size=cfg["per_device_train_batch_size"],
                        per_device_eval_batch_size=args.eval_batch_size, dataloader_num_workers=4,
                        logging_steps=max(1, len(tr_rows) // (cfg["per_device_train_batch_size"] * 4)),
                        report_to="none", bf16=bf16, fp16=torch.cuda.is_available() and not bf16, seed=seed)
                    trainer = Trainer(model=model, args=targs, train_dataset=ds.select(tr_rows),
                                      eval_dataset=ds.select(va_rows), compute_metrics=metrics_fn,
                                      processing_class=tok, data_collator=collator)
                    t0 = time.time()
                    out = trainer.train()
                    train_time = time.time() - t0
                    model = trainer.model
                    best_epoch = getattr(trainer.state, "best_global_step", None)
                    res = {"model": name, "model_hf_name": cfg["hf_name"], "study_size": n_tr, "study_pool_size": study_n,
                           "n_splits": args.n_splits, "seed": seed, "fold": fold + 1, "cross_validation": True,
                           "cv_method": "shuffle_split", "stratified": False, "train_runtime": train_time,
                           "train_runtime_hf": out.metrics.get("train_runtime"), "best_global_step": best_epoch}
                    for set_name, rows in (("val", va_rows), ("test", te_rows)):
                        t0 = time.time()
                        losses = per_sample_losses(scorer.probs(model, rows), labels[rows])
                        res.update({f"{set_name}_{k}": v for k, v in summary(losses).items()})
                        res[f"{set_name}_runtime"] = time.time() - t0
                        if set_name == "test":
                            fold_ids.append(te_pos)
                            for k in fold_loss:
                                fold_loss[k].append(losses[k])
                    t0 = time.time()
                    b_losses = per_sample_losses(scorer.probs(model, bench_order, presorted=True), labels[bench_order])
                    res.update({f"benchmark_{k}": v for k, v in summary(b_losses).items()})
                    res["benchmark_runtime"] = time.time() - t0
                    res["benchmark_size"] = int(len(bench_order))
                    hidden = {"accuracies": [], "nlls": [], "briers": []}
                    if fold == 0:
                        n_te = len(te_rows)
                        n_chunks = min(args.max_outer_chunks, len(outer_idx) // n_te)
                        rows = outer_idx[: n_chunks * n_te]
                        t0 = time.time()
                        o = per_sample_losses(scorer.probs(model, rows), labels[rows])
                        for c in range(n_chunks):
                            sl = slice(c * n_te, (c + 1) * n_te)
                            m = summary({k: v[sl] for k, v in o.items()})
                            hidden["accuracies"].append(m["accuracy"]); hidden["nlls"].append(m["nll"]); hidden["briers"].append(m["brier"])
                        res["hidden_runtime"] = time.time() - t0
                        if args.check_padding:
                            # rows in the benchmark's natural order, so that blocks of 1,000 mix lengths as before
                            chk = check_padding(model, scorer, bench_idx[np.sort(np.random.default_rng(0).choice(
                                len(bench_idx), args.check_padding, replace=False))], labels)
                            print("padding check:", json.dumps(chk))
                            (out_dir / f"padding_check_{name}_seed{seed}.json").write_text(json.dumps(chk, indent=1))
                    res.update({"hidden_test_accuracies": hidden["accuracies"] or None, "hidden_test_nlls": hidden["nlls"] or None,
                                "hidden_test_briers": hidden["briers"] or None})
                    group.append(res)
                    print(f"[{name} n_tr={n_tr} seed={seed} split {fold + 1}/{args.n_splits}] train {train_time:.0f}s, "
                          f"test acc {res['test_accuracy']:.3f}, bench acc {res['benchmark_accuracy']:.4f} "
                          f"({res['benchmark_runtime']:.0f}s)", flush=True)
                    del trainer, model
                    shutil.rmtree(tmp, ignore_errors=True)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                # study-only redundancy statistics, as in the PCam script
                for tag, sel in (("all", len(group)),) + tuple((None, k) for k in range(2, len(group) + 1)):
                    ids_k = fold_ids[:sel]
                    rb, npk, msk = _pairwise_oof_icc(fold_loss["brier"][:sel], ids_k)
                    rn, _, _ = _pairwise_oof_icc(fold_loss["nll"][:sel], ids_k)
                    re01, _, _ = _pairwise_oof_icc(fold_loss["err01"][:sel], ids_k)
                    an = {m: _oof_anova(fold_loss[k][:sel], ids_k) for m, k in
                          (("squared_error", "brier"), ("nll", "nll"), ("error01", "err01"))}
                    if tag == "all":
                        for e in group:
                            e.update({"study_squared_error_rho_all_oof_intersection": rb, "study_nll_rho_all_oof_intersection": rn,
                                      "study_error01_rho_all_oof_intersection": re01, "oof_intersection_pairs": npk,
                                      "oof_intersection_mean_size": msk})
                            for m, (sv, fv, icc, _, _) in an.items():
                                e.update({f"study_oof_anova_{m}_sample_var_all": sv, f"study_oof_anova_{m}_fold_var_all": fv,
                                          f"study_oof_anova_{m}_icc_all": icc})
                    else:
                        e = group[sel - 1]
                        e.update({"study_squared_error_rho_cum_oof_intersection": rb, "study_nll_rho_cum_oof_intersection": rn,
                                  "study_error01_rho_cum_oof_intersection": re01, "oof_intersection_pairs_cum": npk,
                                  "oof_intersection_mean_size_cum": msk})
                        for m, (sv, fv, icc, n2, mc) in an.items():
                            e.update({f"study_oof_anova_{m}_sample_var_cum": sv, f"study_oof_anova_{m}_fold_var_cum": fv,
                                      f"study_oof_anova_{m}_icc_cum": icc, "study_oof_anova_n_samples_seen_twice_cum": n2,
                                      "study_oof_anova_mean_oof_count_cum": mc})
                rows_out.extend(group)
            ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            path = out_dir / (f"yelp_results_CV_shuffle_split_unstrat_{name}_nsplits_{args.n_splits}_seeds_"
                              f"{args.seeds[0]}-{args.seeds[1] - 1}_sizes_{n_tr}_{ts}.parquet")
            pd.DataFrame(rows_out).to_parquet(path)
            print(f"wrote {path}")


if __name__ == "__main__":
    a = parse_arguments()
    prepare(a) if a.step == "prepare" else run(a)
