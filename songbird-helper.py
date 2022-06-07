#!/usr/bin/env python3
import os
import click
import shutil
import subprocess
import pandas as pd
from biom import load_table

@click.command()
@click.option(
    "--input-biom", show_default=True, required=True)
@click.option(
    "--metadata-file", show_default=True, required=True)
@click.option(
    "--formula", show_default=True, required=True)
@click.option(
    "--pseudo-model-formula", show_default=True, required=True)
@click.option(
    "--out-dir", show_default=True, required=True)

def songbird_gird_search(input_biom, metadata_file, formula, pseudo_model_formula, out_dir):

    """ 
    Run a simple grid search of songbird perams/formulas
    against a baseline model for the MG seeding project. 
    """
    # min number of counts/sample
    min_samp = 500
    # store summary
    summary_interval = 150
    # epochs
    epch = 4500
    # get formula
    formulas = [pseudo_model_formula, formula]
    #xformulas = [formula]
    # feat freq.
    ffq = [0.05]
    # convert those to samp. #'s
    tbl_ = load_table(input_biom)
    ffq = [int(f_ * tbl_.shape[1])
           for f_ in ffq]
    # learning rates
    lrs = 0.0001
    # batch sizes
    bss = [0.10]
    bss = [int(f_ * tbl_.shape[1])
           for f_ in bss]
    # differential prior
    dps = [0.5]
    """ 
    Run all of the perams above
    """
    # run each peram(s) set
    for formula_ in formulas:
        for feat_ in ffq:
            for bs_ in bss:
                for dp_ in dps:
                    # get output dir
                    out_ = '-'.join(list(map(str, [formula_, feat_, bs_, dp_])))
                    out_ = os.path.join(out_dir, out_)
                    # make the dir (if already made rewrite)
                    if os.path.exists(out_):
                        shutil.rmtree(out_)
                    os.makedirs(out_)
                    # run songbird (save output to out_)
                    p = subprocess.Popen(['songbird', 'multinomial',
                                          '--input-biom', str(input_biom),
                                          '--metadata-file', str(metadata_file),
                                          '--formula', str(formula_),
                                          '--batch-size', str(bs_),
                                          '--epochs', str(epch),
                                          '--differential-prior', str(dp_),
                                          '--learning-rate', str(lrs),
                                          '--min-sample-count', str(min_samp),
                                          '--min-feature-count', str(feat_),
                                          '--summary-interval', str(summary_interval),
                                          '--summary-dir', str(out_),
                                          '--random-seed', str(0)],
                                         stdout=subprocess.PIPE)
                    out, err = p.communicate()

if __name__ == '__main__':
    songbird_gird_search()
