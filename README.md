# Surrogate-Driven Progressive Look-Ahead Framework for Monitoring Network Design

Code accompanying our work on surrogate-assisted, look-ahead optimal monitoring
well placement under ES-MDA data assimilation. The pipeline couples a VAE
parameterization of the hydraulic conductivity field K with two FNO surrogates
(K→H, K→C), uses ES-MDA for sequential assimilation, and selects new wells by
minimizing the expected posterior concentration uncertainty through a virtual
("look-ahead") assimilation step.


## 1. Environment

    pip install -r requirements.txt

The code requires PyTorch with CUDA; install a torch build that matches the
local CUDA toolkit before installing the rest.


## 2. Data and pretrained models

Download the archive and extract it into the repository root:

    https://drive.google.com/file/d/1BBjbfqQv13MZeY1xy2t6oOWb2tXCdEUj/view?usp=sharing

The archive contains two groups of files.

Datasets — raw training/test fields and the packaged test set:

    data/DataK_6000_51_51.mat
    data/DataH_6000_5_51_51.mat
    data/DataC_6000_5_51_51_Original.mat
    data/test/DataK_test_100_51_51.mat
    data/test/DataH_test_100_5_51_51.mat
    data/test/DataC_100_5_51_51_Original.mat
    test_synthetic_data/synthetic_test_100.npz

Pretrained weights — VAE and the two FNO surrogates:

    vae_results/best_vae.pth
    fno_kh_results/best_fno_kh.pth
    fno_kh_results/normalization_params.npz
    fno_kc_results/best_fno_kc.pth
    fno_kc_results/normalization_params.npz


## 3. Train the surrogates and prepare the test set  (skip if using the downloaded files)

    python train_vae.py        # VAE   : K parameterization      -> vae_results/
    python train_f1.py         # FNO-1 : K -> hydraulic head H   -> fno_kh_results/
    python train_f2.py         # FNO-2 : K -> concentration C    -> fno_kc_results/
    python data_synthetic.py   # pack the 100 test cases         -> test_synthetic_data/synthetic_test_100.npz

`data_synthetic.py` builds the packaged test set from the raw .mat truths;
test H and C come from the physical simulator rather than the FNO surrogates
to avoid inverse crime.


## 4. Evaluate the surrogates

    python evaluate_surrogate.py

Reports K reconstruction (VAE), H prediction (FNO-1) and C prediction (FNO-2)
metrics on the 100-sample test set.


## 5. Validate ES-MDA in isolation

This step isolates the assimilation algorithm from the well-placement
strategy: for each test case it computes the prior, the posterior after 3 LHS
wells, and the posterior after 20 LHS wells (all wells drawn by LHS, no
adaptive selection).

    python run_es_mda.py        # -> es_mda_results/sample_XXX.npz
    python evaluate_es_mda.py   # -> es_mda_results/evaluation/


## 6. Main experiment: 100-sample well-placement comparison

    python run_100.py

Compares four placement strategies — Look-ahead, Greedy, LHS and Random —
under an identical sequential ES-MDA protocol. Results are written to
`test_100_results/sample_XXX/{active,greedy,lhs,random}.npz`, where
`active.npz` corresponds to the Look-ahead strategy. A `progress.json`
checkpoint supports resuming.


## 7. Evaluate the four strategies

    python evaluate_unified.py        # -> test_100_results/evaluation/

Produces the box plots, learning curves, scatter plots, pairwise Wilcoxon
tests with Cohen's d, spatial-distribution statistics and the
cost-performance analysis used in the paper.


## 8. Optional: saturation analysis

    python saturation_curve.py        # -> saturation_results/

Sweeps the well count under LHS placement and tracks the ensemble C
uncertainty, supporting the choice of 20 wells as the saturation point.


## 9. Optional: per-sample visualization

    python run_selected_samples.py    # set SELECTED_SAMPLES at the top of the file
    python figure_plot.py

Generates the full per-sample fields and the paper-style composite figures.
Outputs go to `test_selected_results/` and `test_100_results/paper_figures_output/`.


All scripts must be launched from the repository root. Output directories are
not tracked in the repository; they are recreated by the scripts above.
