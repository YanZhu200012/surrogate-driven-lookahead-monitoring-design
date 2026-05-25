# A-Surrogate-Driven-Progressive-Look-Ahead-Framework
This repository provides the code used for the surrogate-model-assisted ES-MDA and Look-ahead monitoring well placement experiments.

The internal file name "active" is retained for compatibility with the original result files, but it corresponds to the Look-ahead strategy in the paper.


1. Download input data and pretrained model files

Download the required files from:

https://drive.google.com/file/d/1BBjbfqQv13MZeY1xy2t6oOWb2tXCdEUj/view?usp=sharing

After downloading and extracting the files, place them in the corresponding folders as follows:

data/
    DataK_6000_51_51.mat
    DataH_6000_5_51_51.mat
    DataC_6000_5_51_51_Original.mat

data/test/
    DataK_test_100_51_51.mat
    DataH_test_100_5_51_51.mat
    DataC_100_5_51_51_Original.mat

test_synthetic_data/
    synthetic_test_100.npz

vae_results/
    best_vae.pth

fno_kh_results/
    best_fno_kh.pth
    normalization_params.npz

fno_kc_results/
    best_fno_kc.pth
    normalization_params.npz


2. Install required packages

Install the required Python packages using:

pip install -r requirements.txt

Please make sure that the PyTorch version is compatible with your local CUDA environment.


3. Train the surrogate models

Run the following scripts from the repository root directory:

python train_vae.py
python train_f1.py
python train_f2.py

The outputs will be saved to:

vae_results/
fno_kh_results/
fno_kc_results/

If pretrained model files are already downloaded, this step can be skipped.


4. Evaluate the surrogate models

After training or downloading the pretrained models, run:

python evaluate_surrogate.py

This script evaluates the three surrogate models:

VAE: K-field reconstruction
FNO-1: K to hydraulic head prediction
FNO-2: K to concentration prediction


5. Validate the ES-MDA assimilation procedure

Run:

python run_es_mda.py
python evaluate_es_mda.py

The first script runs ES-MDA on the test samples.
The second script evaluates the prior and posterior results.

The outputs will be saved to:

es_mda_results/


6. Run the main 100-sample monitoring well placement experiment

Run:

python run_100.py

This script compares four monitoring well placement strategies:

Look-ahead
Greedy
LHS
Random

The result files will be saved to:

test_100_results/

Note:
The saved file name active.npz corresponds to the Look-ahead strategy.


7. Evaluate the four well placement strategies

After running run_100.py, run:

python evaluate_unified.py

This script generates the main performance comparison, statistical tests, spatial distribution analysis, and cost-performance analysis.

The outputs will be saved to:

test_100_results/evaluation/


8. Optional: saturation curve analysis

To evaluate how concentration uncertainty changes with the number of monitoring wells, run:

python saturation_curve.py

This script is used to support the choice of 20 wells as the target monitoring budget.

The outputs will be saved to:

saturation_results/


9. Optional: representative sample visualization

To generate full outputs for selected representative samples, run:

python run_selected_samples.py

Then generate paper-style figures by running:

python figure_plot.py

The outputs will be saved to:

test_selected_results/
test_100_results/paper_figures_output/


10. Notes

All scripts should be run from the repository root directory.

Generated output folders are not included in this repository. They can be reproduced by running the scripts.

Large data files, model weights, and generated results should not be committed to GitHub. They should be downloaded from the external data link or regenerated locally.
