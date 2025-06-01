COMP4434 Course Project

1. Create conda environment(Optional)
    conda create -n basisformer -y python=3.8 
    conda activate basisformer


2. Install dependecies
    pip install -r requirements.txt

3. Download the data
    We follow the same setting as previous work. The datasets for benchmarks can be obtained from [[Autoformer](https://github.com/thuml/Autoformer)].
    We select 'electricity' dataset from the six datasets
    The datasets are placed in the 'datasets' folder of our project. The tree structure of the files are as follows:

Rafu-Former\datasets
│
├─electricity

4. Run the project:
    To Start the 3 basic neural network, please directly run the gru.py, lstm.py, tcn_2.py, respectively.
    To start evaluating our Rafu-Former, please switch to the directory containing to our code, then type the following command in terminal:
        Multivariate forecasting: sh script/M.sh
        Univariate forecasting: sh script/S.sh
        Ablation experiment: change the 'use_attention' attribute in model.py to false, then type sh script/M.sh
        Use pretrain before predicting task: sh script/Mpre.sh

Declaration:
    We reused some of the code from the BasisFormer (the paper we chose) and chose the ELECTRICITY dataset (one of the six datasets from the original paper) for our project.
    We retained the structure of how the data is loaded, how the loss(mae, mse) is computed, and so on. 