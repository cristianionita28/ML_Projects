Perceptron Algorithm & Decision Boundary

This repository contains the Jupyter notebook Perceptron Algorithm with visualisation.ipynb, which implements the perceptron learning algorithm on a 2‑dimensional dataset. The notebook loads the data, trains a perceptron to separate two classes labelled 0/1, reports the training accuracy at each epoch and plots the resulting decision boundary.

Quick setup

Clone this repository and change into its directory:

git clone https://github.com/<username>/<repository_name>.git
cd <repository_name>


Create the environment (pick one method):

Conda/Mamba: if you have an environment.yaml file, run:

conda env create -f environment.yaml
conda activate <env-name>


Replace <env-name> with the environment name defined in the YAML.

Pip/virtualenv: install dependencies from requirements.txt:

python -m venv venv
source venv/bin/activate  # on Windows: venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt

Running the notebook

Start Jupyter in the activated environment and open the notebook:

jupyter lab    # or: jupyter notebook

Run in the cloud

If you prefer not to install anything locally, you can run the notebook in the cloud. Replace <username> and <repository_name> with your GitHub details:

Binder:

Google Colab:

Binder will build a container using your environment.yaml (if present) or requirements.txt. Colab will install dependencies from requirements.txt in the first notebook cells.

Repository contents

Perceptron Algorithm with visualisation.ipynb – the main notebook with code, training logic and decision boundary plot.

environment.yaml – optional Conda/Mamba environment definition.

requirements.txt – list of Python packages for pip installation.
