# SDSC-IRM

## References

O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, et al., "Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved?", IEEE Transactions on Medical Imaging, vol. 37, no. 11, pp. 2514-2525, Nov. 2018. doi: 10.1109/TMI.2018.2837502

## Scientific description

## Results

See reports/

## Setup

Follow these steps in order.

### 1. Common: Set up environment variables
Create a `.env` file from the template and update it with your configuration:
```bash
cp .env.example .env
```

### 2. Set up Virtual Environment (skip if on a standard writable Renku session)
If you are working on your own machine, create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Common: Install Dependencies & Run
Install the required libraries and the project itself, then run your scripts from the project root:
```bash
pip install -r requirements_locked.txt
pip install -e .
```


## References



## Commandes utiles

```bash
# MLflow UI en local (depuis la racine du projet, venv activé)
mlflow ui --backend-store-uri mlruns/
# → http://127.0.0.1:5000

```bash
# Lancer jobs
nohup python -u scripts/run_autoencoder.py > alldims_aiarchi_claudeexp2.txt 2>&1 &
```

```bash
# Créer des archives pour passer entre sessions Renku
tar -czf tempdata_autoencoder.tar.gz tempodata/autoencoder/
tar -czf ACDC.tar.gz /home/renku/work/s3-bucket/ACDC
```