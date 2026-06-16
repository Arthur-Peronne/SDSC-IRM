# SDSC-IRM

## Setup

Follow these steps in order.

### 1. Common: Set up environment variables
Create a `.env` file from the template and update it with your configuration:
```bash
cp .env.example .env
```

### 2. Set up Virtual Environment (skip if on a standard writable Renku session)
If you are working on your own machine, or if Renku's default Python environment
is not writable (see alternative below), create and activate a virtual environment:
```bash
python3 -m venv ../.venv
source ../.venv/bin/activate
```
### 2 ALTERNATIVE. When Renku Python env is not writable (AI Agent project)
If `pip install` on Renku fails with a `Permission denied` error, create and activate
a virtual environment to isolate dependencies. Install pip via a workaround first,
since `ensurepip` is unavailable on this image:
```bash
python3 -m venv ../.venv --without-pip
source ../.venv/bin/activate
unset PYTHONPATH
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
rm get-pip.py
```
NB: unset PYTHONPATH needs to be redone when opening a new terminal everytime...

### 3. Common: Install Dependencies & Run
Install the required libraries and the project itself, then run your scripts from the project root:
```bash
pip install -r requirements_locked.txt
pip install -e .
```