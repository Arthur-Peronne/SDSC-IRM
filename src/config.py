# src/config.py
"""
Central configuration — paths loaded from environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# Environment-dependent paths (defined in .env)
DATADIR = Path(os.environ["DATADIR"])
RESULTS_FOLDER = Path(os.environ["RESULTS_FOLDER"])
PROCESSED_IMAGES_FOLDER = Path(os.environ["PROCESSED_IMAGES_FOLDER"])
MLRUNS_FOLDER = Path(os.environ["MLRUNS_FOLDER"])

# Deprecated — will be removed once all scripts are migrated to PROCESSED_IMAGES_FOLDER + MLflow
TEMPODATA_FOLDER = Path(os.environ["TEMPODATA_FOLDER"])