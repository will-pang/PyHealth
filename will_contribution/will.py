# Change directory to package root
import os
PROJECT_ROOT = '/Users/wpang/Desktop/PyHealth'
os.chdir(PROJECT_ROOT)

# Other General Packages
from datetime import datetime
from typing import Any, Dict, List, Optional

# PyHealth Packages
from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import MultimodalMortalityPredictionMIMIC4

# Paths
EHR_ROOT = os.path.join(PROJECT_ROOT, "srv/local/data/physionet.org/files/mimiciv/2.2")
NOTE_ROOT = os.path.join(PROJECT_ROOT, "srv/local/data/physionet.org/files/mimic-iv-note/2.2")
CXR_ROOT = os.path.join(PROJECT_ROOT,"srv/local/data/physionet.org/files/mimic-cxr-jpg/2.0.0")
CACHE_DIR = os.path.join(PROJECT_ROOT,"srv/local/data/wp/pyhealth_cache")

import shutil
def delete_cache(cache_directory):
    for item in os.listdir(cache_directory):
        item_path = os.path.join(cache_directory, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

    print(f"Cache deleted successfully from: {cache_directory}")

if __name__ == "__main__":
    # delete_cache(CACHE_DIR)
    dataset = MIMIC4Dataset(
            ehr_root=EHR_ROOT,
            note_root=NOTE_ROOT,
            cxr_root=CXR_ROOT,
            ehr_tables=["diagnoses_icd", "procedures_icd",
                    "prescriptions", "labevents"],
            note_tables=["discharge", "radiology"],
            cxr_tables=["metadata", "negbio"],
            cache_dir=CACHE_DIR,
            num_workers=8
        )

    task = MultimodalMortalityPredictionMIMIC4()
    samples = dataset.set_task(task)

    sample = samples[0]
    print(sample)

    print("Done!")