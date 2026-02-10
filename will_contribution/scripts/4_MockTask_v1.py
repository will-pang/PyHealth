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
from pyhealth.tasks.base_task import BaseTask

# Will's Contribution Utilities
from will_contribution.utils import delete_cache

import polars as pl

pl.Config.set_tbl_rows(1000)
pl.Config.set_tbl_cols(100)
pl.Config.set_fmt_str_lengths(1000)

# Paths
EHR_ROOT = os.path.join(PROJECT_ROOT, "srv/local/data/physionet.org/files/mimiciv/2.2")
NOTE_ROOT = os.path.join(PROJECT_ROOT, "srv/local/data/physionet.org/files/mimic-iv-note/2.2")
CXR_ROOT = os.path.join(PROJECT_ROOT,"srv/local/data/physionet.org/files/mimic-cxr-jpg/2.0.0")
CACHE_DIR = os.path.join(PROJECT_ROOT,"srv/local/data/wp/pyhealth_cache")

class EHRFoundationalModelMIMIC4(BaseTask):
    
    task_name: str = "EHRFoundationalModelMIMIC4"
    
    def __init__(self):
        """Initialize the EHR Foundational Model task."""
        self.input_schema: Dict[str, str] = {
            "discharge": "raw",
            "radiology": "raw",
            # "discharge_note_timestamps": "raw",
            # "discharge_note_time_diffs": "raw",
            # "radiology_note_timestamps": "raw",
            # "radiology_note_time_diffs": "raw",
        }
        self.output_schema: Dict[str, str] = {"mortality": "binary"}

    def _clean_text(self, text: Optional[str]) -> Optional[str]:
        """Return text if non-empty, otherwise None."""
        return text if text else None

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        # Get demographic info to filter by age
        demographics = patient.get_events(event_type="patients")
        if not demographics:
            return []

        demographics = demographics[0]

        # Get visits
        admissions = patient.get_events(event_type="admissions")
        if len(admissions) == 0:
            return []

        # Determine which admissions to process iteratively
        # Check each admission's NEXT admission for mortality flag
        admissions_to_process = []
        mortality_label = 0

        for i, admission in enumerate(admissions):
            # Check if THIS admission has the death flag
            if admission.hospital_expire_flag in [1, "1"]:
                # Patient died in this admission - set mortality label
                # but don't include this admission's data
                mortality_label = 1
                break

            # Check if there's a next admission with death flag
            if i + 1 < len(admissions):
                next_admission = admissions[i + 1]
                if next_admission.hospital_expire_flag in [1, "1"]:
                    # Next admission has death - include current, set mortality
                    admissions_to_process.append(admission)
                    mortality_label = 1
                    break

            # No death in current or next - include this admission
            admissions_to_process.append(admission)

        if len(admissions_to_process) == 0:
            return []

        # Get first admission time as reference for lab time calculations
        first_admission_time = admissions_to_process[0].timestamp

        # Aggregated data across all admissions
        all_discharge_notes = []  # List of individual discharge notes
        all_radiology_notes = []  # List of individual radiology notes
        all_discharge_notes_timestamps = [] # List of individual discharge notes timestamps
        all_radiology_notes_timestamps = [] # List of individual discharge notes timestamps
        discharge_note_time_diffs = []
        radiology_notes_time_diffs = []
        

        # Process each admission and aggregate data
        for admission in admissions_to_process:
            # Parse admission discharge time for lab events filtering
            try:
                admission_dischtime = datetime.strptime(
                    admission.dischtime, "%Y-%m-%d %H:%M:%S"
                )
            except (ValueError, AttributeError):
                # If we can't parse discharge time, skip this admission
                continue

            # Skip if discharge is before admission (data quality issue)
            if admission_dischtime < admission.timestamp:
                continue

            # Get notes using hadm_id filtering
            discharge_notes = patient.get_events(
                event_type="discharge", filters=[("hadm_id", "==", admission.hadm_id)]
            )
            radiology_notes = patient.get_events(
                event_type="radiology", filters=[("hadm_id", "==", admission.hadm_id)]
            )

        # Extract and aggregate notes as individual items in lists
            # Note: attribute is "text" (from mimic4_note.yaml), not "discharge"/"radiology"
            for note in discharge_notes:
                try:
                    note_text = self._clean_text(note.text)
                    if note_text:
                        all_discharge_notes.append(note_text)
                        all_discharge_notes_timestamps.append((note.timestamp, "discharge"))
                except AttributeError:
                    pass

            for note in radiology_notes:
                try:
                    note_text = self._clean_text(note.text)
                    if note_text:
                        all_radiology_notes.append(note_text)
                        all_radiology_notes_timestamps.append((note.timestamp, "radiology"))
                except AttributeError:
                    pass

        # Sort discharge_notes by timestamp 
        all_discharge_notes_timestamps.sort(key=lambda x: x[0])
        all_radiology_notes_timestamps.sort(key=lambda x: x[0])

        # Compute time difference for discharge notes (hours)
        discharge_note_time_diffs = [0.0] + [
            (t2[0] - t1[0]).total_seconds() / 3600
            for t1, t2 in zip(
                all_discharge_notes_timestamps,
                all_discharge_notes_timestamps[1:]
            )
        ]

        # Compute time difference for radiology notes (hours)
        radiology_note_time_diffs = [0.0] + [
            (t2[0] - t1[0]).total_seconds() / 3600
            for t1, t2 in zip(
                all_radiology_notes_timestamps,
                all_radiology_notes_timestamps[1:]
            )
        ]

        # ===== MODALITY REQUIREMENTS =====
        # Check notes - need at least one discharge OR radiology note
        has_notes = len(all_discharge_notes) > 0 or len(all_radiology_notes) > 0

        #Return empty list if any required modality is missing
        if not (
            has_notes
        ):
            return []


        return [
            {
                "patient_id": patient.patient_id,
                "discharge": all_discharge_notes,  # List of discharge notes
                "discharge_note_timestamps": [str(t) for t in all_discharge_notes_timestamps],
                "discharge_note_time_diffs": [str(t) for t in discharge_note_time_diffs],
                "radiology": all_radiology_notes,  # List of radiology notes
                "radiology_note_timestamps": [str(t) for t in all_radiology_notes_timestamps],  
                "radiology_note_time_diffs": [str(t) for t in radiology_note_time_diffs],
                "mortality": mortality_label
            }
        ]

if __name__ == "__main__":
    delete_cache(CACHE_DIR)

    dataset = MIMIC4Dataset(
        ehr_root=EHR_ROOT,
        note_root=NOTE_ROOT,
        ehr_tables=["diagnoses_icd", "procedures_icd", "prescriptions", "labevents"],
        note_tables=["discharge", "radiology"],
        cache_dir=CACHE_DIR,
        num_workers=16
    )

    task = EHRFoundationalModelMIMIC4()    

    # Single patient
    # patient = dataset.get_patient("10000032")                                                                           
    # samples = task(patient)    

    # All patients
    samples = dataset.set_task(task, cache_dir=f"{CACHE_DIR}/task", num_workers=8)

    print("Done")