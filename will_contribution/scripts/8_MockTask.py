# Change directory to package root
import os
PROJECT_ROOT = '/Users/wpang/Desktop/PyHealth'
os.chdir(PROJECT_ROOT)

# Other General Packages
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

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
        self.input_schema: Dict[str, Union[str, Tuple[str, Dict]]] = {
            "discharge_note_times": (
                "tuple_time_text",
                {
                    "tokenizer_name": "bert-base-uncased",
                    "type_tag": "note",
                },
            ),
            "radiology_note_times": (
                "tuple_time_text",
                {
                    "tokenizer_name": "bert-base-uncased",
                    "type_tag": "note",
                },
            ),
        }
        self.output_schema: Dict[str, str] = {"mortality": "binary"}

    def _clean_text(self, text: Optional[str]) -> Optional[str]:
        """Return text if non-empty, otherwise None."""
        return text if text else None

    def _compute_time_diffs(self, notes_with_timestamps, first_admission_time):
        """Compute hourly time offsets for notes relative to first admission.

        Sorts notes chronologically by timestamp, then computes each note's
        offset (in hours) from the first admission time.

        Args:
            notes_with_timestamps: List of (text, timestamp) tuples where
                text is the clinical note string and timestamp is a datetime.
            first_admission_time: datetime of the patient's first admission,
                used as the anchor (t=0) for all time offsets.

        Returns:
            Tuple of (texts, time_diffs) where:
                - texts: List[str] of note contents, sorted chronologically
                - time_diffs: List[float] of hours since first admission
            Returns (["<missing>"], [0.0]) if no notes are available.
        """
        result = []

        if not notes_with_timestamps:
            return (["<missing>"], [0.0]) # TODO: Need to also figure out how to tokenize missing timestamps
        notes_with_timestamps.sort(key=lambda x: x[1])
        result = [(text, (ts - first_admission_time).total_seconds() / 3600) for text, ts in notes_with_timestamps]
        texts, time_diffs = zip(*result)
        
        return (list(texts), list(time_diffs))

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

        # Aggregated notes and time offsets across all admissions
        all_discharge_texts: List[str] = []
        all_discharge_times: List[float] = []
        all_radiology_texts: List[str] = []
        all_radiology_times: List[float] = []

        # Process each admission independently (per hadm_id)
        for admission in admissions_to_process:
            try:
                admission_dischtime = datetime.strptime(
                    admission.dischtime, "%Y-%m-%d %H:%M:%S"
                )
            except (ValueError, AttributeError):
                continue

            if admission_dischtime < admission.timestamp:
                continue

            # Get notes for this hadm_id only
            discharge_notes = patient.get_events(
                event_type="discharge", filters=[("hadm_id", "==", admission.hadm_id)]
            )
            radiology_notes = patient.get_events(
                event_type="radiology", filters=[("hadm_id", "==", admission.hadm_id)]
            )

            # Collect (text, timestamp) tuples for this admission
            discharge_notes_timestamped = []
            radiology_notes_timestamped = []

            for note in discharge_notes:
                try:
                    note_text = self._clean_text(note.text)
                    if note_text:
                        discharge_notes_timestamped.append((note_text, note.timestamp))
                except AttributeError:
                    pass

            for note in radiology_notes:
                try:
                    note_text = self._clean_text(note.text)
                    if note_text:
                        radiology_notes_timestamped.append((note_text, note.timestamp))
                except AttributeError:
                    pass

            # Compute time diffs relative to THIS admission's start (per hadm_id)
            if discharge_notes_timestamped:
                texts, times = self._compute_time_diffs(
                    discharge_notes_timestamped, admission.timestamp
                )
                all_discharge_texts.extend(texts)
                all_discharge_times.extend(times)

            if radiology_notes_timestamped:
                texts, times = self._compute_time_diffs(
                    radiology_notes_timestamped, admission.timestamp
                )
                all_radiology_texts.extend(texts)
                all_radiology_times.extend(times)

        # Fall back to <missing> if no notes found across all admissions
        if not all_discharge_texts:
            all_discharge_texts, all_discharge_times = ["<missing>"], [0.0]
        if not all_radiology_texts:
            all_radiology_texts, all_radiology_times = ["<missing>"], [0.0]

        discharge_note_times = (all_discharge_texts, all_discharge_times)
        radiology_note_times = (all_radiology_texts, all_radiology_times)

        return [
            {
                "patient_id": patient.patient_id,
                "discharge_note_times": discharge_note_times,
                "radiology_note_times": radiology_note_times,
                "mortality": mortality_label,
            }
        ]

### END OF CLASS

if __name__ == "__main__":
    # delete_cache(CACHE_DIR)

    dataset = MIMIC4Dataset(
        ehr_root=EHR_ROOT,
        note_root=NOTE_ROOT,
        ehr_tables=["diagnoses_icd", "procedures_icd", "prescriptions", "labevents"],
        note_tables=["discharge", "radiology"],
        cache_dir=CACHE_DIR,
        num_workers=8
    )

    task = EHRFoundationalModelMIMIC4()    

    # Single patient
    patient = dataset.get_patient("10000032")                                                                           
    samples = task(patient)    

    # All patients
    # samples = dataset.set_task(task, cache_dir=f"{CACHE_DIR}/task", num_workers=8)

    print("Done")