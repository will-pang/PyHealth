## MIMIC Dataset
To make the code run faster in Jupyter Notebook, I decided to make the following files smaller:

### EHR
```bash
cd /Users/wpang/Desktop/PyHealth/srv/local/data/physionet.org/files/mimiciv/2.2/hosp head -n 10001 labevents_full.csv > labevents.csv
```

```bash
cd /Users/wpang/Desktop/PyHealth/srv/local/data/physionet.org/files/mimiciv/2.2/hosp head -n 10001 prescriptions_full.csv > prescriptions.csv
```

### Notes
```bash
cd /Users/wpang/Desktop/PyHealth/srv/local/data/physionet.org/files/mimic-iv-note/2.2/note head -n 10001 discharge_full.csv > discharge.csv
```
```bash
cd /Users/wpang/Desktop/PyHealth/srv/local/data/physionet.org/files/mimic-iv-note/2.2/note head -n 10001 radiology_full.csv > radiology.csv
```
