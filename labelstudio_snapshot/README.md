
# Snapshot pipeline

## # 1 Core steps

1) Download Label Studio snapshot as JSON file.
    * Only annotated tasks should be included in the snapshot.

2) Run `snapshot_preparation.py`
   
```bash
# creates directory  with preprocessed files
~/snapshot_prepared
├── email_mapping.csv # mapping original emails to pseudonymized ones
├── email_mapping.parquet
├── snapshot_prepared.json # preprocessed JSON file
└── snapshot_prepared.parquet # preprocessed Parquet file
```

## # 2 Statistics

1) Run `stats_*.py` scripts to get statistics from the preprocessed files (`polars.DataFrame`)
2) Each script saves statistics as `.csv`

```bash
ls_snapshot_output/stats
├── *.csv
└── *.csv
```

## # 3 Push to HF `dataset`

TODO