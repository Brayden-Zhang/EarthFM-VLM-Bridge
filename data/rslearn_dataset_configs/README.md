These are rslearn dataset configuration files for different modalities in Helios.

Some notes:
- `config_sentinel2.json`: this specifies tile store at `/tmp/rslearn_helios_tile_store`
  because materialization is meant to be run from many Beaker jobs. Use
  `python helios/dataset_creation/scripts/sentinel2_l1c/launch_jobs.py` to launch the
  ingestion/materialization jobs. Each job will process one batch (default 10) of
  windows.
