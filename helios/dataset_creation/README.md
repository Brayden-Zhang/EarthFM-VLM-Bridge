Create Windows
--------------

The first step is to create windows in an rslearn dataset based on one or more sampling
strategies.

Make a new folder for the dataset and copy the dataset configuration files that is used
for initializing the dataset (it has NAIP and Sentinel-2 data sources configured, which
are used to pick the window timestamp and filter out windows that don't have Sentinel-2
coverage).

    mkdir dataset/
    cp data/rslearn_dataset_configs/config_init.json dataset/config.json

Run the random sampling strategy:

    python -m helios.dataset_creation.create_windows.random --ds_path dataset/ --count 1000

Create windows based on locations in a JSON file containing a list of [lon, lat]
sub-lists:

    python -m helios.dataset_creation.create_windows.from_lon_lat_list --ds_path dataset/ --fname list.json


Materialize Data
----------------

Now we use rslearn to materialize the data.

Sentinel-1 and Sentinel-2 L2A are ingested from Microsoft Planetary Computer which
supports random access so it is relatively fast. For 10K+ windows, it may be helpful to
parallelize the materialize commands.

    cp data/rslearn_dataset_configs/config_X.json dataset/config.json
    rslearn dataset prepare --root dataset/ --group res_10 --workers 64 --no-use-initial-job
    rslearn dataset materialize --root dataset/ --group res_10 --workers 64 --no-use-initial-job

NAIP is also from Planetary Computer, but it only applies to the 0.625 m/pixel windows:

    cp data/rslearn_dataset_configs/config_naip.json dataset/config.json
    rslearn dataset prepare --root dataset/ --group res_0.625 --workers 64 --no-use-initial-job
    rslearn dataset materialize --root dataset/ --group naip --workers 64 --no-use-initial-job

OpenStreetMap can be processed on one machine:

    cp data/rslearn_dataset_configs/config_openstreetmap.json dataset/config.json
    rslearn dataset prepare --root dataset/ --group res_10 --workers 16
    rslearn dataset ingest --root dataset/ --group res_10 --workers 16 --no-use-initial-job
    rslearn dataset materialize --root dataset/ --group res_10 --workers 64 --no-use-initial-job

WorldCover can also be processed on one machine:

    cp data/rslearn_dataset_configs/config_worldcover.json dataset/config.json
    rslearn dataset prepare --root dataset/ --group res_10 --workers 32
    rslearn dataset ingest --root dataset/ --group res_10 --workers 32 --no-use-initial-job
    rslearn dataset materialize --root dataset/ --group res_10 --workers 32 --no-use-initial-job

Landsat 8/9 data is from an AWS bucket and should be materialized on an AWS machine.
Then the data can be transferred back:

    cp data/rslearn_dataset_configs/config_landsat.json dataset/config.json
    rslearn dataset prepare --root dataset/ --group res_10 --workers 64 --no-use-initial-job
    rslearn dataset materialize --root dataset/ --group res_10 --workers 64 --no-use-initial-job

Sentinel-2 L1C does not support random access. A special Beaker launcher exists to
launch many jobs for materializing the data. The dataset needs to be stored on Weka.

    cp data/rslearn_dataset_configs/config_sentinel2.json /weka/dfive-default/helios/dataset_creation/rslearn_dataset/config.json
    docker build -t helios-sentinel2-l1c -f helios/dataset_creation/scripts/sentinel2_l1c/Dockerfile .
    beaker image create --name helios-sentinel2-l1c helios-sentinel2-l1c
    python helios/dataset_creation/scripts/sentinel2_l1c/launch_jobs.py --ds_path /weka/dfive-default/helios/dataset_creation/rslearn_dataset/ --image_name [BEAKER_USERNAME]/helios-sentinel2-l1c --clusters 'ai2/jupiter-cirrascale-2' --batch_size 10 --max_jobs 200

For parallelizing the materialization (besides for Sentinel-2 L1C), it can be done with
Beaker job or just session:

```
beaker session create --budget ai2/d5 --workspace ai2/earth-systems --priority high --gpus 1 --shared-memory 128GiB --bare --mount weka://dfive-default=/dfive-default
```

TODO: create Dockerfile so the steps above can be launched as Beaker job.


Convert Data
------------

Now we convert the data to Helios format.

    python -m helios.dataset.rslearn_to_helios.landsat --ds_path dataset/ --helios_path gs://ai2-helios/data/.../
    python -m helios.dataset.rslearn_to_helios.naip --ds_path dataset/ --helios_path gs://ai2-helios/data/.../
    python -m helios.dataset.rslearn_to_helios.openstreetmap --ds_path dataset/ --helios_path gs://ai2-helios/data/.../
    python -m helios.dataset.rslearn_to_helios.sentinel1 --ds_path dataset/ --helios_path gs://ai2-helios/data/.../
    python -m helios.dataset.rslearn_to_helios.sentinel2 --ds_path dataset/ --helios_path gs://ai2-helios/data/.../
    python -m helios.dataset.rslearn_to_helios.sentinel2_l2a --ds_path dataset/ --helios_path gs://ai2-helios/data/.../
    python -m helios.dataset.rslearn_to_helios.worldcover --ds_path dataset/ --helios_path gs://ai2-helios/data/.../

These conversions yield individual metadata CSV files for each window. Concatenate them
into the per-modality CSVs:

    python -m helios.dataset_creation.make_meta_summary --helios_path gs://ai2-helios/data/.../ --modality landsat --time_span two_week
    python -m helios.dataset_creation.make_meta_summary --helios_path gs://ai2-helios/data/.../ --modality landsat --time_span year
    python -m helios.dataset_creation.make_meta_summary --helios_path gs://ai2-helios/data/.../ --modality naip
    python -m helios.dataset_creation.make_meta_summary --helios_path gs://ai2-helios/data/.../ --modality openstreetmap
    python -m helios.dataset_creation.make_meta_summary --helios_path gs://ai2-helios/data/.../ --modality sentinel1 --time_span two_week
    python -m helios.dataset_creation.make_meta_summary --helios_path gs://ai2-helios/data/.../ --modality sentinel1 --time_span year
    python -m helios.dataset_creation.make_meta_summary --helios_path gs://ai2-helios/data/.../ --modality sentinel2 --time_span two_week
    python -m helios.dataset_creation.make_meta_summary --helios_path gs://ai2-helios/data/.../ --modality sentinel2 --time_span year
    python -m helios.dataset_creation.make_meta_summary --helios_path gs://ai2-helios/data/.../ --modality sentinel2_l2a --time_span two_week
    python -m helios.dataset_creation.make_meta_summary --helios_path gs://ai2-helios/data/.../ --modality sentinel2_l2a --time_span year
    python -m helios.dataset_creation.make_meta_summary --helios_path gs://ai2-helios/data/.../ --modality worldcover
