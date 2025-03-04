# Worldcover sampling

### 1. Create features from WorldCover

This requires running `20250122_worldcover_sampling.py`. The [`Dockerfile`](Dockerfile) and [`beaker_entrypoints`](beaker_entrypoints.py) allow this to be easily launched from Beaker.

This script will split the globe (according to WorldCover) into `TILE_SIZE x TILE_SIZE` pixels (in the script, `TILE_SIZE` is `1000`, which corresponds to 10km. [This job](https://beaker.allen.ai/orgs/ai2/workspaces/gabi-workspace/work/01JJ79NQDZPFVB0HD8VJHXKAYT?taskId=01JJ79NQEDT06JE5TBYMS88YS7&jobId=01JKXPPW30874PZAHMWC2M5749) is running with `TILE_SIZE == 1000` but is taking a very long time and therefore regularly fails with S3 connection issues).

### 2. Use those features to train a
Once the file is created, we use a k-means clustering algorithm to find the centroids of k clusters, where k corresponds to the number of points we want to export.

Two strategies are adopted - one where we sample 50 points per tile (this is saved in `"esa_grid_subsampled.csv"`) and one where we take k points globally (this is saved in `"esa_grid_subsampled_global.csv"` and takes far longer to run).
