# CS3264_Team6_Project

Make sure that you've the JAAD Dataset locally. Recommended to clone the JAAD Dataset 2.0 before working on the Project.

## Context Feature Pipeline (New)

`main.ipynb` now includes a hybrid, strictly-causal surroundings/context pipeline for crossing prediction.

### Added Interfaces

- `build_frame_context_cache(video_ids, db, jaad_api, cache_dir="./cache", device=0) -> None`
- `load_frame_context(cache_dir="./cache") -> tuple[pd.DataFrame, dict]`
- `engineer_context_features(df, db, context_cache_dir="./cache") -> pd.DataFrame`
- `run_ablation_experiments(df_with_context, splits, cat_cols, num_cols, threshold) -> pd.DataFrame`

### Cache Artifacts

- `cache/context_objects.parquet`
- `cache/context_semantics/{video_id}.npz`

### New Context Features

- Vehicle proximity: `nearest_vehicle_dist_norm`, `vehicle_count_r05/r10/r20`
- Pedestrian neighborhood: `neighbor_ped_count_r05/r10`, `nearest_ped_dist_norm`, `mean3_ped_dist_norm`
- Feet-point semantics: `feet_on_road`, `feet_on_sidewalk`, `feet_on_crosswalk`, `feet_semantic_conf`
- Geometry to scene boundaries: `dist_to_curb_norm`, `dist_to_crosswalk_norm`, `in_crosswalk`
- Missingness flags: `ctx_vehicle_missing`, `ctx_semantic_missing`, `ctx_crosswalk_missing`

### Runtime Dependencies

In addition to your current notebook dependencies, this pipeline expects:

- `opencv-python`
- `ultralytics`
- `transformers`
- `torch`
- `pyarrow` (for parquet cache)

To Update...
