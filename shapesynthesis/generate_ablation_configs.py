import yaml

config_str = """
data:
  module: datasets.shapenetcore
  root: ./data/shapenet
  cates:
    - {cate}
  num_workers: 0 # Only for ubuntu.
  batch_size: 16
  pin_memory: True
  num_pts: 2048
  drop_last: False
  force_reload: False
  ectconfig:
    seed: 2024
    num_thetas: 8
    resolution: 8
    scale: 1
    r: {radius}
    ect_type: "points"
    ambient_dimension: 3
    normalized: True


render:
  num_pts: 2048
  num_epochs: 2000
  dataset_name: {cate}
  experiment_name: {resolution}_{scale}_{cate}
  ectconfig:
    seed: 2024
    num_thetas: {resolution}
    resolution: {resolution}
    scale: {scale}
    r: {radius}
    ect_type: "points"
    ambient_dimension: 3
    normalized: True
"""

for resolution in [32, 64, 128]:
    for scale in [resolution, resolution // 2, resolution // 4]:
        for dataset, radius in zip(["airplane", "car", "chair"], [7, 4.5, 5]):
            with open(
                f"./shapesynthesis/configs/ect-render/render_{dataset}_{resolution}_{scale}.yaml",
                "w",
            ) as f:
                config = yaml.safe_load(
                    config_str.format(
                        cate=dataset,
                        resolution=resolution,
                        scale=scale,
                        radius=radius,
                    )
                )
                yaml.safe_dump(
                    config,
                    f,
                )
            # print(
            #     "Dataset",
            #     dataset,
            #     "\tResolution",
            #     resolution,
            #     "\t Scale",
            #     scale,
            # )
