import mlx_lm
import outlines

# from text_albumentations import OutlinesModel, run_augmentation
# from text_albumentations.tasks.bullets import bullet_augmentation
#
model = outlines.from_mlxlm(*mlx_lm.load("mlx-community/Qwen3.5-4B-OptiQ-4bit"))
# runtime = OutlinesModel(model=model)
#
# rows = run_augmentation(
#     "The Transformer replaces recurrence with attention and improves parallelization.",
#     bullet_augmentation,
#     runtime,
# )
#
# for row in rows:
#     print(row.model_dump_json())
#
