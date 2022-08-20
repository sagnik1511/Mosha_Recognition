training = {
    "seed": 42,
    "dataset_config": "base",
    "model_config": "base",
    "optimizer": "adam",
    "loss": "categorical_crossentropy",
    "metrics": ["accuracy"],
    "max_epochs": 20,
    "steps_per_epoch": None,
    "verbose": 1,
    "callbacks": [
        {
            "name": "checkpoint",
            "monitor": "val_loss",
            "save_best": True,
            "save_weights": False,
            "filepath": "results/checkpoint"
        },
        {
            "name": "earlystop",
            "monitor": "val_loss",
            "restore_best_weights": True,
            "patience": 10
        }
    ],
    "result_figure_path": "results/training_result.jpg"
}
