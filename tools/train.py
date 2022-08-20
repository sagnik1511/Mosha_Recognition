import tensorflow as tf
import matplotlib.pyplot as plt
from models.base_model import BaseModel
from utils.training import generate_callbacks
from datasets.custom_dataset import generate_loaders
from config.training.base import training as training_base


def train(training_config):
    
    # loading dataset configuration
    if training_config["dataset_config"] == "base":
        from config.datasets import base as dataset_base
        dataset_config = dataset_base.dataset
    else:
        raise NotImplementedError

    # loading model configuration
    if training_config["model_config"] == "base":
        from config.models import base as model_base
        model_config = model_base.model
    else:
        raise NotImplementedError
    
    # loading dataset
    print(dataset_config)
    train_loader, val_loader = generate_loaders(dataset_config)
    print(train_loader.class_indices)

    # loading model
    print(model_config)
    model = BaseModel(model_config)

    # compilation of model with optimizer
    model.compile(
        optimizer = training_config["optimizer"],
        loss = training_config["loss"],
        metrics = training_config["metrics"]
    )

    # defining callbacks
    callbacks = generate_callbacks(training_config["callbacks"])

    # trainin job
    history = model.fit(
        train_loader,
        epochs = training_config["max_epochs"],
        batch_size = dataset_config["batch_size"],
        steps_per_epoch = training_config["steps_per_epoch"],
        validation_data = val_loader,
        callbacks = callbacks,
        verbose = training_config["verbose"]
    )

    # training steps visualization
    metrics = training_config["metrics"] + ["loss"]
    num_metrics = len(metrics)
    ax, fig = plt.subplots(1, num_metrics, figsize=(20, 6))
    for index, metric in enumerate(metrics):
        plt.subplot(1, num_metrics, index+1)
        plt.plot(history.history[metric], label = f"Train {metric}")
        plt.title(metric)
        plt.plot(history.history[f"val_{metric}"], label = f"Validation {metric}")
        plt.legend()
    plt.savefig(training_config["result_figure_path"])

    print("Training completed...")
    

    
if __name__ == "__main__":
    print(training_base)
    tf.random.set_seed(training_base["seed"])
    train(training_base)
