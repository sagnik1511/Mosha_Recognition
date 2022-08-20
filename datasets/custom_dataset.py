import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def generate_loaders(config):
    
    # loading metadata
    meta_df = pd.read_csv(config["metadata_path"])

    # adding generator
    generator = ImageDataGenerator(
        validation_split = config["validation_split"],
        horizontal_flip = config["horizontal_flip"],
        vertical_flip = config["vertical_flip"],
        zoom_range = config["zoom_range"],
        samplewise_std_normalization = config["samplewise_std_normalization"]
    )
    
    # defining training dataset
    training_dataset = generator.flow_from_dataframe(
            dataframe=meta_df,
            directory=config["data_root"],
            x_col=config["x_id"],
            y_col=config["y_id"],
            target_size = (config["height"], config["width"]),
            batch_size = config["batch_size"],
            subset='training',
            seed=config["seed"]
    )
    # defining validation dataset
    testing_dataset = generator.flow_from_dataframe(
            dataframe=meta_df,
            directory=config["data_root"],
            x_col=config["x_id"],
            y_col=config["y_id"],
            target_size = (config["height"], config["width"]),
            batch_size = config["batch_size"],
            subset='validation',
            seed=config["seed"]
    )

    return training_dataset, testing_dataset