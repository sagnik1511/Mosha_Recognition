from tensorflow.keras import callbacks as Callbacks


def generate_callbacks(callback_list):
    callbacks = []
    for callback in callback_list:
        if callback["name"] == "checkpoint":
            callbacks.append(Callbacks.ModelCheckpoint(
                monitor = callback["monitor"],
                filepath = callback["filepath"],
                save_best = callback["save_best"],
                save_weights = callback["save_weights"],
                ))
        elif callback["name"] == "earlystop":
            callbacks.append(Callbacks.EarlyStopping(
                monitor = callback["monitor"],
                restore_best_weights = callback["restore_best_weights"],
                patience = callback["patience"]
            ))
        else:
            raise NotImplementedError

    return callbacks


