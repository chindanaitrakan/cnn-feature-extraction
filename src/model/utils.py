from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def get_checkpoint_best_epoch(filepath):
  checkpoint_path = filepath
  checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                               save_weights_only=True,
                               save_best_only=True,
                               save_freq='epoch',
                               monitor='val_accuracy',
                               verbose=0)
  return checkpoint


def get_early_stopping():
  callback = EarlyStopping(monitor='val_accuracy', patience=5)
  return callback