import tensorflow as tf
from tensorflow import keras
from jiwer import wer
from .generals import decode_batch_predictions
from .datasets import *
import numpy as np

class CallbackEval(keras.callbacks.Callback):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs=None):
        predictions = []
        targets = []
        for batch in self.dataset:
            y_pred = self.model.predict(batch)
            y_pred = decode_batch_predictions(y_pred)
            predictions.extend(y_pred)
            for label in batch['y_true']:
                label = tf.strings.reduce_join(NUM_TO_CHAR(label)).numpy().decode('utf-8')
                targets.append(label)
        wer_score = wer(targets, predictions)
        print(f'WER: {wer_score:.4f}')
        for i in np.random.randint(0, len(predictions), 24):
            print(f'True: {targets[i]}')
            print(f'Pred: {predictions[i]}')
