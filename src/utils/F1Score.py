import tensorflow as tf
from tensorflow.keras.metrics import Metric
from sklearn.metrics import f1_score
import numpy as np

class F1Score(Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

class F1ScoreCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super(F1ScoreCallback, self).__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        val_data = self.validation_data
        val_pred = self.model.predict(val_data[0])
        val_pred = np.argmax(val_pred, axis=-1)
        val_true = np.argmax(val_data[1], axis=-1)
        f1 = f1_score(val_true.flatten(), val_pred.flatten(), average='weighted')
        logs['val_f1_score'] = f1
        print(f' - val_f1_score: {f1:.4f}')