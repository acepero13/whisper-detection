import tensorflow as tf
from tensorflow import keras


@tf.keras.utils.register_keras_serializable()
class MinMaxNormalization(keras.layers.Layer):
    def __init__(
            self,
            min_val=0.0,
            max_val=1.0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_val = min_val
        self.max_val = max_val

    def build(self, input_shape):
        super(MinMaxNormalization, self).build(input_shape)

    @tf.autograph.experimental.do_not_convert
    def call(self, audio, training=False):
        v_min, v_max = tf.reduce_min(audio), tf.reduce_max(audio)
        norm = (audio - v_min) / (v_max - v_min)
        scaled = norm * (self.max_val - self.min_val) + self.min_val

        return scaled

    def get_config(self):
        config = super(MinMaxNormalization, self).get_config()
        config.update(
            {
                "min_val": self.min_val,
                "max_val": self.max_val,
            }
        )
        return config



