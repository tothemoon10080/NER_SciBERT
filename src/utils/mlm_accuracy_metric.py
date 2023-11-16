import tensorflow as tf

class MaskedLanguageModelAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='mlm_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct_count = self.add_weight(name="correct_count", initializer="zeros")
        self.total_count = self.add_weight(name="total_count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # 假设掩码的标签是-100
        mask = tf.not_equal(y_true, -100)
        # 获取预测概率最高的类别
        predictions = tf.argmax(y_pred, axis=-1)
        # 创建一个布尔数组，表示预测是否正确
        accuracy_mask = tf.math.equal(predictions, tf.cast(y_true, tf.int64))
        # 通过逻辑与操作，仅在掩码的位置考虑准确性
        masked_accuracy = tf.math.logical_and(mask, accuracy_mask)
        # 转换为浮点数来计算准确率
        masked_accuracy = tf.cast(masked_accuracy, tf.float32)
        mask = tf.cast(mask, tf.float32)

        # 更新正确预测的总数和掩码的总数
        self.correct_count.assign_add(tf.reduce_sum(masked_accuracy))
        self.total_count.assign_add(tf.reduce_sum(mask))

    def result(self):
        # 计算准确率
        return self.correct_count / self.total_count

    def reset_states(self):
        # 在每个epoch开始时重置计数器
        self.correct_count.assign(0)
        self.total_count.assign(0)
