import tensorflow as tf

def weighted_crf_loss(class_weights):
    def loss(y_true, y_pred):
        # 创建一个权重矩阵
        weights = tf.reduce_sum(class_weights * tf.cast(y_true, tf.float32), axis=-1)
        # 计算原始CRF损失
        crf_loss = -crf.get_negative_log_likelihood(y_true, y_pred)
        # 应用权重
        weighted_loss = crf_loss * weights
        return tf.reduce_mean(weighted_loss)

    return loss
