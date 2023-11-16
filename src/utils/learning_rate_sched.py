import tensorflow as tf

class CustomLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, max_learning_rate, warmup_steps, total_steps):
        super(CustomLearningRateSchedule, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.max_learning_rate = max_learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.warmup_lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=self.initial_learning_rate,
            decay_steps=self.warmup_steps,
            end_learning_rate=self.max_learning_rate,
            power=1.0
        )
        self.decay_lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=self.max_learning_rate,
            decay_steps=self.total_steps - self.warmup_steps,
            end_learning_rate=0.0,
            power=1.0
        )

    def __call__(self, step):
        return tf.cond(
            step < self.warmup_steps,
            lambda: self.warmup_lr_schedule(step),
            lambda: self.decay_lr_schedule(step - self.warmup_steps)
        )