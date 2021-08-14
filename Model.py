# The start/end/pad tokens will be removed from the string when computing the Levenshtein distance
START_TOKEN = tf.constant(vocabulary_to_int.get('<start>'), dtype=tf.int64)
END_TOKEN = tf.constant(vocabulary_to_int.get('<end>'), dtype=tf.int64)
PAD_TOKEN = tf.constant(vocabulary_to_int.get('<pad>'), dtype=tf.int64)

tf.keras.backend.clear_session()

# initialize the model, a dummy call to the encoder and deocder is made to allow the summaries to be printed
with strategy.scope():
    # # set half precision policy
    mixed_precision.set_policy('mixed_bfloat16' if TPU else 'float32')

    # enable XLA optmizations
    tf.config.optimizer.set_jit(True)

    print(f'Compute dtype: {mixed_precision.global_policy().compute_dtype}')
    print(f'Variable dtype: {mixed_precision.global_policy().variable_dtype}')
    
    # Sparse categorical cross entropy loss is used
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    def loss_function(real, pred):
        per_example_loss = loss_object(real, pred)

        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=BATCH_SIZE)
    
    # Metrics
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    train_loss = tf.keras.metrics.Sum()
    val_loss = tf.keras.metrics.Sum()


    # Encoder
    encoder = Encoder()
    encoder.build(input_shape=[BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, N_CHANNELS])
    encoder_res = encoder(imgs[:2], training=False)
    
    # Decoder
    decoder = Decoder(VOCAB_SIZE, ATTENTION_UNITS, ENCODER_DIM, DECODER_DIM, CHAR_EMBEDDING_DIM)
    h, c = decoder.init_hidden_state(encoder_res, training=False)
    preds, h, c = decoder(lbls[:2, :1], h, c, encoder_res, training=False)
    
    # Adam Optimizer
    optimizer = tf.keras.optimizers.Adam()
