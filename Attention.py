class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.H = tf.keras.layers.Dense(units, name='hidden_to_attention_units')
        self.E = tf.keras.layers.Dense(units, name='encoder_res_to_attention_units')
        self.V = tf.keras.layers.Dense(1, name='score_to_alpha')

    def call(self, h, encoder_res, training, debug=False):
        # dense hidden state to attention units size and expand dimension
        h_expand = tf.expand_dims(h, axis=1) # expand dimension
        if debug:
            print(f'h shape: {h.shape}, encoder_res shape: {encoder_res.shape}')
            print(f'h_expand shape: {h_expand.shape}')
            
        h_dense = self.H(h_expand, training=training)
        
        # dense features to units size
        encoder_res_dense = self.E(encoder_res, training=training) # dense to attention

        # add vectors
        score = tf.nn.relu(h_dense + encoder_res_dense)
        if debug:
            print(f'h_dense shape: {h_dense.shape}')
            print(f'encoder_res_dense shape: {encoder_res_dense.shape}')
            print(f'score tanh shape: {score.shape}')
        score = self.V(score, training=training)
        
        # create alpha vector size (bs, layers)        
        attention_weights = tf.nn.softmax(score, axis=1)
        if debug:
            score_np = score.numpy().astype(np.float32)
            print(f'score V shape: {score.shape}, score min: %.3f score max: %.3f' % (score_np.min(), score_np.max()))
            print(f'attention_weights shape: {attention_weights.shape}')
            aw = attention_weights.numpy().astype(np.float32)
            aw_print_data = (aw.min(), aw.max(), aw.mean(), aw.sum())
            print(f'aw shape: {aw.shape} aw min: %.3f, aw max: %.3f, aw mean: %.3f,aw sum: %.3f' % aw_print_data)
        
        # create attention weights (bs, layers)
        context_vector = encoder_res * attention_weights
        if debug:
            print(f'first attention weights: {attention_weights.numpy().astype(np.float32)[0,0]}')
            print(f'first encoder_res: {encoder_res.numpy().astype(np.float32)[0,0,0]}')
            print(f'first context_vector: {context_vector.numpy().astype(np.float32)[0,0,0]}')
            
            print(f'42th attention weights: {attention_weights.numpy().astype(np.float32)[0,42]}')
            print(f'42th encoder_res: {encoder_res.numpy().astype(np.float32)[0,42,42]}')
            print(f'42th context_vector: {context_vector.numpy().astype(np.float32)[0,42,42]}')
            
            print(f'encoder_res abs sum: {abs(encoder_res.numpy().astype(np.float32)).sum()}')
            print(f'context_vector abs sum: {abs(context_vector.numpy().astype(np.float32)).sum()}')
            
            print(f'encoder_res shape: {encoder_res.shape}, attention_weights shape: {attention_weights.shape}')
            print(f'context_vector shape: {context_vector.shape}')
        
        # reduce to ENCODER_DIM features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector
with tf.device('/CPU:0'):
    attention_layer = BahdanauAttention(ATTENTION_UNITS)
    context_vector, attention_weights = attention_layer(tf.zeros([BATCH_SIZE_DEBUG, DECODER_DIM]), encoder_res, debug=True)

print('context_vector shape: (batch size, units) {}'.format(context_vector.shape))
print('attention_weights shape: (batch_size, sequence_length, 1) {}'.format(attention_weights.shape))
