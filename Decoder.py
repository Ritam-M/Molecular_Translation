import tensorflow as tf

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, attention_units, encoder_dim, decoder_dim, char_embedding_dim):
        super(Decoder, self).__init__()
        
        # LSTM hidden and carry state initialization
        self.init_h = tf.keras.layers.Dense(units=decoder_dim, input_shape=[encoder_dim], name='encoder_res_to_hidden_init')
        self.init_c = tf.keras.layers.Dense(units=decoder_dim, input_shape=[encoder_dim], name='encoder_res_to_inp_act_init')
        # The LSTM cell
        self.lstm_cell = tf.keras.layers.LSTMCell(decoder_dim, name='lstm_char_predictor')
        # dropout before prediction
        self.do = tf.keras.layers.Dropout(0.30, name='prediction_dropout')
        # fully connected prediction layer
        self.fcn = tf.keras.layers.Dense(units=vocab_size, input_shape=[decoder_dim], dtype=tf.float32, name='lstm_output_to_char_probs')
        # character embedding layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, char_embedding_dim, name='character_embedding')

        # used for attention
        self.attention = BahdanauAttention(attention_units)

    def call(self, char, h, c, enc_output, training, debug=False):
        if debug:
            print(f'char shape: {char.shape}, h shape: {h.shape}, c shape: {c.shape}, enc_output shape: {enc_output.shape}')
        # embed previous character
        char = self.embedding(char, training=training)
        char = tf.squeeze(char, axis=1)
        if debug:
            print(f'char embedded and squeezed shape: {char.shape}')
        # get attention alpha and context vector
        context = self.attention(h, enc_output, training=training)

        # concat context and char to create lstm input
        lstm_input = tf.concat((context, char), axis=-1)
        if debug:
            print(f'lstm_input shape: {lstm_input.shape}')
        
        # LSTM call, get new h, c
        _, (h_new, c_new) = self.lstm_cell(lstm_input, (h, c), training=training)
        
        # compute predictions with dropout
        output = self.do(h_new, training=training)
        output = self.fcn(output, training=training)

        return output, h_new, c_new
    
    def init_hidden_state(self, encoder_out, training):
        mean_encoder_out = tf.math.reduce_mean(encoder_out, axis=1)
        h = self.init_h(mean_encoder_out, training=training)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out, training=training)
        return h, c
      
# with tf.device('/CPU:0'):
#    decoder = Decoder(VOCAB_SIZE, ATTENTION_UNITS, ENCODER_DIM, DECODER_DIM, CHAR_EMBEDDING_DIM)
#    h, c = decoder.init_hidden_state(encoder_res[:BATCH_SIZE_DEBUG], training=False)
#    preds, h, c = decoder(lbls[:BATCH_SIZE_DEBUG, :1], h, c, encoder_res, debug=True)

# print ('Decoder output shape: (batch_size, vocab size) {}'.format(preds.shape))
