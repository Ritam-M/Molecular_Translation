class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        
        # output: (bs, 1280, 14, 8)
        self.feature_maps = efn.EfficientNetB2(include_top=False, weights='noisy-student')
        # set global encoder dimension variable
        global ENCODER_DIM
        ENCODER_DIM = self.feature_maps.layers[-1].output_shape[-1]
        
        # output: (bs, 1280, 112)
        self.reshape = tf.keras.layers.Reshape([-1, ENCODER_DIM], name='reshape_featuere_maps')

    def call(self, x, training, debug=False):
        x = self.feature_maps(x, training=training)
        if debug:
            print(f'feature maps shape: {x.shape}')
            
        x = self.reshape(x, training=training)
        if debug:
            print(f'feature maps reshaped shape: {x.shape}')
        
        return x
    
# Example enoder output
with tf.device('/CPU:0'):
    encoder = Encoder()
    encoder_res = encoder(imgs[:BATCH_SIZE_DEBUG], debug=True)

print ('Encoder output shape: (batch size, sequence length, units) {}'.format(encoder_res.shape))
