# convert the integer encoded predictions to a string
def int2char(i_str):
    res = ''
    for i in i_str.numpy():
        c = int_to_vocabulary.get(i)
        if c not in ['<start>', '<end>', '<pad>']:
            res += c
    return res

def evaluate(img, actual=None):
    # get encoder output and initiate LSTM hidden and carry state
    enc_out = encoder(tf.expand_dims(img, axis=0), training=False)
    h, c = decoder.init_hidden_state(enc_out, training=False)
    
    # the "<start>" token is used as first character when predicting
    dec_input = tf.expand_dims([vocabulary_to_int.get('<start>')], 0)
    result = ''
    
    for t in tqdm(range(SEQ_LEN_OUT)):
        predictions, h, c = decoder(dec_input, h, c, enc_out, training=False)
        predicted_id = tf.argmax(predictions[0]).numpy()
        predicted_char = int_to_vocabulary.get(predicted_id)

        # stop predicting when "<end>" token is predicted
        if predicted_char == '<end>':
            break
        
        # add every character except "<start>"
        if result != '<start>':
            result += predicted_char

        # predicted charachter is used as input to the decoder to predict the next character
        dec_input = tf.expand_dims([predicted_id], 0)

    # plot the molecule image
    plt.figure(figsize=(7, 4))
    plt.imshow(img.numpy().astype(np.float32))
    plt.show()
    print(f'predicted: \t{result}')
    print(f'actual: \t{int2char(actual)}')

for n in range(3):
    evaluate(val_imgs[n], actual=val_lbls[n])
