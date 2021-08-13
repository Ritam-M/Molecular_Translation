import tensorflow as tf

def flatten_l_o_l(nested_list):
    """ Flatten a list of lists """
    return [item for sublist in nested_list for item in sublist]


def tf_load_image(path, img_size=(192,384,3), invert=False):
    """ Load an image with the correct size and shape 
    
    Args:
        path (tf.string): Path to the image to be loaded
        img_size (tuple, optional): Size to reshape image to (required for TPU)
        tile_to_3_channel (bool, optional): Whether to tile the single channel
            image to 3 channels which will be required for most off-the-shelf models
        invert (bool, optional): Whether or not to invert the background/foreground
    
    Returns:
        3 channel tf.Constant image ready for training/inference
    
    """
    img = decode_img(tf.io.read_file(path), img_size, n_channels=3, invert=invert)        
    return img
    
    
def decode_image(image_data, resize_to=(192,384,3)):
    """ Function to decode the tf.string containing image information 
    
    
    Args:
        image_data (tf.string): String containing encoded image data from tf.Example
        resize_to (tuple, optional): Size that we will reshape the tensor to (required for TPU)
    
    Returns:
        Tensor containing the resized single-channel image in the appropriate dtype
    """
    image = tf.image.decode_png(image_data, channels=3)
    image = tf.reshape(image, resize_to)
    return tf.cast(image, TARGET_DTYPE)
    
    
# sparse tensors are required to compute the Levenshtein distance
def dense_to_sparse(dense):
    """ Convert a dense tensor to a sparse tensor 
    
    Args:
        dense (Tensor): TBD
        
    Returns:
        A sparse tensor    
    """
    indices = tf.where(tf.ones_like(dense))
    values = tf.reshape(dense, (MAX_LEN*OVERALL_BATCH_SIZE,))
    sparse = tf.SparseTensor(indices, values, dense.shape)
    return sparse

def get_levenshtein_distance(preds, lbls):
    """ Computes the Levenshtein distance between the predictions and labels 
    
    Args:
        preds (tensor): Batch of predictions
        lbls (tensor): Batch of labels
        
    Returns:
        The mean Levenshtein distance calculated across the batch
    """
    preds = tf.where(tf.not_equal(lbls, END_TOKEN) & tf.not_equal(lbls, PAD_TOKEN), preds, 0)
    lbls = tf.where(tf.not_equal(lbls, END_TOKEN), lbls, 0)

    preds_sparse = dense_to_sparse(preds)
    lbls_sparse = dense_to_sparse(lbls)

    batch_distance = tf.edit_distance(preds_sparse, lbls_sparse, normalize=False)
    mean_distance = tf.math.reduce_mean(batch_distance)
    
    return mean_distance
