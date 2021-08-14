# decodes TFRecord
def decode_tfrecord(record_bytes):
    features = tf.io.parse_single_example(record_bytes, {
        'image': tf.io.FixedLenFeature([], tf.string),
        'InChI': tf.io.FixedLenFeature([MAX_INCHI_LEN], tf.int64),
    })

    # decode the PNG and explicitly reshape to image size (required on TPU)
    image = tf.io.decode_png(features['image'])    
    image = tf.reshape(image, [IMG_HEIGHT, IMG_WIDTH, 1])
    # normalize according to ImageNet mean and std
    image = tf.cast(image, tf.float32)  / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    
    if TPU: # if running on TPU image needs to be cast to bfloat16
        image = tf.cast(image, TARGET_DTYPE)
    
    InChI = tf.reshape(features['InChI'], [MAX_INCHI_LEN])
    InChI = tf.cast(InChI, LABEL_DTYPE)
    
    return image, InChI

# Benchmark function to test the dataset throughput performance
def benchmark_dataset(dataset, num_epochs=3, n_steps_per_epoch=25, bs=BATCH_SIZE):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        epoch_start = time.perf_counter()
        for idx, (images, labels) in enumerate(dataset.take(n_steps_per_epoch)):
            if idx is 1 and epoch_num is 0:
                print(f'image shape: {images.shape}, image dtype: {images.dtype}')
                print(f'labels shape: {labels.shape}, label dtype: {labels.dtype}')
            pass
        epoch_t = time.perf_counter() - epoch_start
        mean_step_t = round(epoch_t / n_steps_per_epoch * 1000, 1)
        n_imgs_per_s = int(1 / (mean_step_t / 1000) * bs)
        print(f'epoch {epoch_num} took: {round(epoch_t, 2)} sec, mean step duration: {mean_step_t}ms, images/s: {n_imgs_per_s}')
        
# plots the first images of the dataset
def show_batch(dataset, rows=3, cols=2):
    imgs, lbls = next(iter(dataset))
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols*7, rows*4))
    for r in range(rows):
        for c in range(cols):
            img = imgs[r*cols+c].numpy().astype(np.float32)
            img += abs(img.min())
            img /= img.max()
            axes[r, c].imshow(img)

def get_train_dataset(bs=BATCH_SIZE):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    
    FNAMES_TRAIN_TFRECORDS = tf.io.gfile.glob(f'{GCS_DS_PATH}/train/*.tfrecords')
    train_dataset = tf.data.TFRecordDataset(FNAMES_TRAIN_TFRECORDS, num_parallel_reads=AUTO)
    train_dataset = train_dataset.with_options(ignore_order)
    train_dataset = train_dataset.prefetch(AUTO) # optimize automatically
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.map(decode_tfrecord, num_parallel_calls=AUTO)  # optimize automatically
    train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)
    train_dataset = train_dataset.prefetch(1) # just 1 prefetched batch is needed
    
    return train_dataset
    
train_dataset = get_train_dataset()
benchmark_dataset(train_dataset)

# display statistics about the first image to check if the images are decoded correctly
imgs, lbls = next(iter(train_dataset))
print(f'imgs.shape: {imgs.shape}, lbls.shape: {lbls.shape}')
img0 = imgs[0].numpy().astype(np.float32)
train_batch_info = (img0.mean(), img0.std(), img0.min(), img0.max(), imgs.dtype)
print('train img 0 mean: %.3f, 0 std: %.3f, min: %.3f, max: %.3f, %s' % train_batch_info)

# show first few train images
show_batch(train_dataset)
