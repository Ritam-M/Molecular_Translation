def get_val_dataset(bs=BATCH_SIZE):
    FNAMES_TRAIN_TFRECORDS = tf.io.gfile.glob(f'{GCS_DS_PATH}/val/*.tfrecords')
    val_dataset = tf.data.TFRecordDataset(FNAMES_TRAIN_TFRECORDS, num_parallel_reads=AUTO)
    val_dataset = val_dataset.prefetch(AUTO)
    val_dataset = val_dataset.repeat()
    val_dataset = val_dataset.map(decode_tfrecord, num_parallel_calls=AUTO)
    val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)
    val_dataset = val_dataset.prefetch(1)
    
    return val_dataset

val_dataset = get_val_dataset()
benchmark_dataset(val_dataset)

val_imgs, val_lbls = next(iter(val_dataset))
print(f'val_imgs.shape: {val_imgs.shape}, val_lbls.shape: {val_lbls.shape}')
val_img0 = val_imgs[0].numpy().astype(np.float32)
val_batch_info = (val_img0.mean(), val_img0.std(), val_img0.min(), val_img0.max(), val_imgs.dtype)
print('val img 0 mean: %.3f, 0 std: %.3f, min: %.3f, max: %.3f, %s' % train_batch_info)

show_batch(val_dataset)

