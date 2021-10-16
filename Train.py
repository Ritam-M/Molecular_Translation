import tensorflow as tf
import matplotlib.pyplot as plt

# converts a dense to a sparse tensor
# sparse tensors are required to compute the Levenshtein distance
def dense_to_sparse(dense):
    ones = tf.ones(dense.shape)
    indices = tf.where(ones)
    values = tf.gather_nd(dense, indices)
    sparse = tf.SparseTensor(indices, values, dense.shape)
    
    return sparse

# computes the levenshtein distance between the predictions and labels
def get_levenshtein_distance(preds, lbls):
    preds = tf.cast(preds, tf.int64)

    preds = tf.where(tf.not_equal(preds, START_TOKEN) & tf.not_equal(preds, END_TOKEN) & tf.not_equal(preds, PAD_TOKEN), preds, y=0)
    
    lbls = strategy.gather(lbls, axis=0)
    lbls = tf.cast(lbls, tf.int64)
    lbls = tf.where(tf.not_equal(lbls, START_TOKEN) & tf.not_equal(lbls, END_TOKEN) & tf.not_equal(lbls, PAD_TOKEN), lbls, y=0)
    
    preds_sparse = dense_to_sparse(preds)
    lbls_sparse = dense_to_sparse(lbls)

    batch_distance = tf.edit_distance(preds_sparse, lbls_sparse, normalize=False)
    mean_distance = tf.math.reduce_mean(batch_distance)
    
    return mean_distance

@tf.function()
def distributed_train_step(dataset):
    # Step function
    def train_step(inp, targ):
        total_loss = 0.0

        with tf.GradientTape() as tape:
            enc_output = encoder(inp, training=True)
            h, c = decoder.init_hidden_state(enc_output, training=True)
            dec_input = tf.expand_dims(targ[:, 0], 1)

            # Teacher forcing - feeding the target as the next input
            for idx in range(1, SEQ_LEN_OUT):
                t = targ[:, idx]
                t = tf.reshape(t, [BATCH_SIZE_BASE])
                # passing enc_output to the decoder
                predictions, h, c = decoder(dec_input, h, c, enc_output, training=True)

                # update loss and train metrics
                total_loss += loss_function(t, predictions)
                train_accuracy.update_state(t, predictions)
                # using teacher forcing
                dec_input = tf.expand_dims(t, 1)

        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(total_loss, variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
        optimizer.apply_gradients(zip(gradients, variables))

        batch_loss = total_loss / (SEQ_LEN_OUT - 1)
        train_loss.update_state(batch_loss)
    
    # reset metrics
    train_loss.reset_states()
    train_accuracy.reset_states()
    # perform VERBOSE_FREQ train steps
    for _ in tf.range(tf.convert_to_tensor(VERBOSE_FREQ)):
        strategy.run(train_step, args=next(dataset))
        
def validation_step(inp, targ):
    total_loss = 0.0
    enc_output = encoder(inp, training=False)
    h, c = decoder.init_hidden_state(enc_output, training=False)
    dec_input = tf.expand_dims(targ[:, 0], 1)

    predictions_seq = tf.expand_dims(targ[:, 0], 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, SEQ_LEN_OUT):
        # passing enc_output to the decoder
        predictions, h, c = decoder(dec_input, h, c, enc_output, training=False)

        # add loss 
        # update loss and train metrics
        total_loss += loss_function(targ[:, t], predictions)
        
        # add predictions to pred_seq
        dec_input = tf.math.argmax(predictions, axis=1, output_type=tf.int32)
        dec_input = tf.expand_dims(dec_input, axis=1)
        dec_input = tf.cast(dec_input, LABEL_DTYPE)
        predictions_seq = tf.concat([predictions_seq, dec_input], axis=1)
        
    batch_loss = total_loss / (SEQ_LEN_OUT - 1)
    val_loss.update_state(batch_loss)
    
    return predictions_seq

@tf.function
def distributed_val_step(dataset):
    inp_val, targ_val = next(dataset)
    per_replica_predictions_seq = strategy.run(validation_step, args=(inp_val, targ_val))
    predictions_seq = strategy.gather(per_replica_predictions_seq, axis=0)
    
    return predictions_seq, targ_val

def get_val_metrics(val_dist_dataset):
    # reset metrics
    val_loss.reset_states()
    total_ls_distance = 0.0
    
    for step in range(VAL_STEPS):
        predictions_seq, targ = distributed_val_step(val_dist_dataset)
        levenshtein_distance = get_levenshtein_distance(predictions_seq, targ)
        total_ls_distance += levenshtein_distance
    
    return total_ls_distance / VAL_STEPS

def log(batch, t_start_batch, val_ls_distance=False):
    print(
        f'Step %s|' % f'{batch * VERBOSE_FREQ}/{TRAIN_STEPS}'.ljust(10, ' '),
        f'loss: %.3f,' % (train_loss.result() / VERBOSE_FREQ),
        f'acc: %.3f, ' % train_accuracy.result(),
    end='')
    
    if val_ls_distance:
        print(
            f'val_loss: %.3f, ' % (val_loss.result() / VERBOSE_FREQ),
            f'val lsd: %s,' % ('%.1f' % val_ls_distance).ljust(5, ' '),
        end='')
    # always end with batch duration and line break
    print(
        f'lr: %s,' % ('%.1E' % LRREDUCE.get_lr()).ljust(7),
        f't: %s sec' % int(time.time() - t_start_batch),
    )
    
class Stats():
    def __init__(self):
        self.stats = {
            'train_loss': [],
            'train_acc': [],
        }
        
    def update_stats(self):
        self.stats['train_loss'].append(train_loss.result() / VERBOSE_FREQ)
        self.stats['train_acc'].append(train_accuracy.result())
        
    def get_stats(self, metric):
        return self.stats[metric]
        
    def plot_stat(self, metric):
        plt.figure(figsize=(15,8))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.plot(self.stats[metric])
        plt.grid()
        plt.title(f'{metric} stats', size=24)
        plt.show()
        
# custom learning rate scheduler
class LRReduce():
    def __init__(self, optimizer, lr_schedule):
        self.opt = optimizer
        self.lr_schedule = lr_schedule
        # assign initial learning rate
        self.lr = lr_schedule[0]
        self.opt.learning_rate.assign(self.lr)
        
    def step(self, step):
        self.lr = self.lr_schedule[step]
        # assign learning rate to optimizer
        self.opt.learning_rate.assign(self.lr)
        
    def get_counter(self):
        return self.c
    
    def get_lr(self):
        return self.lr
