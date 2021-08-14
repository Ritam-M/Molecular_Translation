STATS = Stats()
LRREDUCE = LRReduce(optimizer, LR_SCHEDULE)

step_total = 0
for epoch in range(EPOCHS):
    print(f'***** EPOCH {epoch + 1} *****')
    
    t_start = time.time()
    t_start_batch = time.time()
    total_loss = 0
    
    # create distributed versions of dataset
    train_dist_dataset = iter(strategy.experimental_distribute_dataset(train_dataset))
    val_dist_dataset = iter(strategy.experimental_distribute_dataset(val_dataset))

    for step in range(1, STEPS_PER_EPOCH + 1):
        # train step
        distributed_train_step(train_dist_dataset)
        STATS.update_stats()
        # save epoch weights
        if epoch >= 16:
            encoder.save_weights(f'./encoder_epoch_{epoch+1}.h5')
            decoder.save_weights(f'./decoder_epoch_{epoch+1}.h5')

        # end of epoch validation
        if step == STEPS_PER_EPOCH:
            val_ls_distance = get_val_metrics(val_dist_dataset)
            # log with validation
            log(step, t_start_batch, val_ls_distance)
        else:
            # normal log
            log(step, t_start_batch)
            # reset start time batch
            t_start_batch = time.time()
            
        total_loss += train_loss.result()
        
        # learning rate step
        LRREDUCE.step(epoch * TRAIN_STEPS + step * VERBOSE_FREQ - 1)
        
        # stop training when NaN loss is detected, this can be caused by exploding gradients
        if np.isnan(total_loss):
            break
            
    # stop training when NaN loss is detected
    if np.isnan(total_loss):
        break

    print(f'Epoch {epoch} Loss {round(total_loss.numpy() / TRAIN_STEPS, 3)}, time: {int(time.time() - t_start)} sec\n')
   

STATS.plot_stat('train_loss')
STATS.plot_stat('train_acc')
