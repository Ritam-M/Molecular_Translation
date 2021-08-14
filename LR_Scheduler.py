def lrfn(step, WARMUP_LR_START, LR_START, LR_FINAL, DECAYS):
    # exponential warmup
    if step < WARMUP_STEPS:
        warmup_factor = (step / WARMUP_STEPS) ** 2
        lr = WARMUP_LR_START + (LR_START - WARMUP_LR_START) * warmup_factor
    # staircase decay
    else:
        power = (step - WARMUP_STEPS) // ((TOTAL_STEPS - WARMUP_STEPS) / (DECAYS + 1))
        decay_factor =  ((LR_START / LR_FINAL) ** (1 / DECAYS)) ** power
        lr = LR_START / decay_factor

    return round(lr, 8)

# plot the learning rate schedule
def plot_lr_schedule(lr_schedule, name):
    plt.figure(figsize=(15,8))
    plt.plot(lr_schedule)
    schedule_info = f'start: {lr_schedule[0]}, max: {max(lr_schedule)}, final: {lr_schedule[-1]}'
    plt.title(f'Step Learning Rate Schedule {name}, {schedule_info}', size=16)
    plt.grid()
    plt.show()

# Training configuration
EPOCHS = 32
WARMUP_STEPS = 500
TRAIN_STEPS = 1000
VERBOSE_FREQ = 100
STEPS_PER_EPOCH = TRAIN_STEPS // VERBOSE_FREQ
TOTAL_STEPS = EPOCHS * TRAIN_STEPS

# Learning rate for encoder
LR_SCHEDULE = [lrfn(step, 1e-8, 2e-3, 1e-4 ,EPOCHS) for step in range(TOTAL_STEPS)]
plot_lr_schedule(LR_SCHEDULE, 'Ecnoder')
