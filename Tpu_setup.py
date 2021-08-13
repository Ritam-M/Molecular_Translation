print(f"\n... ACCELERATOR SETUP STARTING ...\n")

# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    TPU = tf.distribute.cluster_resolver.TPUClusterResolver()  
except ValueError:
    TPU = None

if TPU:
    print(f"\n... RUNNING ON TPU - {TPU.master()}...")
    tf.config.experimental_connect_to_cluster(TPU)
    tf.tpu.experimental.initialize_tpu_system(TPU)
    strategy = tf.distribute.experimental.TPUStrategy(TPU)
else:
    print(f"\n... RUNNING ON CPU/GPU ...")
    # Yield the default distribution strategy in Tensorflow
    #   --> Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy() 

# What Is a Replica?
#    --> A single Cloud TPU device consists of FOUR chips, each of which has TWO TPU cores. 
#    --> Therefore, for efficient utilization of Cloud TPU, a program should make use of each of the EIGHT (4x2) cores. 
#    --> Each replica is essentially a copy of the training graph that is run on each core and 
#        trains a mini-batch containing 1/8th of the overall batch size
N_REPLICAS = strategy.num_replicas_in_sync
    
print(f"... # OF REPLICAS: {N_REPLICAS} ...\n")

print(f"\n... ACCELERATOR SETUP COMPLTED ...\n")

print("\n... DATA ACCESS SETUP STARTED ...\n")

if TPU:
    # Google Cloud Dataset path to training and validation images
    DATA_DIR = KaggleDatasets().get_gcs_path('bms-train-tfrecords-half-length')
    TEST_DATA_DIR = KaggleDatasets().get_gcs_path('bms-test-dataset-192x384')
else:
    # Local path to training and validation images
    DATA_DIR = "/kaggle/input/bms-train-tfrecords-half-length"
    TEST_DATA_DIR = "/kaggle/input/bms-test-dataset-192x384"
    
print(f"\n... DATA DIRECTORY PATH IS:\n\t--> {DATA_DIR}")
print(f"... TEST DATA DIRECTORY PATH IS:\n\t--> {TEST_DATA_DIR}")

print(f"\n... IMMEDIATE CONTENTS OF DATA DIRECTORY IS:")
for file in tf.io.gfile.glob(os.path.join(DATA_DIR, "*")): print(f"\t--> {file}")

print(f"... IMMEDIATE CONTENTS OF TESTT DATA DIRECTORY IS:")
for file in tf.io.gfile.glob(os.path.join(TEST_DATA_DIR, "*")): print(f"\t--> {file}")

    
print("\n\n... DATA ACCESS SETUP COMPLETED ...\n")

print(f"\n... MIXED PRECISION SETUP STARTING ...\n")
print("\n... SET TF TO OPERATE IN MIXED PRECISION – `bfloat16` – IF ON TPU ...")

# Set Mixed Precision Global Policy
#     ---> To use mixed precision in Keras, you need to create a `tf.keras.mixed_precision.Policy`
#          typically referred to as a dtype policy. 
#     ---> Dtype policies specify the dtypes layers will run in
tf.keras.mixed_precision.set_global_policy('mixed_bfloat16' if TPU else 'float32')

# target data type, bfloat16 when using TPU to improve throughput
TARGET_DTYPE = tf.bfloat16 if TPU else tf.float32
print(f"\t--> THE TARGET DTYPE HAS BEEN SET TO {TARGET_DTYPE} ...")

# The policy specifies two important aspects of a layer: 
#     1. The dtype the layer's computations are done in
#     2. The dtype of a layer's variables. 
print(f"\n... TWO IMPORTANT ASPECTS OF THE GLOBAL MIXED PRECISION POLICY:")
print(f'\t--> COMPUTE DTYPE  : {tf.keras.mixed_precision.global_policy().compute_dtype}')
print(f'\t--> VARIABLE DTYPE : {tf.keras.mixed_precision.global_policy().variable_dtype}')

print(f"\n\n... MIXED PRECISION SETUP COMPLTED ...\n")

print(f"\n... XLA OPTIMIZATIONS STARTING ...\n")

print(f"\n... CONFIGURE JIT (JUST IN TIME) COMPILATION ...\n")
# enable XLA optmizations (10% speedup when using @tf.function calls)
tf.config.optimizer.set_jit(True)

print(f"\n... XLA OPTIMIZATIONS COMPLETED ...\n")

print("\n... BASIC DATA SETUP STARTING ...\n")

# All the possible tokens in our InChI 'language'
TOKEN_LIST = ["<PAD>", "InChI=1S/", "<END>", "/c", "/h", "/m", "/t", "/b", "/s", "/i"] +\
             ['Si', 'Br', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', 'C', 'H', 'B', ] +\
             [str(i) for i in range(167,-1,-1)] +\
             ["\+", "\(", "\)", "\-", ",", "D", "T"]
print(f"\n... TOKEN LIST:")
for i, tok in enumerate(TOKEN_LIST): print(f"\t--> INTEGER-IDX = {i:<3}  –––  STRING = {tok}")

# The start/end/pad tokens will be removed from the string when computing the Levenshtein distance
# We want them as tf.constant's so they will operate properly within the @tf.function context
START_TOKEN = tf.constant(TOKEN_LIST.index("InChI=1S/"), dtype=tf.uint8)
END_TOKEN = tf.constant(TOKEN_LIST.index("<END>"), dtype=tf.uint8)
PAD_TOKEN = tf.constant(TOKEN_LIST.index("<PAD>"), dtype=tf.uint8)

# Prefixes and Their Respective Ordering/Format
#      -- ORDERING --> {c}{h/None}{b/None}{t/None}{m/None}{s/None}{i/None}{h/None}{t/None}{m/None}
PREFIX_ORDERING = "chbtmsihtm"
print(f"\n... PREFIX ORDERING IS {PREFIX_ORDERING} ...")

# Paths to Respective Image Directories
TRAIN_DIR = os.path.join(DATA_DIR, "train_records")
VAL_DIR = os.path.join(DATA_DIR, "val_records")
TEST_DIR = os.path.join(TEST_DATA_DIR, "test_records")

# Get the Full Paths to The Individual TFRecord Files
TRAIN_TFREC_PATHS = sorted(
    tf.io.gfile.glob(os.path.join(TRAIN_DIR, "*.tfrec")), 
    key=lambda x: int(x.rsplit("_", 2)[1]))
VAL_TFREC_PATHS = sorted(
    tf.io.gfile.glob(os.path.join(VAL_DIR, "*.tfrec")), 
    key=lambda x: int(x.rsplit("_", 2)[1]))
TEST_TFREC_PATHS = sorted(
    tf.io.gfile.glob(os.path.join(TEST_DIR, "*.tfrec")), 
    key=lambda x: int(x.rsplit("_", 2)[1]))

print(f"\n... TFRECORD INFORMATION:")
for SPLIT, TFREC_PATHS in zip(["TRAIN", "VAL", "TEST"], [TRAIN_TFREC_PATHS, VAL_TFREC_PATHS, TEST_TFREC_PATHS]):
    print(f"\t--> {len(TFREC_PATHS):<3} {SPLIT:<5} TFRECORDS")

# Paths to relevant CSV files containing training and submission information
TRAIN_CSV_PATH = os.path.join("/kaggle/input", "bms-csvs-w-extra-metadata", "train_labels_w_extra.csv")
SS_CSV_PATH    = os.path.join("/kaggle/input", "bms-csvs-w-extra-metadata", "sample_submission_w_extra.csv")
print(f"\n... PATHS TO CSVS:")
print(f"\t--> TRAIN CSV: {TRAIN_CSV_PATH}")
print(f"\t--> SS CSV   : {SS_CSV_PATH}")

# When debug is true we use a smaller batch size and smaller model
DEBUG=False

print("\n\n... BASIC DATA SETUP COMPLETED ...\n")

print("\n... INITIAL DATAFRAME INSTANTIATION STARTING ...\n")

# Load the train and submission dataframes
train_df = pd.read_csv(TRAIN_CSV_PATH)
ss_df    = pd.read_csv(SS_CSV_PATH)

# --- Distribution Information ---
N_EX    = len(train_df)
N_TEST  = len(ss_df)
N_VAL   = 80_000 # Fixed from dataset creation information
N_TRAIN = N_EX-N_VAL

# --- Batching Information ---
DEBUG=False
BATCH_SIZE_DEBUG   = 2
REPLICA_BATCH_SIZE = 128 # Could probably be 128

if DEBUG:
    REPLICA_BATCH_SIZE = BATCH_SIZE_DEBUG
OVERALL_BATCH_SIZE = REPLICA_BATCH_SIZE*N_REPLICAS


# --- Input Image Information ---
IMG_SHAPE = (192,384,3)

# --- Autocalculate Training/Validation/Testing Information ---
TRAIN_STEPS = N_TRAIN  // OVERALL_BATCH_SIZE
VAL_STEPS   = N_VAL    // OVERALL_BATCH_SIZE
TEST_STEPS  = int(np.ceil(N_TEST/OVERALL_BATCH_SIZE))

# This is for padding our test dataset so we only have whole batches
REQUIRED_DATASET_PAD = OVERALL_BATCH_SIZE-N_TEST%OVERALL_BATCH_SIZE

# --- Modelling Information ---
ATTN_EMB_DIM  = 192
N_RNN_UNITS   = 512

print(f"\n... # OF TRAIN+VAL EXAMPLES  : {N_EX:<7} ...")
print(f"... # OF TRAIN EXAMPLES      : {N_TRAIN:<7} ...")
print(f"... # OF VALIDATION EXAMPLES : {N_VAL:<7} ...")
print(f"... # OF TEST EXAMPLES       : {N_TEST:<7} ...\n")

print(f"\n... REPLICA BATCH SIZE    : {REPLICA_BATCH_SIZE} ...")
print(f"... OVERALL BATCH SIZE    : {OVERALL_BATCH_SIZE} ...\n")

print(f"\n... IMAGE SHAPE           : {IMG_SHAPE} ...\n")

print(f"\n... TRAIN STEPS PER EPOCH : {TRAIN_STEPS:<5} ...")
print(f"... VAL STEPS PER EPOCH   : {VAL_STEPS:<5} ...")
print(f"... TEST STEPS PER EPOCH  : {TEST_STEPS:<5} ...\n")

print("\n... TRAIN DATAFRAME ...\n")
display(train_df.head(3))

print("\n... SUBMISSION DATAFRAME ...\n")
display(ss_df.head(3))

print("\n... INITIAL DATAFRAME INSTANTIATION COMPLETED...\n")

print("\n... SPECIAL VARIABLE SETUP STARTING ...\n")


# Whether to start training using previously checkpointed model
LOAD_MODEL        = True
ENCODER_CKPT_PATH = "/kaggle/input/bms-efficientnetv2-tpu-e2e-pipeline-in-3hrs/encoder_epoch_12.h5"
TRANSFORMER_CKPT_PATH = ""

if LOAD_MODEL:
    if TRANSFORMER_CKPT_PATH != "":
        print(f"... TRANSFORMER MODEL TRAINING WILL RESUME FROM PREVIOUS CHECKPOINT:\n\t-->{TRANSFORMER_CKPT_PATH}\n")
    elif ENCODER_CKPT_PATH != "":
        print(f"\n... ENCODER MODEL TRAINING WILL RESUME FROM PREVIOUS CHECKPOINT:\n\t-->{ENCODER_CKPT_PATH}\n")    
    else:
        print(f"\n... MODEL TRAINING WILL START FROM SCRATCH ...\n")
else:
    print(f"\n... MODEL TRAINING WILL START FROM SCRATCH ...\n")
    
print("\n... SPECIAL VARIABLE SETUP COMPLETED ...\n")
