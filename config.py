"""
Deep Learning Framework
Version 1.0
Authors: Benoit Vuillemin, Frederic Bertrand
Licence: AGPL v3
"""

from Editors import *
from Encoders import *
from DataPreparators import *
from Trainers import *

# ----- CONFIGURATION -----

# Path of the input file
input_path = "Data/helpdesk.csv"
# Name of the result folder, stored inside the "Output" folder
output_name = "helpdesk"
# Size of a chunk when reading the input file, i.e. number of lines to store to memory
input_chunk_size = 50000
# Size of a chunk for pre-processed data, i.e. number of pre-processed data to store to memory
output_chunk_size = 500
# Size of a data batch for training the neural network
batch_size = 32
# Number of epoch for the neural network
epoch_counter = 5
# States if the input file has two timestamps, i.e. a start and end timestamp
double_timestamps = False
# Indexes of the columns where there are dates in the input file
dates_ids = [2]

# The mode can be online, offline or edit_db
mode = "offline"

# The steps for offline mode can be all, encode, decode, prepare, train
offline_steps = ["all"]

# List of encoders
encoders = [DeleteEncoder(0), OneHotEncoder(1, activity=True), TimeDifferenceSingleEncoder(2)]
# List of editors
editors = [SosForAll(), EosForAll()]
# Data preparator
preparator = SlicerLSTM()
# Trainer for the neural network
trainer = LSTMTrainer()
