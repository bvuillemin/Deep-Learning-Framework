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

# ----- DEFAULT CONFIGURATION -----

# Path of the input file
input_path = "Data/helpdesk.csv"
# Path of the co-variable file. Only used with the edit_db to merge with the input file
cov_path = ""
# Name of the result folder, stored inside the "Output" folder
output_name = "helpdesk"
# Size of a chunk when reading the input file, i.e. number of lines to store to memory
input_chunk_size = 50000
# Size of a chunk for pre-processed data, i.e. number of data that will go to the neural network to store to memory
output_chunk_size = 500
# Number of epoch for the neural network
epoch_counter = 5
# States if the input file has two timestamps, i.e. a start and end timestamp
double_timestamps = False
# Indexes of the columns where there are dates in the input file
dates_ids = [2]

# States if the co-variables must be considered or not
consider_cov = False
# States if encoders must be automatically created according to the data types stored inside columns.
# Works only for co-variable columns, not the core columns (case id, activity, start timestamp (or end timestamp))
auto_build_cov_encoders = True

# Set it to true if you want to load a previously made orchestrator
orchestrator_from_file = False

# The mode can be online, offline or edit_db
mode = "offline"

# The steps for offline mode can be all, encode, decode, prepare, train
offline_steps = ["all"]
# Get a debug file from the encoding (human readable csv file)
debug = False

# List of encoders
encoders = [DeleteEncoder(0), OneHotEncoder(1, activity=True), TimeDifferenceSingleEncoder(2)]
# List of co-variables encoders
cov_encoders = []
# List of editors
editors = [SosForAll(), EosForAll()]
# Data preparator
preparator = SlicerLSTM()
# Trainer for the neural network
trainer = LSTMTrainer()
