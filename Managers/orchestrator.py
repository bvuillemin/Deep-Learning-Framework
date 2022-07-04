"""
Deep Learning Framework
Version 1.0
Authors: Benoit Vuillemin, Frederic Bertrand
Licence: AGPL v3
"""

import csv

import numpy as np
import pandas as pd
from tqdm import tqdm

from generic_functions import remove_nan, get_cases_info, get_complete_cases, create_directories, get_names, \
    merge_data_cov
from Managers.editor_manager import EditorManager
from Managers.encoder_manager import EncoderManager
from Encoders import *


class Orchestrator:
    def __init__(self, encoder_manager, editor_manager=None) -> None:
        """
        Orchestrator: combination of an encoder manager, an editor manager and some infos about the input file

        :param encoder_manager: Encoder manager
        :type encoder_manager: EncoderManager
        :param editor_manager: Editors manager
        :type editor_manager: EditorManager
        """
        self.encoder_manager = encoder_manager
        self.editor_manager = editor_manager
        self.input_path = None
        self.output_name = None
        self.column_names = None
        self.dates_ids = None
        self.case_counter = None
        self.total_chunk_counter = None
        self.double_timestamps = None
        self.activity_counter = None
        self.max_case_length = None
        self.features_counter = len(encoder_manager.all_output_column_names)
        self.has_leftovers = len(encoder_manager.get_leftover_names()) > 0
        self.encoder_counter = encoder_manager.get_encoder_counter()
        self.encoder_descriptions = encoder_manager.get_all_encoders_description()

    def insert_infos(self, input_path, output_name, column_names, dates_ids, case_counter, total_chunk_counter,
                     double_timestamps, activity_counter, max_case_length) -> None:
        """
        Description of the input data file

        :param input_path: Name of the input file
        :type input_path: str
        :param output_name: Name of the output folder
        :type output_name: str
        :param case_counter: Total number of cases to have been processed
        :type case_counter: int
        :param column_names: Names of the columns of the file
        :type column_names: np.ndarray
        :param dates_ids: Indexes of the dates columns (for pandas to parse those dates as Timestamps, not strings)
        :type dates_ids: list
        :param total_chunk_counter: Total number of chunks to read a file
        :type total_chunk_counter: int
        :param double_timestamps: Defines if the file had two timestamps (only used for data files)
        :type double_timestamps: bool
        :param activity_counter: Total number of activities inside cases
        :type activity_counter: int
        :param max_case_length: Maximum length of a case
        :type max_case_length: int
        """
        super().__init__()
        self.input_path = input_path
        self.output_name = output_name
        self.column_names = column_names
        self.dates_ids = dates_ids
        self.case_counter = case_counter
        self.total_chunk_counter = total_chunk_counter
        self.double_timestamps = double_timestamps
        self.activity_counter = activity_counter
        self.max_case_length = max_case_length
        self.features_counter = len(self.encoder_manager.all_output_column_names)
        self.has_leftovers = len(self.encoder_manager.get_leftover_names()) > 0
        self.encoder_counter = self.encoder_manager.get_encoder_counter()
        self.encoder_descriptions = self.encoder_manager.get_all_encoders_description()

    def show_infos(self):
        print("--------Orchestrator--------")
        print("Database file name:", self.input_path)
        print("Ouput folder:", self.output_name)
        print("Database column names:", self.column_names.values.tolist())
        print("Indexes of the dates columns:", self.dates_ids)
        print("Number of cases:", self.case_counter)
        print("Number of chunks:", self.total_chunk_counter)
        # Exclusive to data file
        print("Is it a double timestamps file?", self.double_timestamps)
        print("Number of activities:", self.activity_counter)
        print("Maximum length of a case:", self.max_case_length)
        print("Number of features of the neural network:", self.features_counter)
        print("Are there leftovers?", self.has_leftovers)
        print("Number of encoders:", self.encoder_counter)
        print("---------------------------")

    def alter_internal_infos(self):
        """
        Allows the encoders to tamper with the internal orchestrator infos
        
        """
        for editor in self.editor_manager.editors:
            editor.alter_orchestrator_infos(self)

    def alter_encoders_descriptions(self):
        """
        Allows the orchestrator to temper with the internal descriptions of its encoders

        """
        if self.editor_manager:
            for encoder in self.encoder_manager.encoders:
                encoder.tamper(self.editor_manager.activities_to_add)

    def process_cases(self, cases, edit_db=False):
        """
        Edits and encodes the input cases

        :param cases: Cases to process
        :type cases: list
        :param edit_db: Defines if a new csv with the edited data must be built. Else, a numpy file will be built
        :type edit_db: bool
        :return: Encoded cases and their leftovers
        :rtype: (np.ndarray, np.ndarray)
        """
        encoded_cases = []
        leftovers = []
        for case in cases:
            np_case = case.to_numpy()
            if self.editor_manager:
                np_case = self.editor_manager.edit_case(np_case, self)
            if edit_db:
                encoded_cases.append(np_case)
            else:
                encoded_case = self.encoder_manager.encode_case(np_case)
                leftover = self.encoder_manager.get_leftover(np_case)
                encoded_cases.append(encoded_case)
                leftovers.append(leftover)
        leftovers = np.asarray(leftovers)
        return encoded_cases, leftovers

    def process_case(self, case):
        """
        Edits and encodes the input case

        :param case: Case to process
        :type case: pd.Dataframe
        :return: Encoded cases and their leftovers
        :rtype: (np.ndarray, np.ndarray)
        """
        np_case = case.to_numpy()
        if self.editor_manager:
            np_case = self.editor_manager.edit_case(np_case, self)
        encoded_case = self.encoder_manager.encode_case(np_case)
        leftover = self.encoder_manager.get_leftover(np_case)
        leftover = np.asarray(leftover)
        return encoded_case, leftover

    def save_to_file(self):
        """
        Saves all the infos from the orchestrator to a file

        """
        create_directories(self.output_name)
        writer = csv.writer(open("Output/" + self.output_name + "/desc_data.csv", 'w', newline=''))
        writer.writerow(["Name of the input file", self.input_path])
        writer.writerow(["Name of the output folder", self.output_name])
        writer.writerow(["Column names of the file"])
        writer.writerow(self.column_names)
        writer.writerow(["IDs of all the dates columns"])
        writer.writerow(self.dates_ids)
        writer.writerow(["Number of cases", str(self.case_counter)])
        writer.writerow(["Total number of chunks", str(self.total_chunk_counter)])
        writer.writerow(["Is it a double timestamps file ?", str(self.double_timestamps)])
        writer.writerow(["Number of activities", self.activity_counter])
        writer.writerow(["Maximum length of a case", self.max_case_length])
        writer.writerow(["Number of features", self.features_counter])
        writer.writerow(["Has leftovers", self.has_leftovers])
        writer.writerow(["Editors"])
        writer.writerow(self.editor_manager.get_editors_names())
        writer.writerow(["Encoders", self.encoder_counter])
        for info in self.encoder_descriptions:
            writer.writerows(info)

    def init_from_data(self, input_path, output_name, input_chunk_size, double_timestamps, dates_ids=None):
        """
        Reads the input file to generate the internal representations of all the encoders. The file is loaded by
        chunks, to avoid an overflow in the RAM

        :param input_path: Name of the file to read
        :type input_path: str
        :param output_name: Name of the output file
        :type output_name: str
        :param input_chunk_size: Maximum number of lines stored inside a chunk
        :type input_chunk_size: int
        :param double_timestamps: Defines if the file had two timestamps (only used for data files)
        :type double_timestamps: bool
        :param dates_ids: Indexes of the dates columns (for pandas to parse those dates as Timestamps, not strings)
        :type dates_ids: list
        :return: Total number of chunks, number of activities, number of cases, maximum length of a case, names of all
        columns of the data file
        :rtype: (int, int, int, int, np.ndarray)
        """
        chunks = pd.read_csv(input_path, chunksize=input_chunk_size)
        total_chunk_counter = 0
        for _ in chunks:
            total_chunk_counter += 1
        if dates_ids:
            chunks = pd.read_csv(input_path, chunksize=input_chunk_size, parse_dates=dates_ids)
        else:
            chunks = pd.read_csv(input_path, chunksize=input_chunk_size)
        activity_column = 1
        first_chunk = True
        chunk_counter = 0
        column_names = np.asarray([])
        case_counter = 0
        max_case_length = 0
        previous_id = None
        previous_size = None
        for og_chunk in tqdm(chunks, total=total_chunk_counter, desc="Analyze data"):
        # for og_chunk in chunks:
            chunk = remove_nan(og_chunk)
            last_chunk = chunk_counter == total_chunk_counter - 1
            if first_chunk:
                # Get the names of the columns of the CSV (used to reference the correct columns after)
                column_names = chunk.columns
                for encoder in self.encoder_manager.encoders:
                    encoder.set_column_names(column_names)
            previous_id, previous_size, case_counter, max_case_length = \
                get_cases_info(chunk, column_names[0], first_chunk, last_chunk, previous_id, previous_size,
                               case_counter, max_case_length)
            for encoder in self.encoder_manager.encoders:
                encoder.update_encoder(chunk)
            chunk_counter += 1
            if first_chunk:
                first_chunk = False
        for encoder in self.encoder_manager.encoders:
            if encoder.column_id == activity_column:
                activity_encoder = encoder
            encoder.finalize()
        self.encoder_manager.set_all_output_column_names()
        activity_counter = len(activity_encoder.output_column_names)
        self.insert_infos(input_path, output_name, column_names, dates_ids, case_counter, total_chunk_counter,
                          double_timestamps, activity_counter, max_case_length)

    def edit_online(self, input_chunk_size, output_chunk_size):
        """
        Only edits the cases from the raw data.

        Here, we assume the file is built as follows, with the following columns in this order:

        - Case ID
        - Activity
        - Date (or start date)
        - End date (if exists)

        :param input_chunk_size: Number of lines by chunk, used if the database is too big
        :type input_chunk_size: int
        :param output_chunk_size: Number of cases by chunk, used if the database is too big
        :type output_chunk_size: int
        """
        # Get the list of chunks
        chunks = pd.read_csv(self.input_path, chunksize=input_chunk_size, parse_dates=self.dates_ids)
        # Create all preliminary data before the chunks are processed
        id_column = self.column_names[0]
        previous_case = None
        previous_case_id = ""
        first_chunk = True
        chunk_counter = 0
        all_cases = []
        case_counter = 0
        # Process the chunk and record them
        for og_chunk in chunks:
            chunk_counter += 1
            chunk = remove_nan(og_chunk)
            last_chunk = chunk_counter == self.total_chunk_counter
            complete_cases, case_ids, previous_case, previous_case_id = \
                get_complete_cases(chunk, id_column, first_chunk, last_chunk, previous_case, previous_case_id)
            case_counter += len(complete_cases)
            if len(complete_cases) > 0:
                modified_cases, leftovers = self.process_cases(complete_cases, edit_db=False)
                for i in range(len(modified_cases)):
                    all_cases.append(modified_cases[i])
                    if len(all_cases) == output_chunk_size:
                        yield all_cases
                        all_cases = []
            if first_chunk:
                first_chunk = False
        yield all_cases

    def process_online(self, input_chunk_size):
        """
        Converts the raw data into interpretable data for the neural network.

        Here, we assume the file is built as follows, with the following columns in this order:

        - Case ID
        - Activity
        - Date (or start date)
        - End date (if exists)

        :param input_chunk_size: Number of lines by chunk, used if the database is too big
        :type input_chunk_size: int
        """
        # Get the list of chunks
        chunks = pd.read_csv(self.input_path, chunksize=input_chunk_size, parse_dates=self.dates_ids)
        # Create all preliminary data before the chunks are processed
        id_column = self.column_names[0]
        previous_case = None
        previous_case_id = ""
        first_chunk = True
        chunk_counter = 0
        case_counter = 0
        # Process the chunk and record them
        for og_chunk in chunks:
            chunk_counter += 1
            chunk = remove_nan(og_chunk)
            last_chunk = chunk_counter == self.total_chunk_counter
            complete_cases, case_ids, previous_case, previous_case_id = \
                get_complete_cases(chunk, id_column, first_chunk, last_chunk, previous_case, previous_case_id)
            case_counter += len(complete_cases)
            for case in complete_cases:
                modified_case, leftover = self.process_case(case)
                yield modified_case, leftover
            if first_chunk:
                first_chunk = False

    def process_offline(self, input_chunk_size, edit_db=False, cov_path=None, debug=False):
        """
        Converts the raw data into interpretable data for the neural network.

        Here, we assume the file is built as follows, with the following columns in this order:

        - Case ID
        - Activity
        - Date (or start date)
        - End date (if exists)

        :param input_chunk_size: Number of lines by chunk, used if the database is too big
        :type input_chunk_size: int
        :param edit_db: Defines if a new csv with the edited data must be built. Else, a numpy file will be built
        :type edit_db: bool
        """
        create_directories(self.output_name)
        # Get the list of chunks
        chunks = pd.read_csv(self.input_path, chunksize=input_chunk_size, parse_dates=self.dates_ids)
        # Create all preliminary data before the chunks are processed
        id_column = self.column_names[0]
        previous_case = None
        previous_case_id = ""
        first_chunk = True
        chunk_counter = 0
        case_counter = 0
        # Process the chunk and record them
        for og_chunk in tqdm(chunks, total=self.total_chunk_counter, desc='Encode data'):
        #for og_chunk in chunks:
            chunk = remove_nan(og_chunk)
            chunk_counter += 1
            last_chunk = chunk_counter == self.total_chunk_counter
            complete_cases, case_ids, previous_case, previous_case_id = \
                get_complete_cases(chunk, id_column, first_chunk, last_chunk, previous_case, previous_case_id)
            encoded_data, leftovers = self.process_cases(complete_cases, edit_db)
            counter = len(complete_cases)
            if cov_path:
                cov_chunk = pd.read_csv(cov_path, skiprows=case_counter, nrows=len(complete_cases)).to_numpy()
                encoded_data, counter = merge_data_cov(encoded_data, cov_chunk)
            case_counter += counter
            self.save_chunk_to_file(encoded_data, leftovers, first_chunk, edit_db, debug)
            if first_chunk:
                first_chunk = False

    def save_chunk_to_file(self, encoded_chunk, leftovers, first_chunk, edit_db, debug=False):
        """
        Saves the encoded chunk to a file

        :param encoded_chunk: Encoded chunk
        :type encoded_chunk: np.ndarray
        :param leftovers: Leftovers from the encoders, is they exist
        :type leftovers: np.ndarray
        :param first_chunk: Defines if it is the first chunk, i.e. a file is created. Otherwise, data is appended to the
        file
        :type first_chunk: bool
        :param edit_db: Defines if a new csv with the edited data must be built. Else, a numpy file will be built
        :type edit_db: bool
        """
        file_type = get_names(False)
        file_mode_numpy = 'wb' if first_chunk else 'ab'
        file_mode = 'w' if first_chunk else 'a'
        if not edit_db:
            with open("Output/" + self.output_name + "/" + file_type + ".npy", file_mode_numpy) as output_file:
                for element in encoded_chunk:
                    np.save(output_file, element)
            with open("Output/" + self.output_name + "/leftovers_" + file_type + ".csv", file_mode,
                      newline='') as output_file:
                if first_chunk:
                    csv.writer(output_file).writerows([self.encoder_manager.get_leftover_names()])
                csv.writer(output_file).writerows(leftovers)
        if edit_db:
            create_directories(self.output_name, "Edited")
            with open("Output/" + self.output_name + "/Edited/" + file_type + ".csv", file_mode,
                      newline='') as output_file:
                if first_chunk:
                    csv.writer(output_file).writerows([self.column_names])
                for element in encoded_chunk:
                    csv.writer(output_file).writerows(element)
                    if debug:
                        csv.writer(output_file).writerow(["------"])
        if debug:
            create_directories(self.output_name, "Encoded")
            with open("Output/" + self.output_name + "/Encoded/" + file_type + ".csv", file_mode,
                      newline='') as output_file:
                if first_chunk:
                    csv.writer(output_file).writerows([self.encoder_manager.all_output_column_names])
                for element in encoded_chunk:
                    csv.writer(output_file).writerows(element)
                    if debug:
                        csv.writer(output_file).writerow(["------"])
