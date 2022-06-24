"""
Deep Learning Framework
Version 1.0
Authors: Benoit Vuillemin, Frederic Bertrand
Licence: AGPL v3
"""

import os
from math import ceil

import numpy as np
import pandas as pd

from column_type import ColumnType


def get_complete_cases(chunk, id_column, first_chunk, last_chunk, previous_case, previous_case_id):
    """
    Converts the raw chunk into complete cases

    :param chunk: Chunk of data
    :type chunk: pd.DataFrame
    :param id_column: Name of the column corresponding to the case IDs
    :type id_column: str
    :param first_chunk: Defines if it is the first chunk to be processed
    :type first_chunk: bool
    :param last_chunk: Defines if it is the last chunk to be processed
    :type last_chunk: bool
    :param previous_case: Last case of the previous chunk
    :type previous_case: np.ndarray
    :param previous_case_id: Case ID of the last case of the previous chunk
    :type previous_case_id: str
    :return: The complete cases of the chunk, a list of the case IDs, the last case of the chunk and its ID
    :rtype: (list, np.ndarray, np.ndarray, int)
    """
    complete_cases = []
    case_ids = []
    chunk_case_counter = 0
    cases = chunk.groupby([id_column])
    first_case = True
    for case_id, case in cases:
        # Check if it is the last case of the chunk
        chunk_case_counter += 1
        last_case = chunk_case_counter == len(cases)
        # If it is the last case of the chunk, it may be cut in half. So, we save it for the next chunk (if there
        # is one)
        if last_case and not last_chunk:
            # Only if a case is split into multiple chunks
            if not first_chunk and first_case and previous_case[id_column] == case[id_column]:
                previous_case = pd.concat([previous_case, case])
            else:
                previous_case = case
                previous_case_id = case_id
            continue
        # If it is the first case of the chunk, it can be the last part of the last case of the previous chunk (
        # if there is one). So, if it is, concatenate
        if first_case and not first_chunk:
            if previous_case_id == case_id:
                complete_cases.append(pd.concat([previous_case, case]))
            else:
                # Else, add the previous case and the current one
                complete_cases.append(previous_case)
                case_ids.append(case_id)
                complete_cases.append(case)
            first_case = False
        # Else, just modify the case !
        else:
            complete_cases.append(case)
        # Add the case and its id to the corresponding lists
        case_ids.append(case_id)
    return complete_cases, np.asarray(case_ids), previous_case, previous_case_id


def remove_nan(chunk):
    """
    Removes empty values (showed as "NaN", not a Number). Replaces by 0 for number columns, and '' for other columns

    :param chunk: Chunk to process
    :type chunk: pd.DataFrame
    :return: Case without "NaN"
    :rtype: pd.DataFrame
    """
    num_cols = chunk.select_dtypes(include=[np.number]).columns
    chunk[num_cols] = chunk.select_dtypes(include=[np.number]).fillna(0)
    chunk = chunk.fillna('')
    return chunk


def get_names(is_cov):
    """
    Defines file names according to the nature of the processed file (data or co-variable)

    :param is_cov: Defines if a co-variable file is processed
    :type is_cov: bool
    :return: Name for the files
    :rtype: str
    """
    return "cov" if is_cov else "data"


def get_total_chunk_number(case_counter, input_chunk_size):
    """
    Returns the total number of chunks for a particular file

    :param case_counter: Total number of cases to have been processed
    :type case_counter: int
    :param input_chunk_size: Number of lines by chunk, used if the database is too big
    :type input_chunk_size: int
    :return: The total number of chunks for this file
    :rtype: int
    """
    return ceil(case_counter / input_chunk_size)


def get_cases_info(chunk, cid_column_name, first_chunk, last_chunk, previous_id, previous_size, case_counter,
                   max_case_length):
    """
    Updates the info (case counter, max length of a case) from a chunk of cases

    :param chunk: Chunk of data
    :type chunk: pd.DataFrame
    :param cid_column_name: Name of the column where the case ID is
    :type cid_column_name: str
    :param first_chunk: Defines if it is the first chunk to be processed
    :type first_chunk: bool
    :param last_chunk: Defines if it is the last chunk to be processed
    :type last_chunk: bool
    :param previous_id: ID of the last case of the previous chunk (if it exists)
    :type previous_id: str
    :param previous_size: Size of the last case of the previous chunk (if it exists)
    :type previous_size: int
    :param case_counter: Total number of cases to have been processed
    :type case_counter: int
    :param max_case_length: Maximum length of a case
    :type max_case_length: int
    :return: Updated ID and size of the last case, case counter and max length of a case
    :rtype: (str, int, int, int)
    """
    unique, counts = np.unique(chunk[cid_column_name], return_counts=True)
    if not first_chunk:
        index = np.where(unique == previous_id)[0]
        if index.size == 0:
            case_counter += 1
            max_case_length = max(max_case_length, previous_size)
        else:
            counts[index] += previous_size
    if not last_chunk:
        if previous_id == np.asarray(chunk[cid_column_name])[-1]:
            index = np.where(unique == previous_id)[0]
            previous_size += counts[index]
        else:
            previous_id = np.asarray(chunk[cid_column_name])[-1]
            index = np.where(unique == previous_id)[0]
            previous_size = counts[index]
            unique = np.delete(unique, index)
            counts = np.delete(counts, index)
    case_counter += unique.size
    if len(counts) > 0:
        max_case_length = max(max_case_length, max(counts))
    return previous_id, previous_size, case_counter, max_case_length


def create_directories(output_name, sub_folder=None):
    """
    Creates the necessary directories to store the results

    :param output_name: Name of the output folder
    :type output_name: str
    :param sub_folder: Sub folder to create
    :type sub_folder: str
    """
    # Create results directory
    if not os.path.exists("Output"):
        os.makedirs("Output")
    if not os.path.exists("Output/" + output_name):
        os.makedirs("Output/" + output_name)
    if sub_folder is not None:
        # Create edited directory
        if not os.path.exists("Output/" + output_name + "/"+sub_folder+"/"):
            os.makedirs("Output/" + output_name + "/"+sub_folder+"/")


def sort_cov_for_all_file(cov_path, input_chunk_size, dates=None):
    """
    Sorts all columns of a file into co-variables that are either qualitative, quantitative or date

    :param cov_path: Path of the co-variables file
    :type cov_path: str
    :param input_chunk_size: Number of lines by chunk, used if the database is too big
    :type input_chunk_size: int
    :param dates: List of columns that have timestamps
    :type dates: list
    :return: Ordered list of columns, with a column type for each index
    :rtype: list
    """
    if dates:
        chunks = pd.read_csv(cov_path, chunksize=input_chunk_size, parse_dates=dates)
    else:
        chunks = pd.read_csv(cov_path, chunksize=input_chunk_size)
    first_chunk = True
    cov_list = []
    for chunk in chunks:
        if first_chunk:
            # Get the names of the columns of the CSV (used to reference the correct columns after)
            cov_list = sort_cov_for_chunk(chunk)
            first_chunk = False
        if not first_chunk:
            cov_list = update_cov_with_chunk(chunk, cov_list)
    return cov_list


def sort_cov_for_chunk(chunk):
    """
    Sorts all columns of a file into co-variables that are either qualitative, quantitative or date

    :param chunk: Chunk to process
    :type chunk: pd.DataFrame
    :return: Ordered list of columns, with a column type for each index
    :rtype: list
    """
    cov_list = []
    first_column = True
    for column, dtype in chunk.dtypes.iteritems():
        if first_column:
            cov_list.append(ColumnType.CASEID)
            first_column = False
            continue
        if np.issubdtype(dtype, np.number):
            cov_list.append(ColumnType.QUANTITATIVE)
        elif np.issubdtype(dtype, np.datetime64):
            cov_list.append(ColumnType.DATE)
        elif np.issubdtype(dtype, np.bool):
            cov_list.append(ColumnType.BOOLEAN)
        else:
            cov_list.append(ColumnType.QUALITATIVE)
    return cov_list


def update_cov_with_chunk(chunk, cov_list):
    """
    Checks if all columns considered as quantitative or date in the previous chunks are still the same, i.e. there is no
    string inside this column in the new chunk.

    :param chunk: Dataframe to process
    :type chunk: pd.DataFrame
    :param cov_list: Ordered list of columns, with a column type for each index
    :type cov_list: list
    :return: Updated ordered list of columns, with a column type for each index
    :rtype: list
    """
    new_col_list = sort_cov_for_chunk(chunk)
    result = [cov_list[i] + new_col_list[i] for i in range(len(cov_list))]
    return result


def merge_data_cov(cases, cov):
    """
    Merges cases with their co-variables.

    :param cases: Cases to process
    :type cases: list
    :param cov: Array of co-variables
    :type cov: np.ndarray
    :return: Updated list of cases, with the co-variables
    :rtype: list
    """
    new_data = []
    i = 0
    for case in cases:
        if case[0][0] == cov[i][0]:
            cov_matrix = np.tile(np.array(cov[i]), (case.shape[0], 1))
            i += 1
        else:
            cov_matrix = np.empty([case.shape[0], len(cov[i])], dtype=object)
        new_data.append(np.hstack((case, cov_matrix)))
    return new_data, i
