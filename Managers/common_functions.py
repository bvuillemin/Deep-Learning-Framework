"""
Deep Learning Framework
Version 1.0
Authors: Benoit Vuillemin, Frederic Bertrand
Licence: AGPL v3
"""

import csv

from tqdm import tqdm

from Encoders import *
from generic_functions import *
from Managers.editor_manager import EditorManager
from Managers.encoder_manager import EncoderManager
from Managers.orchestrator import Orchestrator


def build_orchestrator(input_path, output_name, input_chunk_size, encoder_manager, editor_manager=None, dates_ids=None,
                       double_timestamps=False):
    """
    Builds and initializes an orchestrator from a (either data or cov) file

    :param input_path: Name of the raw data file
    :type input_path: str
    :param output_name: Name of the output file
    :type output_name: str
    :param input_chunk_size: Number of lines by chunk, used if the database is too big
    :type input_chunk_size: int
    :param encoder_manager: Encoder manager
    :type encoder_manager: EncoderManager
    :param editor_manager: Editor manager
    :type editor_manager: EditorManager
    :param dates_ids: Indexes of the dates columns (for pandas to parse those dates as Timestamps, not strings)
    :type dates_ids: list
    :param double_timestamps: Defines if the file had two timestamps (only used for data files)
    :type double_timestamps: bool
    """
    orchestrator = Orchestrator(encoder_manager, editor_manager)
    orchestrator.alter_encoders_descriptions()
    # Read the whole file once, to get all the information needed for the encoders
    orchestrator.init_from_data(input_path, output_name, input_chunk_size, double_timestamps, dates_ids)
    orchestrator.alter_internal_infos()
    return orchestrator


def load_orchestrator_from_file(output_name):
    """
    Loads an existing orchestrator from a save file

    :param output_name: Name of the output folder
    :type output_name: str
    """
    file_type = get_names(False)
    reader = csv.reader(open("Output/" + output_name + "/desc_" + file_type + ".csv", 'r'))
    input_path = next(reader)[1]
    output_name = next(reader)[1]
    _ = next(reader)
    column_names = np.asarray(next(reader))
    _ = next(reader)
    dates_ids = next(reader)
    dates_ids = [int(i) for i in dates_ids]
    case_counter = int(next(reader)[1])
    total_chunk_counter = int(next(reader)[1])
    double_timestamps = next(reader)[1] == 'True'
    activity_counter = int(next(reader)[1])
    max_case_length = int(next(reader)[1])
    features_counter = int(next(reader)[1])
    has_leftovers = next(reader)[1] == 'True'
    _ = next(reader)
    editors_names = next(reader)
    encoder_counter = int(next(reader)[1])
    encoders = []
    for i in range(encoder_counter):
        encoder_type = next(reader)[0]
        properties = next(reader)
        input_column_names = next(reader)
        input_column_ids = list(map(int, next(reader)))
        output_column_names = next(reader)
        output_column_ids = list(map(int, next(reader)))
        encoder_class = globals()[encoder_type + "Encoder"]
        if len(input_column_ids) == 1:
            encoder = encoder_class(input_column_ids[0])
        else:
            encoder = encoder_class(input_column_ids)
        encoder.set_properties(properties)
        encoder.set_column_names(column_names)
        encoders.append(encoder)
    encoder_manager = EncoderManager(encoders)
    encoder_manager.set_all_output_column_names()
    editor_manager = EditorManager.create_from_names(editors_names)
    orchestrator = Orchestrator(encoder_manager, editor_manager)
    orchestrator.insert_infos(input_path, output_name, column_names, dates_ids, case_counter, total_chunk_counter,
                              double_timestamps, activity_counter, max_case_length)
    return orchestrator


def create_decoder(encoder_description, start_input_column_id=None):
    """
    Creates a decoder from an encoder description

    :param encoder_description: Description of the encoder
    :type encoder_description: list
    :param start_input_column_id: First columns id of the encoded data (only used for neural network output)
    :type start_input_column_id: int
    :return: Decoder
    :rtype: Encoder
    """
    decoder_class = globals()[encoder_description[0][0] + "Decoder"]
    properties = encoder_description[1]
    output_column_names = encoder_description[2]
    output_column_ids = encoder_description[3]
    input_column_names = encoder_description[4]
    if start_input_column_id is None:
        input_column_ids = encoder_description[5]
    else:
        input_column_ids = [start_input_column_id, start_input_column_id + len(input_column_names) - 1]
    decoder = decoder_class(input_column_names, input_column_ids, output_column_names, output_column_ids,
                                properties)
    return decoder


def create_all_decoders(orchestrator):
    """
    Creates all the decoders according to the description of all the encoders

    :param orchestrator: Orchestrator that has the description of all the encoders
    :type orchestrator: Orchestrator
    :return: All the associated decoders inside a encoder manager
    :rtype: EncoderManager
    """
    decoders = []
    for encoder_description in orchestrator.encoder_descriptions:
        decoders.append(create_decoder(encoder_description))
    decoder_manager = EncoderManager(decoders)
    return decoder_manager


def auto_cov_encoders(input_path, input_chunk_size, skip=0):
    """
    Automatically generates a list of encoders according to the type of data stored inside each co-variables column

    :param input_path: Name of the input file
    :type input_path: str
    :param input_chunk_size: Number of lines by chunk, used if the database is too big
    :type input_chunk_size: int
    :param skip: Number of columns to skip, if the file has got colulmns of data that are not co-variables (such as case
    ID, activity, timestamp)
    :type skip: int
    :return: List of generated encoders
    :rtype: list
    """
    cov_encoders = []
    cov_list = sort_cov_for_all_file(input_path, input_chunk_size)
    # Add the encoders (here, one type for each column type)
    for index, column in enumerate(cov_list):
        if index < skip:
            continue
        if column == ColumnType.CASEID:
            cov_encoders.append(DeleteEncoder(index))
        elif column == ColumnType.QUANTITATIVE:
            cov_encoders.append(NormalizeEncoder(index))
        elif column == ColumnType.QUALITATIVE:
            cov_encoders.append(OneHotEncoder(index, activity=False))
        elif column == ColumnType.BOOLEAN:
            cov_encoders.append(BooleanEncoder(index))
    return cov_encoders


def decode_offline(output_name, output_chunk_size):
    """
    Decodes the data of a file, whether it is a case or a co-variable file

    :param output_name: Name of the folder where the file is
    :type output_name: str
    :param output_chunk_size: Number of cases by chunk
    :type output_chunk_size: int
    """
    create_directories(output_name, "Decoded")
    file_type = get_names(False)
    leftover_filename = "Output/" + output_name + "/leftovers_" + file_type
    decoded_filename = "Output/" + output_name + "/Decoded/decoded_" + file_type

    orchestrator = load_orchestrator_from_file(output_name)
    decoder_manager = create_all_decoders(orchestrator)
    if orchestrator.has_leftovers:
        with open(leftover_filename + ".csv") as leftover_file:
            reader = csv.reader(leftover_file)
            leftover_columns = next(reader)
    with open("Output/" + output_name + "/" + file_type + ".npy", 'rb') as input_file:
        for chunk_index in tqdm(range(ceil(orchestrator.case_counter / output_chunk_size)), desc="Decode " + file_type):
        # for chunk_index in range(ceil(orchestrator.case_counter / input_chunk_size)):
            cases = []
            # Compute the number of cases to get
            # If it is the last chunk, there will be less cases left
            if chunk_index == ceil(orchestrator.case_counter / output_chunk_size) - 1:
                chunk_range = orchestrator.case_counter % output_chunk_size
            # Else, the number of cases if equal of the size of a chunk
            else:
                chunk_range = output_chunk_size
            # Add all the cases to a list
            for c in range(chunk_range):
                cases.append(np.load(input_file, allow_pickle=True))
            if orchestrator.has_leftovers:
                # Add additional info to every decoder, if needed (such as the start dates)
                leftover_chunk = pd.read_csv(leftover_filename + ".csv", skiprows=(output_chunk_size * chunk_index) + 1,
                                             nrows=chunk_range, header=None)
                leftover_chunk.columns = leftover_columns
                for encoder in decoder_manager.encoders:
                    if encoder.leftover_name:
                        info = leftover_chunk[encoder.leftover_name]
                        encoder.set_leftover(info)
            decoded_cases = []
            for case in cases:
                decoded_cases.append(decoder_manager.encode_case(case))
            # Write the results!
            # If we are in the first chunk, create a new file
            if chunk_index == 0:
                with open(decoded_filename + ".npy", 'wb') as output_file:
                    for case in decoded_cases:
                        np.save(output_file, case)
                with open(decoded_filename + ".csv", 'w', newline='') as output_file:
                    csv.writer(output_file).writerows([orchestrator.column_names])
                    for case in decoded_cases:
                        csv.writer(output_file).writerows(case.tolist())
            # Else, append to this file
            else:
                with open(decoded_filename + ".npy", 'ab') as output_file:
                    for case in decoded_cases:
                        np.save(output_file, case)
                with open(decoded_filename + ".csv", 'a', newline='') as output_file:
                    for case in decoded_cases:
                        csv.writer(output_file).writerows(case.tolist())
