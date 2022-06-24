"""
Deep Learning Framework
Version 1.0
Authors: Benoit Vuillemin, Frederic Bertrand
Licence: AGPL v3
"""

from Encoders.base_encoder import *


class NormalizeEncoder(SingleColumnEncoder):

    def __init__(self, column_id) -> None:
        """
        Encoder that normalizes the original values

        :param column_id: Index of the column
        :type column_id: int
        """
        super().__init__("Normalize", ColumnType.QUANTITATIVE, column_id)
        self.output_column_names = [column_id]
        self.min = 0
        self.max = 0
        self.uninitialized = True

    def update_encoder(self, chunk):
        values = chunk[self.column_name].to_numpy()
        min_chunk = min(values)
        max_chunk = max(values)
        if self.uninitialized:
            self.min = min_chunk
            self.max = max_chunk
            self.uninitialized = False
        else:
            self.min = min(self.min, min_chunk)
            self.max = max(self.max, max_chunk)

    def get_properties(self):
        return [self.min, self.max]

    def encode_case(self, case):
        """
        Normalizes the data

        :param case: Case to process
        :type case: np.ndarray
        :return: Encoded column
        :rtype: np.ndarray
        """
        # Replace Not a Number (NaN) with 0
        nans = np.argwhere(np.isnan(case[:, self.column_id].astype(float)))
        case[:, self.column_id] = np.nan_to_num(case[:, self.column_id])
        case[:, self.column_id] = (case[:, self.column_id] - self.min) / (self.max - self.min)
        case[nans, self.column_id] = 0
        return case[:, self.column_id].reshape(-1, 1)

    def set_properties(self, properties):
        self.min = float(properties[0])
        self.max = float(properties[1])


class NormalizeDecoder(SingleColumnEncoder):

    def __init__(self, input_column_names, input_column_ids, output_column_names, output_column_ids, properties):
        """
        Decoder that returns the original, not normalized values

        :param input_column_names: Names of the input columns
        :type input_column_names: list
        :param input_column_ids: Indexes of the input columns
        :type input_column_ids: list
        :param output_column_names: Names of the output columns
        :type output_column_names: list
        :param output_column_ids: Indexes of the output columns
        :type output_column_ids: list
        """
        super().__init__("Normalize", ColumnType.QUANTITATIVE, input_column_ids[0])
        self.output_column_names = output_column_names
        self.min = 0
        self.max = 0
        self.set_properties(properties)

    def set_properties(self, properties):
        self.min = float(properties[0])
        self.max = float(properties[1])

    def encode_case(self, case):
        """
        Returns the original values, not normalized, of the column

        :param case: Case to process
        :type case: np.ndarray
        :return: Array of the not normalized values
        :rtype: np.ndarray
        """
        return ((case[:, self.column_id] * (self.max - self.min)) + self.min).reshape(-1, 1)
