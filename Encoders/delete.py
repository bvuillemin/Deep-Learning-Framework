"""
Deep Learning Framework
Version 1.0
Authors: Benoit Vuillemin, Frederic Bertrand
Licence: AGPL v3
"""

from Encoders.base_encoder import *


class DeleteEncoder(SingleColumnEncoder):

    def __init__(self, column_id) -> None:
        """
        Creates an encoder that deletes the column, and keep the first value as a leftover

        :param column_id: Index of the column
        :type column_id: int
        """
        super().__init__("Delete", ColumnType.ANY, column_id)
        self.set_leftover_name(column_id)

    def get_leftover(self, case):
        """
        Returns the first element of the deleted column

        :param case: Case to process
        :type case: np.ndarray
        :return: First element of the deleted column
        :rtype: Any
        """
        return case[0, self.column_id]


class DeleteDecoder(SingleColumnEncoder):
    def __init__(self, input_column_names, input_column_ids, output_column_names, output_column_ids, properties):
        """
        Decoder that returns the original and deleted column

        :param input_column_names: Names of the input columns
        :type input_column_names: list
        :param input_column_ids: Indexes of the input columns
        :type input_column_ids: list
        :param output_column_names: Names of the output columns
        :type output_column_names: list
        :param output_column_ids: Indexes of the output columns
        :type output_column_ids: list
        """
        super().__init__("Delete", ColumnType.ANY, 0)
        self.output_column_names = output_column_names
        self.set_leftover_name(output_column_ids[0])
        self.data = None
        self.data_counter = 0

    def set_leftover(self, leftover):
        """
        Sets the original values of the column that was deleted

        :param leftover: Original values of the column
        :type leftover: np.ndarray
        """
        self.data_counter = 0
        self.data = leftover

    def encode_case(self, case):
        """
        Deletes the column, and keeps the first value as a leftover

        :param case: Case to process
        :type case: np.ndarray
        :return: Original column
        :rtype: np.ndarray
        """
        result = np.full((len(case), 1), self.data[self.data_counter], dtype=object)
        self.data_counter += 1
        return result

    def encode_single_result(self, input, output, leftover):
        return leftover
