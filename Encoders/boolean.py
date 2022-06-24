"""
Deep Learning Framework
Version 1.0
Authors: Benoit Vuillemin, Frederic Bertrand
Licence: AGPL v3
"""

from Encoders.base_encoder import *


class BooleanEncoder(SingleColumnEncoder):

    def __init__(self, column_id) -> None:
        """
        Encoder that transcribe a boolean into a float (False:0.0, True:1.0)

        :param column_id: Index of the column
        :type column_id: int
        """
        super().__init__("Boolean", ColumnType.BOOLEAN, column_id)
        self.output_column_names = [column_id]

    def encode_case(self, case):
        """
        Returns the encoded column of the case

        :param case: Case to process
        :type case: np.ndarray
        :return: Encoded column
        :rtype: np.ndarray
        """
        return (1.0 * case[:, self.column_id]).reshape(-1, 1)


class BooleanDecoder(SingleColumnEncoder):

    def __init__(self, input_column_names, input_column_ids, output_column_names, output_column_ids, properties):
        """
        Decoder that transcribe a float into a boolean (False:0.0, True:1.0)

        :param input_column_names: Names of the input columns
        :type input_column_names: list
        :param input_column_ids: Indexes of the input columns
        :type input_column_ids: list
        :param output_column_names: Names of the output columns
        :type output_column_names: list
        :param output_column_ids: Indexes of the output columns
        :type output_column_ids: list
        """
        super().__init__("Boolean", ColumnType.BOOLEAN, input_column_ids[0])
        self.output_column_names = output_column_names

    def encode_case(self, case):
        """
        Returns the original column of the case

        :param case: Case to process
        :type case: np.ndarray
        :return: Original column
        :rtype: np.ndarray
        """
        return np.asarray(["True" if value == 1 else "False" for value in case[:, self.column_id]]).reshape(-1, 1)
