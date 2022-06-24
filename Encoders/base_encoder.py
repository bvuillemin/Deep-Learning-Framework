"""
Deep Learning Framework
Version 1.0
Authors: Benoit Vuillemin, Frederic Bertrand
Licence: AGPL v3
"""

import numpy as np

from column_type import ColumnType


class Encoder:

    def __init__(self, name, column_type) -> None:
        """
        Subclass of an encoder

        :param column_type: Type of the column to process: Qualitative, Quantitative or Date
        :type column_type: ColumnType
        """
        # Automatically filled attributes
        self.name = name
        self.column_type = column_type
        self.cid_column_name = None
        # Mandatory attribute. Must be filled before the end of the "finalize" method
        self.output_column_names = None
        # Optional attributes
        self.leftover_name = None

    def set_leftover_name(self, column_id):
        """
        Sets the name of the leftover. Usually, it is the name of the encoder (or decoder) and the corresponding column

        :param column_id: Index of the corresponding column
        :type column_id: int
        """
        self.leftover_name = self.name + "_" + str(column_id)

    def encode_case(self, case):
        """
        Encodes the case according to the encoder internal representation. Mandatory function

        :param case: Case to process
        :type case: np.ndarray
        """
        pass

    def update_encoder(self, chunk):
        """
        Updates the internal representation of the encoder. Optional function, only for encoders

        :param chunk: Chunk to process (here, a chunk of a file)
        :type chunk: pd.DataFrame
        """
        pass

    def finalize(self):
        """
        Makes the final operations to the internal representation of the encoder. Optional function, only for encoders

        """
        pass

    def get_leftover(self, case):
        """
        Returns the leftover of the chunk. Optional function, only for encoders

        :param case: Case to process
        :type case: np.ndarray
        """
        pass

    def set_leftover(self, leftover):
        """
        Sets the leftover of the decoder. Optional function, only for decoders

        :param leftover: Leftover
        :type leftover: np.ndarray
        """
        pass

    def get_properties(self):
        """
        Gets the internal properties of the encoder

        """
        return []

    def set_properties(self, properties):
        """
        Sets the internal properties of the encoder

        """
        pass

    def tamper(self, activities_to_add):
        """
        Allows a case editor to tamper with the internal representation of the encoder

        """
        pass

    def encode_single_result(self, input, output, leftover):
        """
        Encodes a single result made from a neural network

        """
        pass


class SingleColumnEncoder(Encoder):

    def __init__(self, name, column_type, column_id) -> None:
        """
        Encoder that processes a single column inside a case

        :param column_type: Type of the encoder
        :type column_type: EncoderType
        :param column_id: Index of the column to process
        :type column_id: int
        """
        super().__init__(name, column_type)
        self.column_id = column_id
        self.column_name = ""

    def set_column_names(self, columns):
        """
        Sets the column name to process according to its index

        :param columns: Columns of the file
        :type columns: list
        """
        self.cid_column_name = columns[0]
        self.column_name = columns[self.column_id]

    def get_description(self):
        """
        Returns all the necessary information of the encoder

        :return: Name of the encoder, names and ids of the input columns, names of the output columns
        :rtype: (list, list, list, list)
        """
        if self.output_column_names is not None:
            return [self.name], self.get_properties(), [self.column_name], [self.column_id], self.output_column_names
        else:
            return [self.name], self.get_properties(), [self.column_name], [self.column_id], [""]

    def get_description_df(self):
        """
        Returns all the necessary information of the encoder for a dataframe

        :return: Name of the encoder, names and ids of the input columns, names of the output columns
        :rtype: (list, list, list, list)
        """
        if self.output_column_names is not None:
            return self.name, self.get_properties(), self.column_name, self.column_id, self.output_column_names
        else:
            return self.name, self.get_properties(), self.column_name, self.column_id, ""

    @staticmethod
    def get_superclass():
        """
        Returns if the encoder is Single Column or Multi Column

        """
        return "SingleColumn"


class MultiColumnEncoder(Encoder):

    def __init__(self, name, column_type, columns) -> None:
        """
        Encoder that processes multiple columns inside a case

        :param column_type: Type of the column to process: Qualitative, Quantitative or Date
        :type column_type: ColumnType
        :param columns: Columns to process
        :type columns: list
        """
        super().__init__(name, column_type)
        self.column_ids = columns
        self.column_names = []

    def set_column_names(self, columns):
        """
        Sets the columns names to process according to its index

        :param columns: Columns of the file
        :type columns: list
        """
        self.cid_column_name = columns[0]
        self.column_names = [columns[c] for c in self.column_ids]

    def get_description(self):
        """
        Returns all the necessary information of the encoder

        :return: Name of the encoder, names and ids of the input columns, names of the output columns
        :rtype: (list, list, list, list)
        """
        if self.output_column_names is not None:
            return [self.name], self.get_properties(), self.column_names, self.column_ids, self.output_column_names
        else:
            return [self.name], self.get_properties(), self.column_names, self.column_ids, [""]

    def get_description_df(self):
        """
        Returns all the necessary information of the encoder for a dataframe

        :return: Name of the encoder, names and ids of the input columns, names of the output columns
        :rtype: (list, list, list, list)
        """
        if self.output_column_names is not None:
            return self.name, self.get_properties(), self.column_names, self.column_ids, self.output_column_names
        else:
            return self.name, self.get_properties(), self.column_names, self.column_ids, ""

    @staticmethod
    def get_superclass():
        """
        Returns if the encoder is Single Column or Multi Column

        """
        return "MultiColumn"
