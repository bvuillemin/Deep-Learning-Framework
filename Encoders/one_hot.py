"""
Deep Learning Framework
Version 1.0
Authors: Benoit Vuillemin, Frederic Bertrand
Licence: AGPL v3
"""

from Encoders.base_encoder import *


class OneHotEncoder(SingleColumnEncoder):

    def __init__(self, column_id, activity=False) -> None:
        """
        "One-hot" encoder, that converts a qualitative value into a "one-hot" vector

        :param column_id: Index of the column to process
        :type column_id: int
        :param activity: Defines if this encoder is processing activities (hence can be modified by the editor manager)
        :type activity: bool
        """
        super().__init__("OneHot", ColumnType.QUALITATIVE, column_id)
        self.encoding = {}
        self.unique_values = np.empty([0])
        self.values_set = set()
        self.activity = activity
        if self.activity:
            self.activities_to_add = set()

    def tamper(self, activities_to_add):
        self.activities_to_add = activities_to_add

    def update_encoder(self, chunk):
        self.values_set.update(set(chunk[self.column_name].astype(str).unique()))

    def finalize(self):
        self.unique_values = np.sort(list(self.values_set))
        if self.activity:
            self.unique_values = np.hstack((np.sort(list(self.activities_to_add)), self.unique_values))
        self.encoding = dict(zip(self.unique_values, range(len(self.unique_values))))
        self.output_column_names = self.unique_values

    def encode_case(self, case):
        """
        Encodes the case according to the encoder internal representation

        :param case: Case to process
        :type case: np.ndarray
        :return: Column of the case with all values replaced by the corresponding "one-hot" vector
        :rtype: np.ndarray
        """
        column = case[:, self.column_id]
        one_hot = np.zeros((len(column), len(self.unique_values)), dtype=np.int8)
        nan_indexes = set(np.where(column == "")[0].tolist())
        oh_indexes = np.array([self.encoding[str(value)] for value in column if value is not None])
        if len(nan_indexes) == 0:
            oh_rows = np.arange(len(oh_indexes))
        else:
            oh_rows = np.asarray([i for i in range(len(column)) if i not in nan_indexes])
        if len(oh_rows) > 0:
            one_hot[oh_rows, oh_indexes] = 1
        return one_hot

    def get_properties(self):
        return self.unique_values.tolist()

    def set_properties(self, properties):
        self.unique_values = np.asarray(properties)
        self.encoding = dict(zip(self.unique_values, range(len(self.unique_values))))
        self.output_column_names = self.unique_values


class OneHotDecoder(MultiColumnEncoder):

    def __init__(self, input_column_names, input_column_ids, output_column_names, output_column_ids,
                 properties) -> None:
        """
        "One-hot" decoder, that converts a "one-hot" vector into a qualitative value

        :param input_column_names: Names of the input columns
        :type input_column_names: list
        :param input_column_ids: Indexes of the input columns
        :type input_column_ids: list
        :param output_column_names: Names of the output columns
        :type output_column_names: list
        :param output_column_ids: Indexes of the output columns
        :type output_column_ids: list
        """
        super().__init__("OneHot", ColumnType.QUALITATIVE, input_column_ids)
        self.column_names = np.asarray(output_column_names)
        self.output_column_names = self.column_names
        self.unique_values = []
        self.set_properties(properties)

    def set_properties(self, properties):
        self.unique_values = np.asarray(properties)

    def encode_case(self, case):
        """
        Decodes the case according to the encoder internal representation

        :param case: Case to process
        :type case: np.ndarray
        :return: Column of the case with all original values
        :rtype: np.ndarray
        """
        if len(self.unique_values) == 1:
            return np.full(len(case), self.unique_values[0]).reshape(-1, 1)
        """result = np.empty((len(case), 1), dtype=object)
        one_hot_indexes = np.where(case[:, self.column_ids[0]:self.column_ids[1] + 1] == 1)
        result[one_hot_indexes[0], 0] = self.unique_values[one_hot_indexes[1]]"""
        result = self.unique_values[case[:, self.column_ids[0]:self.column_ids[1] + 1].argmax(axis=1)]
        result = result.reshape(-1, 1)
        return result

    def encode_single_result(self, input, output, leftover):
        return self.unique_values[np.argmax(output[self.column_ids[0]:self.column_ids[1] + 1])]
