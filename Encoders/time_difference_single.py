"""
Deep Learning Framework
Version 1.0
Authors: Benoit Vuillemin, Frederic Bertrand
Licence: AGPL v3
"""

from Encoders.base_encoder import *


class TimeDifferenceSingleEncoder(SingleColumnEncoder):

    def __init__(self, column_id) -> None:
        """
        Encoder that takes a list of dates, and computes the normalized difference between a date and its previous one,
        associated to the column "Time_diff"

        :param column_id: Index of the dates column
        :type column_id: int
        """
        super().__init__("TimeDifferenceSingle", ColumnType.DATE, column_id)
        self.output_column_names = ["Time_diff"]
        self.set_leftover_name(column_id)
        self.max = 0
        self.previous_date = None
        self.previous_id = None

    def update_encoder(self, chunk):
        """
        Updates the internal representation of the encoder

        :param chunk: Chunk to process (here, a chunk of a file)
        :type chunk: pd.DataFrame
        """
        if self.previous_id:
            dates = np.hstack(([self.previous_date], chunk[self.column_name]))
            ids = np.hstack(([self.previous_id], chunk[self.cid_column_name]))
        else:
            dates = chunk[self.column_name].to_numpy()
            ids = chunk[self.cid_column_name].to_numpy()
        dates_difference = dates[1:] - dates[:-1]
        dates_difference = (dates_difference / 1000000000).astype(int)
        dates_filter = np.array([i == j for i, j in zip(ids[1:], ids[:-1])])
        dates_difference = dates_difference * dates_filter
        self.max = max(self.max, dates_difference.max())
        self.previous_date = dates[-1]
        self.previous_id = ids[-1]

    def get_properties(self):
        """
        Gets the internal properties of the encoder

        """
        return [self.max]

    def encode_case(self, case):
        """
        Encodes the case's date: computes the difference between a date and its previous one

        :param case: Case to process
        :type case: np.ndarray
        :return: Array of the encoded dates
        :rtype: np.ndarray
        """
        dates = np.asarray([x.value for x in case[:, self.column_id]])
        dates = (dates / 1000000000).astype(int)
        dates_difference = np.append(0, dates[1:] - dates[:-1]).reshape(-1, 1) / self.max
        return dates_difference

    def get_leftover(self, case):
        """
        Returns the first date of the column

        :param case: Case to process
        :type case: np.ndarray
        :return: First date of the column
        :rtype: np.datetime64
        """
        return case[0, self.column_id]

    def set_properties(self, properties):
        self.max = int(properties[0])


class TimeDifferenceSingleDecoder(SingleColumnEncoder):

    def __init__(self, input_column_names, input_column_ids, output_column_names, output_column_ids,
                 properties) -> None:
        """
        Encoder that takes a list of dates, and computes the difference between a date and its previous one,
        associated to the column "Time_diff"

        :param input_column_names: Names of the input columns
        :type input_column_names: list
        :param input_column_ids: Indexes of the input columns
        :type input_column_ids: list
        :param output_column_names: Names of the output columns
        :type output_column_names: list
        :param output_column_ids: Indexes of the output columns
        :type output_column_ids: list
        """
        super().__init__("TimeDifferenceSingle", ColumnType.DATE, input_column_ids[0])
        self.date_counter = 0
        self.output_column_names = output_column_names
        self.first_dates = []
        self.set_leftover_name(output_column_ids[0])
        self.max = 0
        self.set_properties(properties)

    def set_properties(self, properties):
        self.max = int(properties[0])

    def get_properties(self):
        """
        Gets the internal properties of the decoder

        """
        return [self.max]

    def set_leftover(self, first_dates):
        """
        Saves the start dates of the chunk inside the decoder

        :param first_dates: Start dates of the chunk
        :type first_dates: np.ndarray
        """
        self.date_counter = 0
        self.first_dates = first_dates

    def encode_case(self, case):
        """
        Decodes the case's date: returns the original dates

        :param case: Case to process
        :type case: np.ndarray
        :return: Array of the decoded dates
        :rtype: np.ndarray
        """
        start_date = np.datetime64(self.first_dates[self.date_counter])
        self.date_counter += 1
        result = [start_date]
        for i in range(1, len(case)):
            difference = np.timedelta64(int(case[i, self.column_id] * self.max), 's')
            new_date = result[i - 1] + difference
            result.append(new_date)
        result = np.reshape(np.asarray(result, dtype=object), (-1, 1))
        return result

    def encode_single_result(self, input, output, leftover):
        new_date = np.datetime64(leftover)
        for i in range(1, len(input)):
            difference = np.timedelta64(int(input[i, self.column_id] * self.max), 's')
            new_date += difference
        difference = np.timedelta64(int(output[self.column_id] * self.max), 's')
        new_date += difference
        return new_date
