"""
Deep Learning Framework
Version 1.0
Authors: Benoit Vuillemin, Frederic Bertrand
Licence: AGPL v3
"""


class DataPreparator:

    def __init__(self) -> None:
        """
        Initializes the DataPreparator object

        """
        self.input_chunk_size = None
        self.output_chunk_size = None
        self.batch_size = None
        self.orchestrator = None

    def build(self, input_chunk_size, output_chunk_size, batch_size, orchestrator):
        """
        Assigns the necessary information to the preparator

        :param input_chunk_size: Size of a chunk (number of cases inside a chunk)
        :type input_chunk_size: int
        :param output_chunk_size: Number of cases per chunk
        :type output_chunk_size: int
        :param batch_size: Size of a batch to train the neural network
        :type batch_size: int
        :param orchestrator: Orchestrator
        :type orchestrator: Orchestrator
        """
        self.input_chunk_size = input_chunk_size
        self.output_chunk_size = output_chunk_size
        self.batch_size = batch_size
        self.orchestrator = orchestrator

    def run_online(self, get_leftovers=False):
        """
        Separates the encoded data into inputs and outputs (online mode)

        """
        pass

    def get_epoch_size_online(self):
        """
        Computes the size of an epoch (online mode)

        """
        pass

    def run_offline(self):
        """
        Separates the encoded data into inputs and outputs (offline mode)

        """
        pass

    def get_epoch_size_offline(self):
        """
        Computes the size of an epoch (offline mode)

        """
        pass
