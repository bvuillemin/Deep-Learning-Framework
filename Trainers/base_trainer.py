"""
Deep Learning Framework
Version 1.0
Authors: Benoit Vuillemin, Frederic Bertrand
Licence: AGPL v3
"""


class Trainer:
    def __init__(self) -> None:
        """
        Creates a Trainer object

        """
        super().__init__()
        self.preparator = None
        self.epoch_counter = None
        self.decoders = []
        self.model = None

    def build(self, preparator, epoch_counter):
        """
        Sets the internal properties of the Trainer

        :param preparator: Data preparator
        :type preparator: DataPreparator
        :param epoch_counter: Number of inputs per epoch
        :type epoch_counter: int
        """
        self.preparator = preparator
        self.epoch_counter = epoch_counter

    def train_model_online(self):
        """
        Trains the neural network online

        """
        pass

    def train_model_offline(self):
        """
        Trains the neural network offline

        """
        pass

    def get_prediction(self, input, leftovers):
        """
        Gets the prediction of the neural network for an input

        :param input: Input of the neural network
        :type input: np.ndarray
        :param leftovers: Leftovers of the input
        :type leftovers: pd.DataFrame
        :return: The predicted decoded output of the neural network
        :rtype: list
        """
        pass
