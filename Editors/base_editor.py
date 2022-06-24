"""
Deep Learning Framework
Version 1.0
Authors: Benoit Vuillemin, Frederic Bertrand
Licence: AGPL v3
"""

import numpy as np


class Editor:
    def __init__(self, name) -> None:
        """
        Creates an Editor object

        :param name: Name of the editor
        :type name: str
        """
        super().__init__()
        self.name = name
        self.activities_to_add = set()

    def alter_orchestrator_infos(self, orchestrator):
        """
        Alters the internal information of the orchestrator (such as maximum length of a case, etc)

        :param orchestrator: Orchestrator to modify
        :type orchestrator: Orchestrator
        """
        pass

    def edit_case(self, case, orchestrator):
        """
        Edits the case

        :param case: Case to process
        :type case: np.ndarray
        :param orchestrator: Orchestrator
        :type orchestrator: Orchestrator
        :return: Edited case
        :rtype: np.ndarray
        """
        pass
