"""
Deep Learning Framework
Version 1.0
Authors: Benoit Vuillemin, Frederic Bertrand
Licence: AGPL v3
"""

from Editors.base_editor import *


class SosForAll(Editor):
    def __init__(self) -> None:
        super().__init__("SosForAll")
        self.activities_to_add = {"SoS"}

    def alter_orchestrator_infos(self, orchestrator):
        orchestrator.max_case_length += 1

    def edit_case(self, case, orchestrator):
        """
        Adds a "Start of State" activity to the beginning of the case

        :param case: Case to process
        :type case: np.ndarray
        :param orchestrator: Orchestrator
        :type orchestrator: Orchestrator
        :return: Case with a "Start of State" activity to the beginning
        :rtype: np.ndarray
        """
        if not orchestrator.double_timestamps:
            if len(case[0]) == 3:
                new_case = np.vstack(([case[0, 0], "SoS", case[0, -1]], case))
            else:
                new_case = np.vstack((np.hstack(([case[0, 0], "SoS", case[0, 2]], case[0, 3:])), case))
        else:
            if len(case[0]) == 4:
                new_case = np.vstack(([case[0, 0], "SoS", case[0, 2], case[0, 2]], case))
            else:
                new_case = np.vstack((np.hstack(([case[0, 0], "SoS", case[0, 2], case[0, 2]], case[-1, 4:])), case))
        return new_case
