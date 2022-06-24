"""
Deep Learning Framework
Version 1.0
Authors: Benoit Vuillemin, Frederic Bertrand
Licence: AGPL v3
"""

from Editors.base_editor import *


class EosForAll(Editor):
    def __init__(self) -> None:
        super().__init__("EosForAll")
        self.activities_to_add = {"EoS"}

    def alter_orchestrator_infos(self, orchestrator):
        orchestrator.max_case_length += 1

    def edit_case(self, case, orchestrator):
        """
        Adds an "End of State" activity to the end of the case

        :param case: Case to process
        :type case: np.ndarray
        :param orchestrator: Orchestrator
        :type orchestrator: Orchestrator
        :return: Case with an "End of State" activity to the end
        :rtype: np.ndarray
        """
        if not orchestrator.double_timestamps:
            if len(case[0]) == 3:
                new_case = np.vstack((case, [case[0, 0], "EoS", case[-1, -1]]))
            else:
                new_case = np.vstack((case, np.hstack(([case[0, 0], "EoS", case[-1, 2]], case[-1, 3:]))))
        else:
            if len(case[0]) == 4:
                new_case = np.vstack((case, [case[0, 0], "EoS", case[-1, -1], case[-1, -1]]))
            else:
                new_case = np.vstack((case, np.hstack(([case[0, 0], "EoS", case[-1, 3], case[-1, 3]], case[-1, 4:]))))
        return new_case
