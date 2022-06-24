"""
Deep Learning Framework
Version 1.0
Authors: Benoit Vuillemin, Frederic Bertrand
Licence: AGPL v3
"""

import numpy as np
from Editors import *


class EditorManager:

    def __init__(self, editors) -> None:
        """
        Case editor manager. Edits the case at will, i.e. adds an "End of State" activity at the end, etc.

        :param editors: List of activity and date editors
        :type editors: list
        """
        self.editors = editors
        self.activities_to_add = set()
        for editor in editors:
            self.activities_to_add |= editor.activities_to_add

    def edit_case(self, case, orchestrator):
        """
        Edits the case according to its editors

        :param case: Case to process
        :type case: np.ndarray
        :param orchestrator: Orchestrator
        :type orchestrator: Orchestrator
        :return: Case with an "End of State" activity to the end
        :rtype: np.ndarray
        """
        for editor in self.editors:
            case = editor.edit_case(case, orchestrator)
        return case

    def get_editors_names(self):
        """
        Returns all the names of the editors of the manager

        :return: The list of the editors' names
        :rtype: list
        """
        return [editor.name for editor in self.editors]

    @staticmethod
    def create_from_names(names):
        """
        Creates an EditorManager from a list of editors' names

        :param names: List of editors' names
        :type names: list
        :return: An EditorManager with the corresponding editors
        :rtype: EditorManager
        """
        editors = []
        for name in names:
            editor_class = globals()[name]
            editors.append(editor_class())
        return EditorManager(editors)
