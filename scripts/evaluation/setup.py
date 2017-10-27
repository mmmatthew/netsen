"""
This sets up the folder structure for evaluating WATSEN for a specific case study
"""

import evaluation_settings as settings


def run(directory):
    """

    :param directory: Folder in which project will be created. The project requires
    :return: full paths to each subdir
    """
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)

    for subdir in settings.stages:
        if not os.path.exists(os.path.join(directory, subdir)):
            os.makedirs(os.path.join(directory, subdir))
