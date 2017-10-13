"""
This sets up the folder structure for evaluating WATSEN for a specific case study
"""

stages = [
    '1_fetch_videos',
    '2_extract_frames',
    '3_select_for_labelling',
    '4_select_training_subsets',
    '5_train',
    '6_predict',
    '7_classify',
    '8_compare',
    '9_analyze'
]

def run(directory):
    """

    :param directory: Folder in which project will be created. The project requires
    :return: full paths to each subdir
    """
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)

    for subdir in stages:
        if not os.path.exists(os.path.join(directory, subdir)):
            os.makedirs(os.path.join(directory, subdir))

    return [os.path.join(directory, s) for s in stages]