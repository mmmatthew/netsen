stages = [
    '1_fetch_videos',
    '2_extract_frames',
    '3_select_for_labelling',
    '4_select_training_subsets',
    '5_train',
    '6_test',
    '7_predict',
    '8_classify',
    '9_compare',
    '10_analyze'
]

frame_extraction_combinations_large = [
    [0, 0, 0],
    [0, 0, 0.1],
    [0, 0, 0.2],
    [0, 0, 0.5],
    [0, 0, 1],
    [0, 0, 5],
    [0, 0.1, 0.2],
    [0, 0.2, 0.4],
    [0, 0.5, 1],
    [0, 1, 2],
    [0, 5, 10],
    [0, 0.1, 0.3],
    [0, 0.2, 0.6],
    [0, 0.5, 1.5],
    [0, 0.2, 1.2]
]

frame_extraction_combinations_small = [
    [0, 0, 5],
    [0, 0, 0.2],
    [0, 0.2, 0.4]
]

frame_extraction_new_dim = (640, 360)
select_sample_images = {
    "intra-event": {
        "cameras": ["cam1"],
        "name": 'intra',
        "start": "161007_110000",
        "end": "161007_121000",
        "count": 100,
        "encourage_flooded": 1  # weight given to flooded frames to encourage selection. the weight given to
                                # non-flooded frames is 1
    },
    "inter-event": {
        "cameras": ["cam1"],
        "name": 'inter',
        "start": "161007_121000",
        "end": "161007_134830",
        "count": 30,
        "encourage_flooded": 3
    },
    "extra": {
        "cameras": ["cam2", "cam3", "cam4", "cam5"],
        "name": 'extra',
        "start": "161007_110000",
        "end": "161007_134830",
        "count": 30,
        "encourage_flooded": 3
    }

}
datasets = {
    "intra": {
        "frac_train": 0.7,
        "frac_validate": 0,
        "frac_test": 0.3
    },
    "inter": {
        "frac_train": 0,
        "frac_validate": 0,
        "frac_test": 1
    },
    "extra": {
        "frac_train": 0,
        "frac_validate": 0,
        "frac_test": 1
    }
}
network = {
    "channels": 3,
    "classes": 3,
    "layers": 4,
    "features_root": 16,
    "class_weights": [1, 1, 1]
}
train = {
    "optimizer": "adam",
    "batch_size": 8,
    "training_iters": 100,
    "epochs": 20,
    "display_step": 50,
    "dropout": 0.8
}
y_scaling = 200
