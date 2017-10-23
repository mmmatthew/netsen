frame_extraction_combinations = [
    [0, 0, 0],
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

frame_extraction_new_dim = [640, 360],
select_sample_images ={
    "intra-event": {
      "start": "161007_110000",
      "end": "161007_121000",
      "count": 100
    },
    "inter-event":{
      "start": "161007_121000",
      "end": "161007_134830",
      "count": 30
    }

}
network = {
    "channels": 3,
    "classes": 3,
    "layers": 4,
    "features_root": 16,
    "class_weights": [1,1,1]
  }
train = {
    "optimizer": "adam",
    "batch_size": 8,
    "training_iters": 100,
    "epochs": 20,
    "display_step": 50,
    "dropout": 0.8
  }