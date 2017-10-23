from glob import glob
import train_classifier
datasets = glob("E:\\watson_eval\\4_select_training_subsets\\*.csv")

for dataset in datasets:
    train_classifier.train(dataset, "E:\\watson_eval\\5_train")
