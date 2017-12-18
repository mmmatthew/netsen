
from tf_unet import unet
from tf_unet import image_util
import evaluation_settings as s
import os


def train(dataset_csv, working_dir, appendum='', force=False):

    # create folder name
    foldername = os.path.join(
        working_dir, s.stages[4], '__'.join([
            os.path.splitext(os.path.basename(dataset_csv))[0],
            'ly' + str(s.network['layers']) + 'ftr' + str(s.network['features_root']) + appendum, ''
            ]))

    # only retrain if necessary
    if not os.path.exists(foldername) or force:
        generator = image_util.ImageDataProvider(dataset_path=dataset_csv)
        net = unet.Unet(
            channels=s.network['channels'],
            n_class=s.network['classes'],
            layers=s.network['layers'],
            features_root=s.network['features_root'],
            cost_kwargs=dict(class_weights=s.network['class_weights'])
        )
        trainer = unet.Trainer(net, optimizer=s.train['optimizer'], batch_size=s.train['batch_size'])

        trainer.train(generator, foldername, training_iters=s.train['training_iters'], epochs=s.train['epochs'],
                      display_step=s.train['display_step'], dropout=s.train['dropout'],
                      restore=False, write_graph=True)
    else:
        print(os.path.basename(foldername), ' already exists. Skipping.')
