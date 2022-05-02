from keras.utils import Sequence
import numpy as np
import random
random.seed(20)
class DataGenerator(Sequence):

    def __init__(self, setting):
        self.setting = setting

    def __getitem__(self, index):

        batch_x, batch_y = self.setting.dataProvider.get_batch_data(self.setting, index)
        return batch_x, batch_y

    def __len__(self):
        return int(np.floor(self.setting.num_samples / self.setting.batch_size))

    # def on_epoch_end(self):
    #     """Method called at the end of every epoch.