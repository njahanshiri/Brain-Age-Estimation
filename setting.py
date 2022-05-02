from Utils import Util

class SettingRepository:
    def __init__(self, data_provider, batch_size, num_sample, type_, data_type, ax_):
        self.helper = Util()
        self.dataProvider = data_provider
        self.batch_size = batch_size
        self.num_samples = num_sample
        self.type_ = type_
        self.ax = ax_
        self.data_type = data_type


