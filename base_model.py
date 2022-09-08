import os
import json
from abc import ABCMeta,abstractmethod
with open('config.json', 'r') as f:
    configs = json.load(f)

class BaseResourceInit(metaclass=ABCMeta):
    def __init__(self):
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
        gpu_devices = ','.join(configs['gpu_id'])
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices
        if configs['model_type'] == 'tensorflow':
            self.tensorflow_init()
        elif configs['model_type'] == 'pytorch':
            self.pytorch_init()

    def tensorflow_init(self):
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=configs['gpu_memory_size'])])
    def pytorch_init(self):
        pass


class BaseModel(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.model_path = configs['model_path']
        self.warmup_path = configs['warmup_path']
        self.max_infer_batch_size = configs['max_infer_batch_size']

    def get_info(self):
        return configs

    def check_data(self, raw_data):
        if not isinstance(raw_data, list):
            raise AssertionError('Data passed is not a list')
        
        if hasattr(self, 'extra_info'):
            if not isinstance(self.extra_info, list):
                raise AssertionError('Extra info passed is not a list')
            if len(raw_data) != len(self.extra_info):
                raise AssertionError('Length of data and extra info are not the same')
    @abstractmethod
    def model_init(self):
        pass

    @abstractmethod
    def preprocess(self, image_paths):
        pass

    @abstractmethod
    def predict(self, batch_data):
        pass
    
    @abstractmethod
    def postprocess(self, result):
        pass