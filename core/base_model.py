import os
import json
from abc import ABCMeta,abstractmethod
with open('config/config.json', 'r') as f:
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
        self.model_init()

    def get_info(self):
        return configs

    @abstractmethod
    def check_data(self, posted_dict:dict):
        '''
        实现此函数以达到以下功能：
        1. 检查传入字典中是否有你期望的key
        2. 检查该字典中的键值是否符合你的预期
        若不符合, 即抛出assertion error
        返回值：用于预处理的数据
        
        EXAMPLE:
        def check_data(self, posted_dict):
            if 'data' not in posted_dict:
                raise AssertionError('The field "data" does not exist')
            image_paths = posted_dict['data']
            if not isinstance(image_paths, list):
                raise AssertionError('Data passed is not a list')
            return image_paths
        '''
        pass
    
    @abstractmethod
    def model_init(self):
        '''
        实现此函数以达到以下功能：
        1. 加载模型
        2. 传入热身数据
        
        EXAMPLE:
        def model_init(self):
            self.model = tf.keras.models.load_model(self.model_path, custom_objects={'UpdatedMeanIoU': UpdatedMeanIoU})
            self.postprocess(self.predict(self.preprocess([self.warmup_path])))
        '''
        pass

    @abstractmethod
    def preprocess(self, parsed_data):
        '''
        实现此函数以满足你的模型对于预处理的需求,如：
        1. 遍历图片路径
        2. 读取图片
        3. 将图片叠加为batch进入模型
        返回值: 用于模型处理的成batch数据
        
        EXAMPLE:
        def preprocess(self, image_paths):
            preprocessed_data = []
            for image_path in image_paths:
                image = cv2.imread(str(image_path))
                resized_image = cv2.resize(image, (960,540))
                preprocessed_data.append(resized_image)
            preprocessed_data = np.stack(preprocessed_data, axis=0)
            return preprocessed_data
        '''
        pass

    @abstractmethod
    def predict(self, batch_data):
        '''
        实现此函数以达到：
        1. 模型推理
        返回值: 推理结果
        
        EXAMPLE:
        def predict(self, batch_data):
            result = self.model.predict(batch_data, batch_size = self.max_infer_batch_size)
            return result
        '''
        pass
    
    @abstractmethod
    def postprocess(self, result):
        '''
        实现此函数以满足你对后处理的需求，如：
        1. 将结果取argmax
        返回值: 可被序列化的结果, 以用于API直接返回
        
        EXAMPLE:
        def postprocess(self, result):
            result = np.argmax(result, axis=-1)
            return result.tolist()
        '''
        pass