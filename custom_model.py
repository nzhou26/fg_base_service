from base_model import BaseModel, BaseResourceInit
import cv2
import numpy as np
import tensorflow as tf
from helper import *
import os
class CustomResourceInit(BaseResourceInit):
    def __init__(self):
       super().__init__()
       self.other_init()
    def other_init(self):
        pass

class CustomModel(BaseModel):
    
    def __init__(self) -> None:
        super().__init__()
        self.model_init()
    
    def check_data(self, raw_data):
        super().check_data(raw_data)
        for path in raw_data:
            if not os.path.exists(path):
                raise AssertionError(f'Cannot find {path}')
    
    def model_init(self):
        self.model = tf.keras.models.load_model(self.model_path, custom_objects={'UpdatedMeanIoU': UpdatedMeanIoU})
        self.postprocess(self.predict(self.preprocess([self.warmup_path])))

    def preprocess(self, image_paths):
        preprocessed_data = []
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            resized_image = cv2.resize(image, (960,540))
            preprocessed_data.append(resized_image)
        preprocessed_data = np.stack(preprocessed_data, axis=0)
        return preprocessed_data

    def predict(self, batch_data):
        result = self.model.predict(batch_data, batch_size = self.max_infer_batch_size)
        return result

    def postprocess(self, result):
        result = np.argmax(result, axis=-1)
        return result.tolist()

