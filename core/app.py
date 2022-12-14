from flask import Flask, request, Response
from flask_restful import Resource, Api
from custom.custom_model import CustomModel, CustomResourceInit
import json
import logging
import traceback

logging.basicConfig(filename='record.log', level=logging.DEBUG, format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s', filemode='w')

with open('config/config.json', 'r') as f:
    configs = json.load(f)


resource_init = CustomResourceInit()
custom_model = CustomModel()

print('Your model is loaded')
app = Flask(__name__)
api = Api(app)


class ModelTask(Resource):

    def get(self):
        model_info = custom_model.get_info()
        return {'Your model status: ': model_info}
    
    def post(self):
        try:
            posted_dict = request.get_json(force=True)
        except:
            app.logger.error(f'Error when parsing posted json')
            app.logger.error(traceback.format_exc())
            return {'error':'Error when parsing posted json'},400


        try:
            parsed_data = custom_model.check_data(posted_dict)
        except AssertionError:
            app.logger.error(f'Error when checking data')
            app.logger.error(traceback.format_exc())
            return {'error': 'Error when checking files'},400
        app.logger.info(f"Received data:\n{parsed_data}\n")
        
        try:
            preprocess_data = custom_model.preprocess(parsed_data)    
            prediction_result = custom_model.predict(preprocess_data)
            postprocess_result = custom_model.postprocess(prediction_result)
        except:
            app.logger.error(f'Internal error when processing data')
            app.logger.error(traceback.format_exc())
            return {'error': 'Internal error'}, 500
        return {'result': postprocess_result}

api.add_resource(ModelTask, '/')


if __name__ == '__main__':
    from waitress import serve
    serve(app, host=configs['ip_address'], port=configs['port'])
    