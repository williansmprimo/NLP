import json
import numpy as np
import triton_python_backend_utils as pb_utils
from joblib import load
import tensorflow as tf


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args['model_config'])

        predicao_config = pb_utils.get_output_config_by_name(
            model_config, "PREDICAO")
        
        self.predicao_dtype = pb_utils.triton_string_to_numpy(
            predicao_config['data_type'])

        version_path =  args['model_repository'] + '/' + args['model_version']

        self.vectorize = load(version_path + '/model_vec.pkl')
        self.model = tf.saved_model.load(version_path + '/model_net')
        self.model.predict = self.model.signatures['serving_default']

    def execute(self, requests):
        responses = []

        for request in requests:
            in_x = pb_utils.get_input_tensor_by_name(request, "ENTRADA")

            input_x = in_x.as_numpy()
            input_x = self.vectorize.transform(input_x).toarray()
            output_x = self.model(input_x.tolist())
            predicao_tensor = pb_utils.Tensor("PREDICAO", output_x.numpy().astype(self.predicao_dtype))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[predicao_tensor])
            responses.append(inference_response)

        return responses

    def finalize(self):
        print('Cleaning up...')
