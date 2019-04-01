import kfp.components as comp

EXPERIMENT_NAME='lightweight python components'

#Define a Python function
def add(a: float, b: float) -> float:
   '''Calculates sum of two arguments'''
   return a + b

add_op = comp.func_to_container_op(add)

# Advanced function
# Demonstrates imports, helper functions and multiple outputs
from typing import NamedTuple


def my_divmod(dividend: float, divisor: float, output_dir: str = './') -> NamedTuple('MyDivmodOutput',
                                                                                     [('quotient', float),
                                                                                      ('remainder', float)]):
    '''Divides two numbers and calculate  the quotient and remainder'''
    # Imports inside a component function:
    import numpy as np

    # This function demonstrates how to use nested functions inside a component function:
    def divmod_helper(dividend, divisor):
        return np.divmod(dividend, divisor)

    (quotient, remainder) = divmod_helper(dividend, divisor)

    from tensorflow.python.lib.io import file_io
    import json

    # Exports a sample tensorboard:
    metadata = {
        'outputs': [{
            'type': 'tensorboard',
            'source': 'gs://ml-pipeline-dataset/tensorboard-train',
        }]
    }
    with open(output_dir + 'mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)

    # Exports two sample metrics:
    metrics = {
        'metrics': [{
            'name': 'quotient',
            'numberValue': float(quotient),
        }, {
            'name': 'remainder',
            'numberValue': float(remainder),
        }]}

    with file_io.FileIO(output_dir + 'mlpipeline-metrics.json', 'w') as f:
        json.dump(metrics, f)

    from collections import namedtuple
    divmod_output = namedtuple('MyDivmodOutput', ['quotient', 'remainder'])
    return divmod_output(quotient, remainder)

my_divmod(100, 7)

divmod_op = comp.func_to_container_op(my_divmod, base_image='tensorflow/tensorflow:1.11.0-py3')

import kfp.dsl as dsl


@dsl.pipeline(
    name='Calculation pipeline',
    description='A toy pipeline that performs arithmetic calculations.'
)
def calc_pipeline(
        a='a',
        b='7',
        c='17',
):
    # Passing pipeline parameter and a constant value as operation arguments
    add_task = add_op(a, 4)  # Returns a dsl.ContainerOp class instance.

    # Passing a task output reference as operation arguments
    # For an operation with a single return value, the output reference can be accessed using `task.output` or `task.outputs['output_name']` syntax
    divmod_task = divmod_op(add_task.output, b, '/')

    # For an operation with a multiple return values, the output references can be accessed using `task.outputs['output_name']` syntax
    result_task = add_op(divmod_task.outputs['quotient'], c)


pipeline_func = calc_pipeline
pipeline_filename = pipeline_func.__name__ + '.pipeline.zip'
import kfp.compiler as compiler
compiler.Compiler().compile(pipeline_func, pipeline_filename)


#Specify pipeline argument values
arguments = {'a': '7', 'b': '8'}

#Get or create an experiment and submit a pipeline run
import kfp
client = kfp.Client('127.0.0.1:8085/pipeline')
experiment = client.create_experiment(EXPERIMENT_NAME)

#Submit a pipeline run
run_name = pipeline_func.__name__ + ' run'
run_result = client.run_pipeline(experiment.id, run_name, pipeline_filename, arguments)

#vvvvvvvvv This link leads to the run information page. (Note: There is a bug in JupyterLab that modifies the URL and makes the link stop working)