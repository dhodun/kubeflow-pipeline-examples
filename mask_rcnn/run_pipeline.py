#Get or create an experiment and submit a pipeline run
import kfp

EXPERIMENT_NAME='mask_rcnn_tests'
PIPELINE_RUN_NAME='mask_rcnn_run'
PIPELINE_FILENAME='mask_rcnn.py.tgz'

client = kfp.Client('127.0.0.1:8085/pipeline')
experiment = client.create_experiment(EXPERIMENT_NAME)
arguments = {'project': 'dhodun1',
             'bucket': 'gs://maskrcnn-kfp'}
#Submit a pipeline run
run_name = PIPELINE_RUN_NAME + ' run'
run_result = client.run_pipeline(experiment.id, run_name, PIPELINE_FILENAME, arguments)
print('Pipeline run {0} submitted'.format(run_name))