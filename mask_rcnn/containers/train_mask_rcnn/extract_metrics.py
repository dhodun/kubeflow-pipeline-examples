import sys
import tensorflow as tf

#path = 'gs://maskrcnn-kfp/mask-rcnn-model/job_190404_181429/eval/events.out.tfevents.1554401803.mask-rcnn-7dc79-147958465'
path = sys.argv[1]

file = open("eval_metrics.csv","w")

metrics = {}

print('Exporting all metrics to {0} from TF Event: {1}'.format(file.name, path))

for e in tf.train.summary_iterator(path):
    for v in e.summary.value:
        metrics[v.tag] = v.simple_value
        file.write('{0},{1}\n'.format(v.tag, v.simple_value))
        print('{0}: {1}'.format(v.tag, v.simple_value))



file.close()

# output mAP_box
file = open("/map_box.txt", "w")
file.write(str(metrics['APm']))
file.close()

# output mAP_segm
file = open("/map_segm.txt", "w")
file.write(str(metrics['mask_APm']))
file.close()


