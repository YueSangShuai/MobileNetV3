import argparse
import os
import numpy as np
from cv2 import imread, resize as cv2_resize

from openvino.tools.pot import Metric, DataLoader, IEEngine, \
    load_model, save_model, compress_model_weights, create_pipeline
from openvino.tools.pot.utils.logger import init_logger
from openvino.tools.pot.api.samples.utils.argument_parser import get_common_argparser
from int8_dataset import ImageNetDataLoader
from torchvision import transforms
from int8_metric import Accuracy

# Initialize the logger to print the quantization process in the console.
init_logger(level='INFO')

# Custom implementation of classification accuracy metric.
class Accuracy(Metric):

    # Required methods
    def __init__(self, top_k=1):
        super().__init__()
        self._top_k = top_k
        self._name = 'accuracy@top{}'.format(self._top_k)
        self._matches = []

    @property
    def avg_value(self):
        """ Returns accuracy metric value for all model outputs. """
        return {self._name: np.ravel(self._matches).mean()}

    def update(self, output, target):
        """ Updates prediction matches.
        :param output: model output
        :param target: annotations
        """
        if len(output) > 1:
            raise Exception('The accuracy metric cannot be calculated '
                            'for a model with multiple outputs')
        if isinstance(target, dict):
            target = list(target.values())
        predictions = np.argsort(output[0], axis=1)[:, -self._top_k:]
        match = [float(t in predictions[i]) for i, t in enumerate(target)]

        self._matches.append(match)

    def reset(self):
        """ Resets collected matches """
        self._matches = []

    def get_attributes(self):
        """
        Returns a dictionary of metric attributes {metric_name: {attribute_name: value}}.
        Required attributes: 'direction': 'higher-better' or 'higher-worse'
                             'type': metric type
        """
        return {self._name: {'direction': 'higher-better',
                             'type': 'accuracy'}}

def get_configs(args):

    model_config = {
        'model_name': 'sample_model',
        'model': args.xml,
        'weights': args.bin
    }
    engine_config = {
        'device': args.data_deveice,
    }
    dataset_config = {
        'data_source': args.data_source,
    }

    compression=[
        {
            'name': 'DefaultQuantization',
            'params': {
                'target_device': args.target_device,
                'preset': 'accuracy',
                'stat_subset_size': args.stat_subset_size
            }
        }
    ]
    

    return model_config, engine_config, dataset_config, compression

def optimize_model(args,transform):
    model_config, engine_config, dataset_config, algorithms = get_configs(args)


    # Step 1: Load the model.
    model = load_model(model_config)

    # Step 2: Initialize the data loader.
    data_loader = ImageNetDataLoader(dataset_config,transform)
    # Step 3 (Optional. Required for AccuracyAwareQuantization): Initialize the metric.
    metric = Accuracy(top_k=1)

    # Step 4: Initialize the engine for metric calculation and statistics collection.
    engine = IEEngine(engine_config, data_loader, None)

    # Step 5: Create a pipeline of compression algorithms.
    pipeline = create_pipeline(algorithms, engine)

    # Step 6: Execute the pipeline.
    compressed_model = pipeline.run(model)

    # Step 7 (Optional): Compress model weights quantized precision
    #                    in order to reduce the size of final .bin file.
    if not args.keep_uncompressed_weights:
        compress_model_weights(compressed_model)

    return compressed_model, pipeline


def main(agrs,transform):
    # Steps 1-7: Model optimization
    compressed_model, pipeline = optimize_model(args,transform)
    # # Step 8: Save the compressed model to the desired path.
    save_model(compressed_model,save_path=agrs.save_path,model_name=agrs.save_name)
    # # Step 9 (Optional): Evaluate the compressed model. Print the results.
    metric_results = pipeline.evaluate(compressed_model)
    if metric_results:
        for name, value in metric_results.items():
            print('{: <27s}: {}'.format(name, value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--xml',
        type=str,
        default='/home/yuesang/Pythonproject/MobileNetV3/runs/train/exp1/bestfp16.xml',
        help='xml path',
    )
    parser.add_argument(
        '--bin',
        type=str,
        default='/home/yuesang/Pythonproject/MobileNetV3/runs/train/exp1/bestfp16.bin',
        help='bin path',
    )
    parser.add_argument(
        '--data_deveice',
        type=str,
        default='CPU',
        help='读取数据集的设备',
    )
    
    parser.add_argument(
        '--data_source',
        type=str,
        default='/mnt/hdd-4t/data/hanyue/mobileNet/flower_for_cnn/train',
        help='数据集位置',
    )
    parser.add_argument(
        '--target_device',
        type=str,
        default='CPU',
        help='量化适配的设备',
    )

    parser.add_argument(
        '--stat_subset_size',
        type=int,
        default=3000,
        help='子集的大小',
    )
    
    parser.add_argument(
        '--save_path',
        type=str,
        default="/home/yuesang/Pythonproject/MobileNetV3/runs/train/exp1",
        help='保存路径',
    )
    parser.add_argument(
        '--save_name',
        type=str,
        default="defaultint8",
        help='保存名称',
    )
    
    parser.add_argument(
        '--keep_uncompressed_weights',
        type=bool,
        default=False,
        help='是否保存权重',
    )
    
    data_transform = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }
    
    
    args = parser.parse_args()
    main(args,data_transform['val'])
