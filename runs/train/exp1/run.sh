/home/yuesang/miniconda3/envs/mobileNetV3/bin/python /home/yuesang/.local/bin/mo --input_model /home/yuesang/Pythonproject/MobileNetV3/runs/train/exp3/best.onnx --output_dir /home/yuesang/Pythonproject/MobileNetV3/runs/train/exp3/ --model_name bestfp16 --compress_to_fp16=True
python ./openvino/tools/mo/openvino/tools/mo/mo.py --input_model /home/yuesang/Pythonproject/MobileNetV3/runs/train/exp1/bestfp16.xml

/home/yuesang/miniconda3/envs/mobileNetV3/bin/python /home/yuesang/.local/bin/benchmark_app -i /home/yuesang/Pythonproject/MobileNetV3/runs/train/exp1/lr.png -m /mnt/hdd-4t/data/hanyue/transformer/model/best.onnx -api sync -d CPU -b 1


convert_annotation imagenet --annotation_file Pythonproject/MobileNetV3/onnxinference/int8inference/my_annotation.txt --labels_file /home/yuesang/Pythonproject/MobileNetV3/onnxinference/int8inference/my_labels.txt --has_background False -o /mnt/hdd-4t/data/hanyue/new_annotations -a /home/yuesang/Pythonproject/MobileNetV3/onnxinference/int8inference/imagenet.pickle -m /home/yuesang/Pythonproject/MobileNetV3/onnxinference/int8inference/imagenet.json

python benchmark_app.py -m /home/yuesang/Pythonproject/MobileNetV3/runs/train/exp1/MobileNetV3.xml -d CPU -api async -i /mnt/hdd-4t/data/hanyue/mobileNet/flower_for_cnn/val/daisy/21652746_cc379e0eea_m.jpg -b 1

pot -c /home/yuesang/Pythonproject/MobileNetV3/onnxinference/int8inference/AccuracyAwareQuantization/model.json --output-dir /home/yuesang/Pythonproject/MobileNetV3/runs/train/exp1
/home/yuesang/miniconda3/envs/mobileNetV3/bin/python /home/yuesang/.local/bin/pot -c /home/yuesang/Pythonproject/MobileNetV3/onnxinference/int8inference/config/defualtQuantization/model.json --output-dir /home/yuesang/Pythonproject/MobileNetV3/runs/train

/home/yuesang/miniconda3/envs/mobileNetV3/bin/python /home/yuesang/.local/bin/accuracy_check -c /home/yuesang/Pythonproject/MobileNetV3/onnxinference/int8inference/config/accuary/mobileNetV3Fp32.yaml -s /mnt/hdd-4t/data/hanyue/mobileNet/flower/ -td CPU
/home/yuesang/miniconda3/envs/mobileNetV3/bin/python /home/yuesang/miniconda3/envs/mobileNetV3/bin/accuracy_check -c /home/yuesang/Pythonproject/transformer/int8inference/config/accuary/mobileNetV3Fp32.yaml -s /mnt/hdd-4t/data/hanyue/mobileNet/facevp112 -td CPU