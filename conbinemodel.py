import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List
import onnx

def ConbineModel(model_list):
    
    list_prefix = []
    for i in range(len(model_list)):
        list_prefix.append("output"+str(i)+"_")
    
    input_name = 'input'
    output_name = 'output'
    
    
    modelX_input=modelX = onnx.ModelProto(ir_version=model_list[0].ir_version,
                         producer_name=model_list[0].producer_name,
                         producer_version=model_list[0].producer_version,
                         opset_import=model_list[0].opset_import)
    #1、添加 input
    model1_input_tensor_type = model_list[0].graph.input[0].type.tensor_type
    input_elem_type = model1_input_tensor_type.elem_type
    input_shape = []
    for s in model1_input_tensor_type.shape.dim:
        if (s.dim_value > 0):
            input_shape.append(s.dim_value)
        else:
            input_shape.append(s.dim_param)   
    
    modelX_input = onnx.helper.make_tensor_value_info(
                                    input_name,
                                    input_elem_type,
                                    input_shape
                                )
    modelX.graph.input.append(modelX_input)
    
    #2、添加 output
    model1_output_tensor_type = model_list[0].graph.output[0].type.tensor_type
    output_elem_type = model1_output_tensor_type.elem_type
    output_shape = []
    for s in model1_output_tensor_type.shape.dim:
        if (s.dim_value > 0):
            output_shape.append(s.dim_value)
        else:        
            output_shape.append(s.dim_param)
    
    modelX_output = onnx.helper.make_tensor_value_info(
                                    output_name,
                                    output_elem_type,
                                    output_shape
                                )


    #3、添加输出前的 add 节点
    add_node = onnx.helper.make_node(
        'Add',
        name='add_fea',
        inputs=[list_prefix[i]+model.graph.output[0].name for i,model in enumerate(model_list)],
        outputs=[output_name],
    )


    
    
    for idx in range(len(model_list)):
        model=model_list[idx]
        for node in model.graph.node:
            for i in range(len(node.input)):
                if (node.input[i] != input_name):
                    node.input[i] = list_prefix[idx] + node.input[i]
    
            for i in range(len(node.output)):
                node.output[i] = list_prefix[idx] + node.output[i]
                
            node.name = list_prefix[idx] + node.name
            print(node.name)
            modelX.graph.node.append(node) 
            
        for weight in model.graph.initializer:
            weight.name = list_prefix[idx] + weight.name
            modelX.graph.initializer.append(weight)


    
    modelX.graph.node.append(add_node)
    modelX.graph.output.append(modelX_output)

    
    onnx.save(modelX, '/home/yuesang/Pythonproject/MobileNetV3/runs/train/exp3/modelX.onnx')

    
    

if __name__=="__main__":
    net1=onnx.load("/home/yuesang/Pythonproject/MobileNetV3/runs/train/exp2/best.onnx")
    net2=onnx.load("/home/yuesang/Pythonproject/MobileNetV3/runs/train/exp3/best.onnx")
    net_list=[net1,net2,net2]
    Conbinenet=ConbineModel(net_list)
    