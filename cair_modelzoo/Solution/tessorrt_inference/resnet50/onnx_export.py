import copy
import argparse
import torch
import torchvision.models as models

def export(model, output, shape=(3,224,224), opset=10, input_names=['input'], output_names=['output']):
    # 定义输入名称，list结构，可能有多个输入
    # input_names = ['input']
    # 定义输出名称，list结构，可能有多个输出
    # output_names = ['output']
    dynamic_axes = {
                'input': {0: 'batch_size'}
            }
    # 构造输入用以验证onnx模型的正确性
    c, h, w = shape
    input = torch.rand(1, c, h, w)
    # 导出
    torch.onnx.export(model, input, output,
                            export_params=True,
                            opset_version=opset,
                            do_constant_folding=True,
                            input_names=input_names,
                            output_names=output_names,
                            dynamic_axes=dynamic_axes)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--model_src', type=str, default="./models/pytorch/resnet50/resnet50_gpu_98.pth")
    parser.add_argument('--model_dst', type=str, default="/workspace/model/private/tensorrt_infer/onnx/resnet50/resnet50_gpu_98_op11.onnx")
    parser.add_argument('--shape', type=tuple, default=(3,224,224))
    parser.add_argument('--opset', type=int, default=10)
    args = parser.parse_args()

    # 构造模型实例
    model = models.__dict__["resnet50"]()
    
    # 反序列化权重参数
    resume_point = torch.load(args.model_src, map_location=torch.device('cpu'))
    resume_point_replace = {}
    for key in resume_point['state_dict'].keys():
        split_key = key.split('.')
        split_origin = copy.deepcopy(split_key)
        for item in split_origin:
            if item == "module":
                split_key.remove("module")
            elif item == "submodule":
                split_key.remove("submodule")
        resume_point_replace[".".join(split_key)] = resume_point['state_dict'][key]
    model.load_state_dict(resume_point_replace, strict=True)
    model.eval()

    export(model, args.model_dst, shape=args.shape, opset=args.opset)