#MMDetection(PyTorch)
## 支持情况

Models  | Framework  | Supported GPU   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
MMDetection  | PyTorch  | V100/A100  | FP32  | Yes  | Not Tested

## Quick Start Guide
GPU Golden Bench通过运行launch.sh启动，通过设置环境变量和提供参数来运行。
### docker容器卷说明
- **CONT**：运行网络时的镜像名，通过模型文件夹的Dockerfile生成。
- **DATASETS**：数据集的名称。
- **TEST_DIR**：模型测试文件夹路径。
- **LOG_DIR**：模型训练输出结果路径。
- **CONFIG_DIR**：模型权重路径。
- **DOCKERFILE**:模型的dockerfile。  

### Run
通过以下命令运行：
`bash launch.sh`
### result
得到faster_rcnn的测试结果
### exec
docker exec -it mmdetection_docker /bin/bash 进入容器可使用训练模型和测试模型
