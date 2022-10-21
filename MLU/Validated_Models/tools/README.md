# Validated models 快速测试

## 一、运行测试

运行validate_all_models.sh批量测试模型性能。

#### 脚本参数：

`-m`: 必选，表示测试模式，与test_*.sh的参数相同。可在引号中用空格分隔来输入多个测试模式。

`-c`: 可选，表示测试模型类别。仅测试输入的文件夹下的模型。可在引号中用空格分隔来输入多个文件夹。默认测试所有文件夹。

#### 例：

测试所有模型的fp32精度mlu单卡和多卡的性能：

`./validate_all_models.sh -m "fp32-mlu fp32-mlu-ddp"`

测试Classification和Detection模型的fp32精度GPU单卡性能：

`./validate_all_models.sh -m fp32-gpu -c "Classification Detection"`

## 二、上传测试结果

运行upload_results.sh将测试结果上传到superset数据库。运行前请备份results文件夹。

运行前需安装cndb库：

```shell
cd cndb
pip install -r requirments.txt
python setup.py install
cd ..
```

并需要保证soft_info.json和upload_results.sh中包含测试时使用的基础镜像的信息。

## 三、清理测试镜像

运行clear_all_docker.sh移除所有测试用镜像。