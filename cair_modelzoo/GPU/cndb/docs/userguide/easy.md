# CNDB EASY 模块

Easy 模块主要提供如下功能：

* 获取物理机信息
* 获取 Tensorflow 软件版本信息
* 获取 Pytorch 软件版本信息（ongoing）

## API 描述

**dump_tf_info**

```text
dump_tf_info(name, file=None)
    Dump TensorFlow and dependent libraries version to STDOUT or file in YAML format.
    
    Get TensorFlow and Cambricon Neuware Libraries version information, include
    
    - "tf": TensorFlow version
    - "camb_tf": Cambricon Neuware version
    - "driver": MLU Driver version
    - "cnrt": CNRT version
    - "cnnl": CNNL version
    - "cnml": CNML version
    
    Args:
        name: information name
        file: output filename. Print to stdout when None.
    
    Returns:
        None
```

**dump_mlu_machine_info**

```text
dump_mlu_machine_info(name=None, file=None)
    Dump MLU Machine information to STDOUT or file in YAML format.
    
    Get machine information include
    
    - name: machine name
    - cpu: cpu information, include socket number, logical core number
    - mem: total memory
    - dev: mlu information, include mlu name, number, pcie capability
    
    Args:
        name: information name. Use platform node name if None.
        file: output filename. Print to stdout when None.
    
    Returns:
        None
```

**get_mlu_name**

```text
get_mlu_name()
    MLU device name
    
    Get device name from '/proc/driver/cambricon/mlus/*/information'
    
    Returns:
        Device name or 'unknown'
```
