# 废钢通用模型服务

本服务基于[Flask-restful](https://flask-restful.readthedocs.io/en/latest/index.html)框架，用于废钢模型的快速部署

## 开发思路
todo:
## 安装环境
todo：Dockerfile编写， requirements.txt编写

## 如何基于本框架部署你的模型
### 部署你的模型
* 在 [_custom_](custom) 文件夹中，完成 [_custom_model.py_](custom/custom_model.py)
* 需继承 [_base_model.py_](core/base_model.py) 中 _BaseModel_ 和 _BaseResourceInit_ 两个父类，实现父类中所规定的必须实现的函数， 如 _preprocess_， _predict_ 等。
* 请勿更改 [_core_](core) 中任何代码
### 填写配置文件
* 参考 [_template_config.json_](config/template_config.json)，填写[_config.json_](config/config.json)
## 运行
```
python main.py
```

## 调用格式
todo: 规范数据格式
## 测试脚本
todo: 编写标准测试脚本