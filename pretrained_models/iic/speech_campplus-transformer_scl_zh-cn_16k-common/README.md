---
tasks:
- speaker-diarization
model_type:
- CAM++-transformer
domain:
- audio
frameworks:
- pytorch
backbone:
- CAM++-transformer
license: Apache License 2.0
language:
- cn
tags:
- speaker change locating
- 两人转换点定位
- 中文模型
widgets:
  - task: speaker-diarization
    model_revision: v1.0.0
    inputs:
      - type: audio
        displayType: AudioUploader
        name: input
        title: 音频
        validator:
          max_size: 1M
    examples:
      - inputs:
          - data: git://examples/scl_example1.wav
      - inputs:
          - data: git://examples/scl_example2.wav
    output:
        displayType: Text
    inferencespec:
      cpu: 8 #CPU数量
      memory: 1024
---

# 基于CAM++和transformer的说话人转换点定位模型
该模型分为说话人特征提取模型和说话人转换点定位模型，说话人特征提取模型采用CAM++结构，转换点定位模型采用transformer结构。该模型适合两人对话场景下的转换点定位，适合作为说话人日志（speaker diarization）系统的后端，在说话人转换点附近精确的定位转换点位置，减少时间误差。

## 模型简述
该模型可用于改善说话人日志系统中经常存在的说话人转换点定位不准的情况。该模型结果主要分为两部分。第一部分是说话人特征提取模型，采用CAM++结构，是为了提取混合音频中的说话人特征。第二部分是transformer结构，输入说话人的帧级特征，利用self-attention机制预测发生转换点的语音帧位置，模型可额外输入预先提取的说话人全局特征embedding，可获得更准确的帧级预测结果。

## 训练数据
本模型使用大规模的中文两人合成音频数据集进行训练。
## 模型效果评估
采用预测转换时间点和实际转换时间点平均误差作为评估标准。
| 测试集 | 时间误差（秒） |
|:-----:|:------:|
|合成测试集|0.03|
|真实测试集|0.2|

# 如何快速体验模型效果
如果在本地使用，需要先安装modelscope并配置相应的环境，相关教程请参考[这里](https://modelscope.cn/docs/%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85)。
对于有开发需求的使用者，特别推荐您使用Notebook进行离线处理。先登录ModelScope账号，点击模型页面右上角的“在Notebook中打开”按钮出现对话框，首次使用会提示您关联阿里云账号，按提示操作即可。关联账号后可进入选择启动实例界面，选择计算资源，建立实例，待实例创建完成后进入开发环境，输入api调用实例。
``` python
from modelscope.pipelines import pipeline
scl_pipeline = pipeline(
    task='speaker-diarization',
    model='damo/speech_campplus-transformer_scl_zh-cn_16k-common',
    model_revision='v1.0.0'
)
input_wav = 'https://modelscope.cn/api/v1/models/damo/speech_campplus-transformer_scl_zh-cn_16k-common/repo?Revision=master&FilePath=examples/scl_example1.wav'
result = scl_pipeline(input_wav)
print(result)
# 一般情况下，在speaker diarization系统中会获得两人的全局embedding，如果额外输入两人的全局embedding会得到更准确的预测结果，例如：
# result = scl_pipeline(input_wav, embds=[emb1, emb2])
# print(result)
```
# 模型的局限性
当前模型只适用于两人对话、且一定存在转换点的场景下，对转换点进行定位。其他场景的说话人转换点定位模型正在开发中。