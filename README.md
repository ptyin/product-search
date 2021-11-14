# Product Search

## 文件结构

- src: 模型代码
    - common: 通用组件，包括一些loss和metrics计算方法
    - HEM: HEM和QEM代码
        - **AmazonDataset.py**: 数据集文件转模型数据输入
        - **evaluate.py**: 测试
        - **run.py**: 模型实验入口
        - **Model.py**: 模型代码
    - AEM: AEM和ZAM代码
        - ...标准文件结构，含义同上。
    - LSE: LSE代码
        - ...
    - MetaSearch: 元学习方法
        - ...
    - GraphSearch: CIKM
        - ...
    - QL: QL和UQL代码
        - ...
    - TranSearchText: 单一文本模态模型代码
        - ...
- preprocess 数据预处理
    - preprocess.py: 预处理入口
    - neg_candidate: 负采样
    - doc2vec: 将review和query整理TranSearch的文档输入
    - word2vec: 生成w2v，目前无模型依赖，可用于实验
    - core: 预处理核心组件包，目录下文件顾名思义
    - transform @Deprecated: 可将csv格式数据输入转换为ESRT模型输入
- utils @Deprecated
- experiment_cf
    - CIKM实验相关，没太整理，建议重新写
- experiment_cold_start
    - metrics_bought: 根据用户购买量分类统计metrics
    - statistics: 统计数据集用户物品信息

## 数据预处理说明
```shell script
cd preprocess/
python preprocess.py --data_path <原数据存储路径> --dataset <数据集名> --processed_path <预处理后数据集文件路径> --unprocessed_path <只转换csv格式而不预处文件路径>
```

## 训练说明
```shell script
cd src/
python main.py <模型名> [--<参数键> <值>]
```

<模型名>可以是:ql, uql, lse, hem, aem, zam, tran_search, graph_search, meta_search
参数键在src/<模型>/run.py和common/data_preparation.py中定义

e.g., 训练hem模型在Automotive数据集上

```shell script
python main.py hem --dataset Automotive --save_str hem --debug --load
```
(save_str代表保存参数文件的文件名，开启debug开关代表启用进度条，否则不启用只输出log，load开关代表从模型保存位置加载)
其它具体参数可以通过以下查看：
```shell script
python main.py hem -h
```
