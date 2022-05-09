# Product Search

> SOTA Product Search models implementation in PyTorch .

## Implemented Models

> - [LSE] Christophe Van Gysel, Maarten de Rijke, and Evangelos Kanoulas. 2016. Learning Latent Vector Spaces for Product Search. In Proceedings of the 25th ACM International on Conference on Information and Knowledge Management. ACM, 165–174
> - [HEM] Qingyao Ai, Yongfeng Zhang, Keping Bi, Xu Chen, and W. Bruce Croft. 2017. Learning a Hierarchical Embedding Model for Personalized Product Search. In Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval. ACM, 645–654.
> - [TranSearch] Yangyang Guo, Zhiyong Cheng, Liqiang Nie, Xin-Shun Xu, and Mohan Kankanhalli. 2018. Multi-Modal Preference Modeling for Product Search. In Proceedings of the 26th ACM International Conference on Multimedia. ACM, 1865–1873.
> - [AEM/ZAM] Qingyao Ai, Daniel N. Hill, S. V. N. Vishwanathan, and W. Bruce Croft. 2019. A Zero Attention Model for Personalized Product Search. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management. ACM, 379–388.

## File Structure

- src: Sources root
    - common: Common components，including computations of common loss and metrics
    - HEM: Hierarchical Embedding Model
        - AmazonDataset.py: Dataset input to model feed
        - evaluate.py: Model testing
        - run.py: Model entry
        - Model.py: Model implementation
    - AEM: Attention Embedding Model & Zero Attention Embedding Model
    - LSE: Latent Semantic Entity
    - MetaSearch: A model with meta learning methods (not published) 
    - **GraphSearch**:  A model with hierarchical heterogeneous graph neural network methods (**HHGNN**, not published) 
    - QL: QL & UQL
    - TranSearchText: Transearch with only text modality
- preprocess: Data preprocess directory
    - preprocess.py: Preprocess entry
    - neg_candidate: Negative sampling
    - doc2vec: Convert queries and reviews to doc vectors
    - word2vec: Convert word corpus to word vectors
    - core
    - transform @Deprecated: transform CSV formatted dataset file to [ESRT](https://github.com/utahIRlab/ESRT) formatted dataset files.

## Preprocess
```shell script
cd preprocess/
python preprocess.py --data_path <原数据存储路径> --dataset <数据集名> --processed_path <预处理后数据集文件路径> --unprocessed_path <只转换csv格式而不预处文件路径>
```

## Model training
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
