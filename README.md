# LEA-GCN

<p align="left">
  <img src='https://img.shields.io/badge/python-3.6+-blue'>
  <img src='https://img.shields.io/badge/Tensorflow-1.12+-blue'>
  <img src='https://img.shields.io/badge/NumPy-1.16-brightgreen'>
  <img src='https://img.shields.io/badge/pandas-0.22.0-brightgreen'>
  <img src='https://img.shields.io/badge/scipy-1.5.3-brightgreen'>
</p> 

## **Overall description** 
Here presents the code of LEA-GCN. We upload the relevant datasets (i.e., Douban and Amazon) on Bitbucket: [https://bitbucket.org/jinyuz1996/lea-gcn-data/src/main/](https://bitbucket.org/jinyuz1996/lea-gcn-data/src/main/). You should download them before running LEA-GCN. The code is attached to our paper: **"Towards Lightweight Cross-domain Sequential Recommendation via External Attention-enhanced Graph Convolution Network" (DASFAA 2023)**. If you want to use our codes or datasets in your research, please cite our paper. Note that, our paper is still in the state of 'Unavailable' before April 17, 2023 (i.e., the important date of the main conference, DASFAA, 2023), but you can get the preprint version on Arxiv: [https://arxiv.org/abs/2302.03221](https://arxiv.org/abs/2302.03221).
## **Code description** 
### **Vesion of implements and tools**
1. python 3.6
2. tensorflow 1.12.0
3. scipy 1.5.3
4. numpy 1.16.0
5. pandas 0.22.0
6. matplotlib 3.3.4
7. Keras 1.0.7
8. tqdm 4.60.0
### **Source code of LEA-GCN**
1. the definition of LEA-GCN see: LEA-GCN/LEA_Model.py
2. the definition of training process see: LEA-GCN/LEA_Train.py
3. the definition of Evaluating process see: LEA-GCN/LEA_Evaluate.py
4. the preprocess of dataset see: LEA-GCN/LEA_Config.py
5. the parameter settings of LEA-GCN see: LEA-GCN/LEA_Setting.py
6. to run the training method see: LEA-GCN/LEA_Main.py and the training log printer was defined in: LEA-GCN/LEA_Printer.py

** Note that, the directory named Checkpoint is used to save the trained recommender systems, and you should establish an folder named "checkpoint" in the same level of above code files to make sure that your model can be saved. 
    



