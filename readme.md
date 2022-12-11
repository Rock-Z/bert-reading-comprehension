# Dual Co-Matching Network for multiple choice questions

This repo contains an re-implementation of Dual Co-Matching Network for multiple choice questions introduced in [this](https://arxiv.org/pdf/1908.11511.pdf) paper. The model uses `BERT-Small` with 4 layers of encoders and a hidden size of 256, and achieves accuracy around 0.6138. The model definition is stored in `model.py`, and `main.py` is the script for running the training task. The current model is trained with a subset of the high school questions in the RACE dataset.

## References

Original Paper:

```
@inproceedings{dcmn,
    title={DCMN+: Dual Co-Matching Network for Multi-choice Reading Comprehension},
    author={Shuailiang Zhang and Hai Zhao and Yuwei Wu and Zhuosheng Zhang and Xi Zhou and Xiang Zhou},
    year={2020},
    booktitle = "{The Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI)}",
}
```

Dataset:

```
@article{lai2017large,
    title={RACE: Large-scale ReAding Comprehension Dataset From Examinations},
    author={Lai, Guokun and Xie, Qizhe and Liu, Hanxiao and Yang, Yiming and Hovy, Eduard},
    journal={arXiv preprint arXiv:1704.04683},  
    year={2017}
}
```