## *SPIN*: Sparsifying and Integrating Internal Neurons in Large Language Models for Text Classification


[![ACL'24](https://img.shields.io/badge/ACL'24-Findings-8c1b13.svg?logo=data:image/svg%2bxml;base64,PHN2ZyBoZWlnaHQ9IjI2MC4wOTA0ODIiIHZpZXdCb3g9IjAgMCA2OCA0NiIgd2lkdGg9IjM4NC40ODE1ODIiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHBhdGggZD0ibTQxLjk3NzU1MyAwdjMuMDE1OGgtMzQuNDkwNjQ3Ni03LjQ4NjkwNTR2Ny40ODQ5OSAyNy45Nzc4OCA3LjUyMTMzaDcuNDg2OTA1NCA0Mi4wMTM4OTY2IDcuNDg2OTA2IDExLjAxMjI5MnYtMTUuMDA2MzJoLTExLjAxMjI5MnYtMjAuNDkyODktNy40ODQ5OWMwLTEuNTczNjkgMC0xLjI1NDAyIDAtMy4wMTU4em0tMjYuOTY3Mzk4IDE3Ljk4NTc4aDI2Ljk2NzM5OHYxMy4wMDc5aC0yNi45NjczOTh6IiBmaWxsPSIjZmZmZmZmIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiLz48L3N2Zz4=)](https://aclanthology.org/2024.findings-acl.277/)
[![arXiv](https://img.shields.io/badge/arXiv-2311.15983-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2311.15983)
[![Web Demo](https://img.shields.io/badge/GitHub-Web_Demo-seagreen.svg?logo=github)](https://liuyilun2000.github.io/spin-visualization/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)

Official code repository for our paper [***SPIN*** **: Sparsifying and Integrating Internal Neurons in Large Language Models for Text Classification**](https://aclanthology.org/2024.findings-acl.277/) by Difan Jiao, Yilun Liu, Zhenwei Tang, Daniel Matter, Jürgen Pfeffer and Ashton Anderson. 
This repository hosts all experimental infrastructure essential for the paper.

To visually explore how *SPIN* works, pleaase visit our [interactive visualization web demo](https://liuyilun2000.github.io/spin-visualization/).

### Citation
We would be delighted if our provided resources has been useful in your research or development! 🥰 If so, please consider citing our paper:

```bibtex
@inproceedings{jiao-etal-2024-spin,
    title = "{SPIN}: Sparsifying and Integrating Internal Neurons in Large Language Models for Text Classification",
    author = {Jiao, Difan  and
      Liu, Yilun  and
      Tang, Zhenwei  and
      Matter, Daniel  and
      Pfeffer, J{\"u}rgen  and
      Anderson, Ashton},
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.277",
    pages = "4666--4682",
    abstract = "Among the many tasks that Large Language Models (LLMs) have revolutionized is text classification. Current text classification paradigms, however, rely solely on the output of the final layer in the LLM, with the rich information contained in internal neurons largely untapped. In this study, we present SPIN: a model-agnostic framework that sparsifies and integrates internal neurons of intermediate layers of LLMs for text classification. Specifically, SPIN sparsifies internal neurons by linear probing-based salient neuron selection layer by layer, avoiding noise from unrelated neurons and ensuring efficiency. The cross-layer salient neurons are then integrated to serve as multi-layered features for the classification head. Extensive experimental results show our proposed SPIN significantly improves text classification accuracy, efficiency, and interpretability.",
}
```
