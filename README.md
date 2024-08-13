## *SPIN*: Sparsifying and Integrating Internal Neurons in Large Language Models for Text Classification

[![Python 3.10](https://img.shields.io/badge/python-3.10-lightgrey.svg?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)
[![GitHub Code](https://img.shields.io/badge/GitHub-Code_Repo-blue.svg?logo=github)](https://github.com/difanj0713/SPIN/)
[![Web Demo](https://img.shields.io/badge/GitHub-Web_Demo-seagreen.svg?logo=github)](https://liuyilun2000.github.io/spin-visualization/)
[![OpenReview](https://img.shields.io/badge/OpenReview-ACL_2024_Findings-8c1b13.svg?logo=openreview)](https://aclanthology.org/2024.findings-acl.277/)
[![arXiv](https://img.shields.io/badge/arXiv-2311.15983-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2311.15983)

Official code repository for our paper [***SPIN*** **: Sparsifying and Integrating Internal Neurons in Large Language Models for Text Classification**](https://aclanthology.org/2024.findings-acl.277/) by Difan Jiao, Yilun Liu, Zhenwei Tang, Daniel Matter, Jürgen Pfeffer and Ashton Anderson.

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
