# BoostAdapter: Improving Test-Time Adaptation via Regional Bootstrapping

### [[Paper](https://arxiv.org/abs/2410.15430)] 




[Taolin Zhang](https://scholar.google.com/citations?user=DWnu_G0AAAAJ),  [Jinpeng Wang](https://scholar.google.com/citations?user=853-0n8AAAAJ&hl=zh-CN), [Hang Guo](https://scholar.google.com/citations?user=fRwhfpoAAAAJ&hl=zh-CN), [Tao Dai](https://scholar.google.com/citations?user=MqJNdaAAAAAJ&hl=zh-CN), [Bin Chen](https://scholar.google.com.hk/citations?user=Yl0wv7AAAAAJ&hl=zh-CN) and [Shu-Tao Xia](https://scholar.google.com/citations?user=koAXTXgAAAAJ)

> **Abstract:**  Adaptation of pretrained vision-language models such as CLIP to various downstream tasks have raised great interest in recent researches. Previous works have proposed a variety of test-time adaptation (TTA) methods to achieve strong generalization without any knowledge of the target domain. However, existing training-required TTA approaches like TPT necessitate entropy minimization that involves large computational overhead, while training-free methods like TDA overlook the potential for information mining from the test samples themselves. In this paper, we break down the design of existing popular training-required and training-free TTA methods and bridge the gap between them within our framework. Specifically, we maintain a light-weight key-value memory for feature retrieval from instance-agnostic historical samples and instance-aware boosting samples. The historical samples are filtered from the testing data stream and serve to extract useful information from the target distribution, while the boosting samples are drawn from regional bootstrapping and capture the knowledge of the test sample itself. We theoretically justify the rationality behind our method and empirically verify its effectiveness on both the out-of-distribution and the cross-domain datasets, showcasing its applicability in real-world situations.


<p align="center">
    <img src="assets/framework.png" style="border-radius: 15px">
</p>


## <a name="installation"></a> Installation

This codebase was tested with the following environment configurations. It may work with other versions.

- Python 3.9
- CUDA 11.8
- PyTorch 2.2.0 + cu118

## Datasets
Please follow [Datasets](docs/DATASETS.md) to download the OOD and Cross-Domain benchmarks. 


## <a name="training"></a>  Running

Run the following script to run BoostAdapter:
```
bash scripts/run.sh $GPU $EXP_NAME
```

For example, you can run the a demo with the following command:  
```
bash scripts/run.sh 0 demo
```


## <a name="cite"></a> Citation

Please cite us if our work is useful for your research.

```
@article{zhang2024boostadapter,
  title={BoostAdapter: Improving Test-Time Adaptation via Regional Bootstrapping},
  author={Zhang, Taolin and Wang, Jinpeng and Guo, Hang and Dai, Tao and Chen, Bin and Xia, Shu-Tao},
  journal={arXiv preprint arXiv:2410.15430},
  year={2024}
}
```

## License

This project is released under the [MIT license](LICENSE).

## Acknowledgement

This code is based on [TDA](https://github.com/kdiAAA/TDA). Thanks for their awesome work.

## Contact

If you have any questions, feel free to approach me at zhangtlin3@gmail.com.
