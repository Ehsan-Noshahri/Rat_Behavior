This repository contains the code and instructions for replicating the experiments presented in the paper **[Classifying Rodent Behaviors One Frame at a Time: A CNN-Based Method](https://doi.org/10.1016/j.ecoinf.2026.103927)**.

## Main Requirements

The main Python requirements are:

* **PyTorch**: used for running and training the deep learning models.
* **OpenCV**: used for image processing, frame handling, and computer vision operations.
* **scikit-learn**, **scikit-image**, and **SciPy**: used for machine learning utilities, image analysis, feature extraction, and scientific computations.
* **ptflops**: used to estimate the computational complexity of deep models, including FLOPs and parameter counts.
* **Matplotlib** and **Seaborn**: used for plotting figures, visualizing results, and generating graphs.
* **regex**: used for file name handling and pattern-based processing.

## Video Models

The video-based experiments use action-recognition architectures such as **3D ResNet-18**, **(2+1)D ResNet-18**, **MC3**, **X3D**, and **MViT**.

These models are based on **PyTorchVideo**:

https://github.com/facebookresearch/pytorchvideo

PyTorchVideo can have specific compatibility requirements with PyTorch and TorchVision. For this reason, the video-model environment may need to be adjusted depending on the operating system, CUDA version, and GPU configuration.

## DINOv3 Models

The DINOv3 experiments use pretrained models from the official Meta AI DINOv3 repository:

https://github.com/facebookresearch/dinov3

The DINOv3 experiments were prepared in a separate environment with **Python 3.11**. Additional dependencies and pretrained weights are handled according to the official DINOv3 repository structure.

## Environment Setup

For the main frame-based CNN experiments, a minimal Conda environment can be created as follows:

```bash
conda create -n rat_behavior python=3.10
conda activate rat_behavior
```

PyTorch can then be installed according to the available hardware configuration using the official installation instructions:

https://pytorch.org/get-started/locally/

After installing PyTorch, the remaining general dependencies can be installed with:

```bash
pip install opencv-python scipy scikit-learn scikit-image matplotlib seaborn ptflops regex jupyterlab notebook
```

For the DINOv3 experiments, a separate Python 3.11 environment can be created:

```bash
conda create -n dinov3 python=3.11
conda activate dinov3
```

The remaining DINOv3 dependencies and pretrained weights can be obtained from the official DINOv3 repository:

https://github.com/facebookresearch/dinov3

Optional dependencies for video models and DINOv3 are only required for the corresponding experiments.

## Notes

The complete environment may vary depending on the operating system, CUDA version, GPU driver, and PyTorch/TorchVision compatibility. Therefore, the listed requirements focus on the core dependencies used in the experiments, while PyTorchVideo and DINOv3 are kept as separate optional components.

## Citation

If you find this repository useful in your research, please consider citing the following article:

```bibtex
@article{noshahri2026rodent,
  title   = {Classifying rodent behaviors one frame at a time: A CNN-based method},
  author = {Ehsan Noshahri and Andres Molares-Ulloa and Carlota Fraga-Seijas
            and Alejandro Puente-Castro and Alvaro Rodriguez},
  journal = {Ecological Informatics},
  volume  = {97},
  pages   = {103927},
  year    = {2026},
  issn    = {1574-9541},
  doi     = {10.1016/j.ecoinf.2026.103927},
  url     = {https://www.sciencedirect.com/science/article/pii/S1574954126003341}
}
