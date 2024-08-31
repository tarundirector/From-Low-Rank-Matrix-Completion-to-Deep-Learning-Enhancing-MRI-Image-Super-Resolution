# From-Low-Rank-Matrix-Completion-to-Deep-Learning-Enhancing-MRI-Image-Super-Resolution

## 📚 Repository Overview

This repository provides the complete implementation of a study comparing two methodologies—Low-Rank Matrix Completion and Deep Learning—for Magnetic Resonance Imaging (MRI) image super-resolution (SR). The aim is to enhance MRI images by improving their resolution, critical for accurate diagnosis and treatment planning in medical imaging.
## 🧠 Dataset Details

The study uses the IXI Dataset, a well-known collection of MRI scans comprising T1-weighted, T2-weighted, Proton Density-weighted, Magnetic Resonance Angiography (MRA), and Diffusion-weighted images. For more details, visit the IXI Dataset website. The preprocessing steps involve extracting mid-slice images from the T2-weighted modality, which are then used for training and evaluation. Preprocessing code is provided in the DL_ELRTV_main.ipynb file. Additionally, the preprocessed mid-slice image dataset is included in this repository.
## 🔄 Reproducing the Experiments

To reproduce the experiments, set up Google Colab with the following libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `optuna`
- `pickle`
- `skimage.metrics`
- `tqdm`
- `tensorboardX`
- `torchmetrics`
- `cv2` (OpenCV)
- `scipy.ndimage` (from SciPy)
- `torch` (PyTorch) 

#### 🧩 Low-Rank Approach
- **Environment**: Conducted on a high RAM CPU environment in Google Colab.
- **Hyperparameters**: Optimized using the Optuna framework. Key parameters include `lambda_tv_base`, `lambda_rank_base`, and `rho`.
- **Upscaling Steps**: Set to 2 for both 2x and 4x SR tasks.

#### 🤖 Deep Learning Approach
- **Environment**: Conducted on a Google Colab environment with a T4 GPU (15.0 GB) and high RAM (51.0 GB).
- **Configurations**:
  - **2x SR**: Input size 128×128, output size 256×256. Uses LIIF+EDSR model.
  - **4x SR**: Input size 64×64, output size 256×256. Uses LIIF+EDSR model.
  - **Off-Domain SR**: Trained on 64×64 to 128×128 (2x), tested on 64×64 to 256×256 (4x).

#### Common Settings for Deep Learning:
- **MLP Size**: 128-unit MLP for 2x, 128 and 256-unit MLP for 4x.
- **Optimizer**: Adam with a learning rate of 1.e-4.
- **SMAM Configuration**: Multi-head attention with three heads (1×1, 3×3, 5×5 kernels).

### 📊 Running the Experiments

- **Low-Rank Approach**: Run the provided code in `DL_ELRTV_main.ipynb` for hyperparameter tuning or testing.
- **Deep Learning Approach**:
  - **Training**: 
    ```
    !python train_liif.py --config [yaml file path] --name [model name]
    ```
    - YAML files are in `configs/train-IXI-MRI`. Use files as per the required model.
  - **Testing**:
    ```
    !python test.py --config [yaml file path] --model [model path]
    ```
    - Model paths are in the `/save` folder.
  - **Demo**:
    ```
    !python demo.py --input [input image path] --model [model path] --resolution [desired output resolution] --output [output path] --gpu 0
    ```
## 📈 Results Summary


| Method                                             | Score Improvement (PSNR/SSIM) x2 | Score Improvement (PSNR/SSIM) x4 | Reported Score 2x (PSNR/SSIM) | Reported Score 4x (PSNR/SSIM) |
|----------------------------------------------------|----------------------------------|----------------------------------|--------------------------------|--------------------------------|
| ELRTV                                              | +3.45 / +0.0275                  | +1.01 / +0.0208                  | 32.54/0.9430                   | 26.89/0.8456                   |
| LIIF (EDSR baseline, 128 MLP) + SMAM + EEM         | +8.59 / +0.0557                  | +4.01 / +0.0788                  | 37.68/0.9712                   | 29.89/0.9036                   |
| LIIF (EDSR baseline, 256 MLP) + SMAM + EEM         | -                                | +4.1 / +0.0798                   | -                              | 29.98/0.9046                   |


| Method                                             | Epochs | Time (Train/Test) | Train PSNR & Loss               |
|----------------------------------------------------|--------|-------------------|---------------------------------|
| ELRTV 2x                                           | -      | - / 37s           | -                               |
| ELRTV 4x                                           | -      | - / 51s           | -                               |
| LIIF (EDSR baseline, 128 MLP) + SMAM + EEM [2x]    | 32     | 8.0h / 5s         | 36.9305 / loss=0.0150           |
| LIIF (EDSR baseline, 128 MLP) + SMAM + EEM [4x]    | 88     | 12h / 5s          | 28.9726 / loss=0.0281           |
| LIIF (EDSR baseline, 256 MLP) + SMAM + EEM [4x]    | 51     | 4.15h / 5s        | 29.1495 / loss=0.0281           |
| LIIF (EDSR baseline, 128 MLP) + SMAM + EEM [Off-domain] | 34 | 2.1h / 5s         | 37.3636 / loss=0.0130           |

These tables summarize the performance and computational requirements of different methods explored in this study. The results show the efficacy of deep learning models in enhancing MRI images, making them a viable solution for high-resolution medical imaging.

Bicubic baseline scores: 29.09/0.9155 for 2x, 25.88/0.8248 for 4x.

---

![ComparitiveFigure-2x](https://github.com/user-attachments/assets/1bb9e97f-236c-4391-acaa-45caa1ddd977)
**FIGURE**: Qualitative results and error maps for 2x SR using the IXI dataset; Top row: PSNR vs. Epochs plot showing the progression of PSNR values over training epochs (left) and Error maps generated by each method, with warmer colours (cmap='jet') indicating higher errors (right); Middle row: Super-resolved images produced by each method; Bottom row: Zoomed-in views of the super-resolved images for detailed comparison.

---

![ComparitiveFigure-4x](https://github.com/user-attachments/assets/3680ffd3-ec73-4e62-acc9-0a27719f63aa)
**FIGURE**: Qualitative results and error maps for 4x SR using the IXI dataset; Top row: PSNR vs. Epochs plot showing the progression of PSNR values over training epochs (left) and Error maps generated by each method, with warmer colours (cmap='jet') indicating higher errors (right); Middle row: Super-resolved images produced by each method; Bottom row: Zoomed-in views of the super-resolved images for detailed comparison.

---

![ComparitiveFigure-4x-OOD](https://github.com/user-attachments/assets/750a80c9-cdb5-453d-a6f7-2597490ecd0a)
**FIGURE**: Qualitative results and error maps for 2x off-domain SR using the IXI dataset; Top row: PSNR vs. Epochs plot showing the progression of PSNR values over training epochs (left) and Error maps generated by each method, with warmer colours (cmap='jet') indicating higher errors (right); Middle row: Super-resolved images produced by each method; Bottom row: Zoomed-in views of the super-resolved images for detailed comparison.
 
