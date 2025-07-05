# VTON-X: A Modernized Virtual Try-On Project

> **Note:** This project is a customized and modernized fork of the official [CP-VTON+ (CVPRW 2020)](https://github.com/minar09/cp-vton-plus) implementation. All credit for the core research and model architecture belongs to the original authors.

This repository, **VTON-X**, aims to make the powerful CP-VTON+ model more accessible and easier to set up by leveraging modern Python development tools and providing a streamlined process for testing and inference.

[Original Project Page](https://minar09.github.io/cpvtonplus/) | [Original Paper](https://minar09.github.io/cpvtonplus/cvprw20_cpvtonplus.pdf)

---

## What's New in VTON-X?

This fork focuses on improving the developer experience and modernizing the project's foundation. Key changes include:

- **Modern Dependency Management:** Uses **`pipenv`** for robust, reproducible environments, replacing the `requirements.txt` file.
- **Updated Libraries:** Configured to work with newer library versions, including `PyTorch >= 1.10` and `Torchvision >= 0.11`.
- **Simplified Setup:** A clear, step-by-step guide to get the demo running in minutes.
- **Focus on Inference:** The primary goal of this fork is to make it easy to run the pre-trained models on new images.

## Quickstart: Run the Demo

Get the virtual try-on running with just a few commands.

#### **Prerequisites:**

- Python 3.6+
- Git
- `pipenv` (`pip install pipenv`)
- An NVIDIA GPU with a compatible CUDA version installed is highly recommended.

#### 1. Clone the Repository

```bash
git clone [repo-url]
cd VTON-X
```

#### 2. Setup the Environment with Pipenv

### TODO

#### 4. Run the Try-On Inference

The pipeline runs in two stages: GMM (warping the cloth) and TOM (generating the final image).

```bash
# Stage 1: Run the Geometric Matching Module (GMM)
python test.py --name gmm_test --stage GMM --workers 4 --dataroot ./data --data_list test_pairs.txt --checkpoint checkpoints/gmm.pth

# Stage 2: Run the Try-On Module (TOM)
python test.py --name tom_test --stage TOM --workers 4 --dataroot ./data --data_list test_pairs.txt --checkpoint checkpoints/gen.pth
```

#### 5. View Your Results!

The final images are saved in the `results/tom_test/test/try-on/` directory.

---

## Advanced Usage (Training)

While this fork focuses on inference, the original training scripts are preserved. To train the models from scratch, please follow the logic from the original repository:

1.  **Prepare Data:** Download the full `VITON_PLUS` dataset and prepare it in the `data` directory as described in the original `README`.
2.  **Train GMM:** Run `train.py` for the GMM stage.
    ```bash
    pipenv shell
    python train.py --name GMM_train --stage GMM --workers 4 --save_count 5000 --shuffle
    ```
3.  **Generate Warped Clothes:** Use the trained GMM to generate warped clothes for the training set by running `test.py`.
4.  **Train TOM:** Run `train.py` for the TOM stage, using the newly generated warped clothes as input.
    ```bash
    pipenv shell
    python train.py --name TOM_train --stage TOM --workers 4 --save_count 5000 --shuffle
    ```

## Using Custom Images

To run the model on your own images, follow these steps:

1.  **Prepare Inputs:** You need to generate several input files for each try-on pair. See the original authors' notes on this in the section below.
    - `image` (person image, 256x192)
    - `cloth` (clothing image, 256x192)
    - `image-parse` (person segmentation map)
    - `cloth-mask` (binary mask of the cloth)
    - `pose` (pose keypoints JSON file)
2.  **Organize Files:** Place your files in the corresponding subdirectories within the `data/` folder.
3.  **Update Pair List:** Add a new line to `data/test_pairs.txt` with the filenames of your person image and cloth image (e.g., `my_photo.jpg my_shirt.png`).
4.  **Run Inference:** Execute the commands from the **Quickstart** Step 4.

> #### _Notes from the Original Authors on Custom Images:_
>
> - You can generate `image-parse` with pre-trained networks like [CIHP_PGN](https://github.com/Engineering-Course/CIHP_PGN) or [Graphonomy](https://github.com/Gaoyiminggithub/Graphonomy).
> - `cloth-mask` can be generated with simple image processing functions in Pillow or OpenCV.
> - `pose` keypoints can be generated using the official [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) repository (COCO-18 model).

---

## Citation & Acknowledgements

This project would not be possible without the foundational work of the original `CP-VTON+` authors. If this code helps your research, please cite their paper:

```
@InProceedings{Minar_CPP_2020_CVPR_Workshops,
	title={CP-VTON+: Clothing Shape and Texture Preserving Image-Based Virtual Try-On},
	author={Minar, Matiur Rahman and Thai Thanh Tuan and Ahn, Heejune and Rosin, Paul and Lai, Yu-Kun},
	booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
	month = {June},
	year = {2020}
}
```

This implementation is also heavily based on the original [CP-VTON](https://github.com/sergeywong/cp-vton). We are extremely grateful for their public implementation, which laid the groundwork for this entire line of research.
