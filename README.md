# SegAttnFusion

This project implements a deep learning model for medical image analysis, combining segmentation and attention mechanisms to generate descriptive text from medical images.

## Project Structure

The project is organized into the following directories:

-   `src/`: Contains the core source code for the model, including the CNN encoder, word decoder, segmentation model, and attention mechanisms.
-   `data/`: The dataset directory. It should contain an `images` subdirectory and an `annotation.json` file.
-   `experiments/`: Contains scripts for training, evaluation, and logging. This includes the main training script, data loaders, metrics, and the training logger.
-   `results/`: Contains scripts for visualizing results and any output images.
-   `inference/`: Contains scripts for running inference on new images.
-   `utils/`: Contains utility scripts, such as for loading configuration files.

## Setup

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd SegAttnFusion
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training

To train the model, run the `main.py` script from the `experiments` directory:

```bash
python experiments/main.py --mode train
```

You can also specify other arguments, such as the batch size, learning rate, and number of epochs. For a full list of arguments, run:

```bash
python experiments/main.py --help
```

### Evaluation

To evaluate the model, run the `evaluation.py` script from the `experiments` directory:

```bash
python experiments/evaluation.py
```

### Inference

To run inference on a custom image, modify the `custom_image_paths` list in `inference/inference.py` and run the script:

```bash
python inference/inference.py
```

## Configuration

The `experiments/config.yaml` file contains the configuration for the model, training, and data. You can modify this file to change the model architecture, hyperparameters, and other settings.