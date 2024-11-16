# UNet Segmentation Project

This project implements a UNet model for image segmentation using the DRIVE dataset. The model is trained, validated, and tested on the dataset, and the results are logged and visualized.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Prepare the dataset and place it in the appropriate directory.

2. Run the training script:
    ```bash
    python tb.py
    ```

## Training

The training process includes:
- Splitting the dataset into training, validation, and test sets.
- Training the UNet model with early stopping based on validation loss.
- Logging the training and validation metrics.

## Evaluation

After training, the model is evaluated on the test set, and the results are logged.

## Results

The training and validation loss and DICE scores are plotted and saved. The trained model is saved as `unet_model.pth`.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
