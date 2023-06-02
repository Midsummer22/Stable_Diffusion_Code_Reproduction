# Stable_Diffusion_Code_Reproduction
Stable Diffusion code reproduction.Sorry for the poor coding skills!

Sure, I'd be happy to help you translate this README into English. Here it is:

# Stable Diffusion Code Reproduction

This repository contains the diffusers and stable diffusion model parameters publicly released by the authors, as well as code for using the stable diffusion model and a video demonstrating how to use it. There are two main functions: generating images based on text prompts, and generating images based on given images and text prompts (the latter has less satisfactory results).

## Installation

- Linux system
- Python 3.6 or higher
- NVIDIA GPU with at least 10GB of memory

To install the dependencies, run the following commands in a Linux terminal:

```
git clone https://github.com/huggingface/diffusers.git
pip install diffusers
cd diffusers
pip install .
```

## Usage

To use the application, run the following commands:

1. Generating images based on text prompts:

```
cd ..
python txt2jpg_inference.py
```

This will generate a PNG image named "txt2jpg.png" in the current directory.

2. Generating images based on images and text prompts:

```
python jpg2jpg_inference.py
```

This will read an input file named "car.png" and write the processed result to an output file named "jpg2jpg.png".

## Contributing

If you would like to contribute to this project, you can fork this repository and send a pull request to the original author. They welcome any form of contribution, including code modifications, documentation improvements, and error fixes. This is someone else's project that I used to reproduce, and I thank them for providing the project code (I don't know their GitHub account).

I hope this README will help you use the application better. If you have any questions or concerns, please feel free to ask me.
