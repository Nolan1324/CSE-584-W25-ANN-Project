# Installation Instructions

We assume that you're running on a Linux based system. We often refer to external setup guides, just since the setup instructions for many of the required tools vary heavily depending on your specific operating system. Note that you'll require at least 16 gigabytes of RAM and 150 gigabytes of storage to download and run our datasets.

## Step 1: Install the Required Tools

1. Follow the [Docker installation guide](https://docs.docker.com/engine/install/) depending on your operating system and hardware.
2. Follow the [Milvus Docker guide](https://milvus.io/docs/install_standalone-docker.md) to install the Milvus docker image to the `bin` directory of the project.
3. Follow the [Python installation guide](https://www.python.org/downloads/) to install the correct version of Python 3.13 depending on your operating system and hardware.

## Step 2: Initialize the Python Environment

Using your Python package manager of choice, create a virtual environment and install the required packages from either the `pyproject.toml` or the `requirements.txt` file. For example, to use venv and pip run the following commands from the root directory of the project:
   1. `python -m venv .venv`
   2. `source .venv/bin/activate`
   3. `pip install -r requirements.txt`

## Step 3: Download and Unpack the Datasets

First, create the `data/datasets` directory.
To download either the `sift`, `siftsmall`, or `sift1b` datasets create a new directory in the `data` directory with the name of the datasets. The `sift` and `siftsmall` datasets can be easily downloaded using the `download_datasets.sh` script. For example,
```sh
# install siftsmall
mkdir siftsmall
cd siftsmall
../download_datasets.sh
cd ../
# install sift
mkdir sift
cd sift
../download_datasets.sh
cd ../
```
Due to its size, the `sift1b` dataset must be [downloaded from source](http://corpus-texmex.irisa.fr/) and manually unzipped. Due to its large size, this will take a while and you will need about 150 gigabytes of free space.

The directory structure should be as follows (if all three SIFT datasets have been correctly installed and unzipped):
```sh
data
├── download_datasets.sh
├── datasets
│   ├── sift
│   │   ├── sift_base.fvecs
│   │   ├── sift_groundtruth.ivecs
│   │   ├── sift_learn.fvecs
│   │   └── sift_query.fvecs
│   ├── sift1b
│   │   ├── bigann_base.bvecs
│   │   ├── bigann_learn.bvecs
│   │   ├── bigann_query.bvecs
│   │   └── gnd
│   └── siftsmall
│       ├── siftsmall_base.fvecs
│       ├── siftsmall_groundtruth.ivecs
│       ├── siftsmall_learn.fvecs
│       └── siftsmall_query.fvecs
```

### Step 4: Setup Configuration Files

You will need to save your sudo password to an environment file to allow the testing script to automatically restart docker containers if it detects problems. To do so, run the following command:
```sh
echo "PASSWORD=<your_sudo_password>" >> config/.env
```