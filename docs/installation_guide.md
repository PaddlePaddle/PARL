# Installation Guide

## Detailed Installation Steps

1. **Environment Preparation**
    - Supported Python versions: 3.7 - 3.10 (tested on Linux systems).

2. **Install the DL Framework**
   - For the CPU version, simply run the following command:
     ```bash
     pip install paddlepaddle
     ```
   - For the GPU version:
     - On Linux, the maximum supported version is 2.5:
       ```bash
       pip install paddlepaddle-gpu==2.5
       ```
     - On Windows, the maximum supported version is 2.2.1:
       ```bash
       pip install paddlepaddle-gpu==2.2.1
       ```

3. **Adjust Numpy Version**
   - After installing PaddlePaddle, if the current numpy version is higher than 1.23.5, reinstall numpy:
     ```bash
     pip install numpy==1.23.5
     ```

4. **Install PARL and Gym**
   - Run the following command to install the latest versions of PARL and Gym:
     ```bash
     pip install parl gym
     ```

5. **Test the Installation**
   - Use the following command to run the quick-start test script:
     ```bash
     python examples/QuickStart/train.py
     ```

---
