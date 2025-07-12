# This repo is created for the EMBC 2025 paper "Towards Automated EEG-Based Epilepsy Detection Using Deep Convolutional Autoencoders"

# Example Usage Guide

## Installation
Make sure you have all dependencies installed:

```bash
pip install -r requirements.txt
```     

## Example Usage

The provided example data is already preprocessed (resampling, filtering, montage) and is only a very small subset of the original dataset. It is from the TUH EEG Seizure Corpus (dev set) and contain 4 windows with 2 seconds of EEG data each. The labels are balanced, with 2 windows labeled as seizure (label=1) and 2 as non-seizure (label=0).

For a complete example, you can run the following script:

```bash
python example_usage.py
``` 

## References
Author: Annika Stiehl

Email: [annika.stiehl@hs-ansbach.de](mailto:annika.stiehl@hs-ansbach.de)