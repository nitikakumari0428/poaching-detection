import streamlit as st
import cv2
import os
from ultralytics import YOLO
import pickle
import numpy as np
from deepface import DeepFace
from retinaface import RetinaFace



import importlib

# List of packages to check
packages = {
    "streamlit": "1.40.2",
    "ultralytics": None,
    "opencv-python-headless": None,
    "numpy": None,
    "deepface": None,
    "retinaface": "1.1.1",
    "tensorflow": ">=2.6.0"
}

for package, required_version in packages.items():
    try:
        # Dynamically import the package
        pkg = importlib.import_module(package)
        installed_version = pkg.__version__

        if required_version:
            # Compare the versions if required version is specified
            from packaging import version
            if version.parse(installed_version) >= version.parse(required_version.replace(">=", "")):
                print(f"{package}: {installed_version} (meets requirement: {required_version})")
            else:
                print(f"{package}: {installed_version} (does not meet requirement: {required_version})")
        else:
            print(f"{package}: {installed_version} (no specific version requirement)")
    except ImportError:
        print(f"{package} is not installed.")
    except AttributeError:
        print(f"{package} is installed but version could not be determined.")
