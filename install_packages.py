"""
install_packages.py: install python packages.

Hyungwon Yang
24.06.10
MediaZen
"""
import subprocess

# List of packages to install normally
normal_packages = ["numpy>=1.19.1",
"torch>=1.10",
"torchvision>=0.8.2",
"black",
"einops",
"omegaconf",
"matplotlib",
"gradio>=3.5.0",
"onnx==1.14.1",
"diffusers",
"transformers==4.41.1",
"accelerate",
"safetensors",
"fsspec",
"tritonclient[all]",
"fastapi",
"uvicorn",
"gunicorn",
"python-daemon",
"git+https://github.com/openai/CLIP.git"]

# Write the normal packages to a temporary requirements file
with open("requirements.txt", "w") as f:
    f.write("\n".join(normal_packages))

# Install the remaining packages from the temporary requirements file
subprocess.run(["pip", "install", "-r", "requirements.txt"])

# List of packages to install with no dependencies
no_deps_packages = ["fastt5==0.1.4"]

# Install packages with no dependencies
for pkg in no_deps_packages:
    subprocess.run(["pip", "install", "--no-deps", pkg])
