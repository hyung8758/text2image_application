# text2image_application
Integration of text-to-image, speech recognition, and translation models.


### Information
- This project does not encompass the implementation of pre- and post-processing components. Users intending to utilize this system will need to develop these components independently.
- This project is primarily designed for Korean-speaking users to describe imagined images in Korean, generating corresponding visual outputs.

### Plan
- [input] (eng)text -> [output] image
- [input] (eng)audio -> (eng)text -> [output] image
- [input] (kor)audio -> (eng)translated text -> [output] image
    - Use either the Whisper Speech Translator or the independent Translator module.

### Usage 
- install dependencies
    ```bash
    pip install -r requirements.txt
    ```
- gradio demo
    - ./gradioApp/text2imageCpuClient.py: 
        - [input] text -> [output] image
        - This is a CPU-based version of the Triton Server for text-to-image processing, which may result in slower performance.
        ```bash
        # Ensure that your current directory is 'text2image_application/' and execute the following commands.

        # 1. Start the Triton Server using Docker. If you want to keep the current container, omit the '--rm' option; otherwise, the container will be automatically removed when you exit.
        docker run -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ -v ${PWD}/models/cpu/text2image_model:/models nvcr.io/nvidia/tritonserver:24.01-py3 bash

        # 2. Once the container is started, install the following libraries
        pip install torch torchvision torchaudio
        pip install transformers ftfy scipy accelerate
        pip install diffusers==0.9.0
        pip install transformers[onnxruntime]

        # 3. Activate the Triton Server.
        tritonserver --model-repository=/models

        # 4. After activating Triton Server, leave the terminal window open and open a new terminal window to start the client server.
        cd gradioApp
        python3 text2imageCpuClient.py
        ```
    - ./gradioApp/text2imageGpuClient.py
        - [input] text -> [output] image
        - The GPU-based version of the Triton Server for text-to-image porcessing.
        ```bash
        # Ensure that your current directory is 'text2image_application/' and execute the following commands.

        # 1. Export and convert the models.
        docker run -it --gpus all --rm -p 8888:8888 -v ${PWD}:/mount nvcr.io/nvidia/pytorch:24.01-py3

        pip install transformers ftfy scipy accelerate
        pip install transformers[onnxruntime]
        pip install diffusers==0.9.0
        huggingface-cli login # login with the huggingface token.
        cd /mount
        python export.py

        # Accelerating VAE with TensorRT
        trtexec --onnx=vae.onnx --saveEngine=vae.plan --minShapes=latent_sample:1x4x64x64 --optShapes=latent_sample:4x4x64x64 --maxShapes=latent_sample:8x4x64x64 --fp16

        # Place the models in the model repository
        mkdir models/cuda/text2image_model/vae/1
        mkdir models/cuda/text2image_model/text_encoder/1
        mv vae.plan models/cuda/text2image_model/vae/1/model.plan
        mv encoder.onnx models/cuda/text2image_model/text_encoder/1/model.onnx
        exit # exit the container and it will be automatically deleted.

        # 2. Start the Triton Server using Docker. If you want to keep the current container, omit the '--rm' option; otherwise, the container will be automatically removed when you exit.
        docker run --gpus all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ -v ${PWD}/models/cuda/text2image_model:/models nvcr.io/nvidia/tritonserver:24.01-py3 bash

        # 3. Once the container is started, install the following libraries
        pip install torch torchvision torchaudio
        pip install transformers ftfy scipy accelerate
        pip install diffusers==0.9.0
        pip install transformers[onnxruntime]

        # 4. Activate the Triton Server.
        # Return to the previously activated Triton server.
        tritonserver --model-repository=/models

        # 5. After activating Triton Server, leave the terminal window open and open a new terminal window to start the client server.
        cd gradioApp
        python3 text2imageGpuClient.py
        ```
- API server


### List of Models
- ASR
    - faster_whipser
- text-to-image
    - Karlo (kakaobrain)
    - koala (Etri)
