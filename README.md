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
        $
        ```
- API server


### List of Models
- ASR
    - faster_whipser
- text-to-image
    - Karlo (kakaobrain)
    - koala (Etri)
