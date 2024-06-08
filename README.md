# text2image_application
- Integration of text-to-image, speech recognition, and translation models.
- Mostly referenced from [triton server tutorial](https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_6-building_complex_pipelines).


## Information
- This project does not encompass the implementation of pre- and post-processing components. Users intending to utilize this system will need to develop these components independently.
- This project is primarily designed for Korean-speaking users to describe imagined images in Korean, generating corresponding visual outputs.

## Plan
- [input] (eng)text -> [output] image
- [input] (eng)audio -> (eng)text -> [output] image
- [input] (kor)audio -> (eng)translated text -> [output] image
    - Use either the Whisper Speech Translator or the independent Translator module.

## Usage 
- Install dependencies
    ```bash
    # [recommanded] Establish your own virtual environment.
    conda create -n triton_server python=3.10
    conda activate triton_server
    # 1. Install the required dependencies.
    pip install -r requirements.txt 
    ```
- CPU version: /gradioApp/text2imageCpuClient.py
    - [input] text -> [output] image
    - The CPU-based version of the Triton Server for text-to-image porcessing.
    ```bash
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
- GPU version: ./gradioApp/text2imageGpuClient.py
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
    python src/exports/export_nvidia_example.py

    # Accelerating VAE with TensorRT
    trtexec --onnx=vae.onnx --saveEngine=vae.plan --minShapes=latent_sample:1x4x64x64 --optShapes=latent_sample:4x4x64x64 --maxShapes=latent_sample:8x4x64x64 --fp16
    rm vae.onnx

    # Place the models in the model repository
    mkdir models/cuda/text2image_model/vae/1
    mkdir models/cuda/text2image_model/text_encoder/1
    mv vae.plan models/cuda/text2image_model/vae/1/model.plan
    mv encoder.onnx models/cuda/text2image_model/text_encoder/1/model.onnx
    exit # exit the container and it will be automatically deleted.

    # 2. Start the Triton Server using Docker. If you want to keep the current container, omit the '--rm' option; otherwise, the container will be automatically removed when you exit.
    docker run --gpus all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ -v ${PWD}/models/cuda/text2image_karlo_model_onnx2:/models nvcr.io/nvidia/tritonserver:24.01-py3 bash
    # if you would like to specify gpu devices then allocate the device numbers in --gpus options. (in case of 4 gpus, use only 1,2,3 except 0)
    # docker run --gpus '"device=1,2,3"' -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ -v ${PWD}/models/cuda/text2image_karlo_model_onnx2:/models nvcr.io/nvidia/tritonserver:24.01-py3 bash

    # 3. Once the container is started, install the following libraries
    pip install torch torchvision torchaudio
    pip install transformers==4.41.1
    pip install ftfy scipy accelerate progress sentencepiece
    pip install diffusers==0.27.2
    pip install onnxruntime==1.14.1
    pip install fastt5==0.1.4 --no-deps
    pip install transformers[onnxruntime]

    # 4. Activate the Triton Server.
    # Return to the previously activated Triton server.
    tritonserver --model-repository=/models

    # 5. After activating Triton Server, leave the terminal window open and open a new terminal window to start the client server.
    cd gradioApp
    python3 text2imageGpuClient.py
    ```
- API server
    ```bash
    # Run a UVicorn server with Gunicorn to handle multiple connections.
    num_jb=2
    gunicorn -w $num_jb -k uvicorn.workers.uvicornWorker server:app -b 0.0.0.0:33010

    # Test the server by using a client script
    python client.py
    # Within the client.py script, customize the input_text, save_image, and save_name variables to suit your requirements:
    """
    json_request = dict(
        input_text='초록색의 개구리 한 마리가 나뭇잎 위에 앉았다.',
        save_image='true',
        save_name='green_frog1.png'
    )
    """
    ```

## List of Models
- ASR
    - faster_whipser
- text-to-image
    - Karlo (kakaobrain)
    - koala (Etri)
- translator
    - ke-t5
