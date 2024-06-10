# text2image_application
- Integration of text-to-image and translation models.
- Mostly referenced from [triton server tutorial](https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_6-building_complex_pipelines).


## Information
- This project does not encompass the implementation of pre- and post-processing components. Users intending to utilize this system will need to develop these components independently.
- This project is primarily designed for Korean-speaking users to describe imagined images in Korean, generating corresponding visual outputs.

## Pipeline
- [input] (kor)text -> (eng)translated text -> [output] image

## Usage 
- Install dependencies
    ```bash
    # [recommanded] Establish your own virtual environment.
    conda create -n triton_server python=3.10
    conda activate triton_server
    # 1. Install the required dependencies.
    python install_packages.py
    ```
- GPU version
    - The GPU-based version of the Triton Server for text-to-image porcessing.
    ```bash
    # Ensure that your current directory is 'text2image_application/' and execute the following commands.

    # 1. Export and convert the models.
    python src/export_karlo.py # optional
    python src/export_kor2eng_translator.py

    # 2. Start the Triton Server using Docker. 
    # Adjust parameters to match your environment
    cd docker
    docker-compose build
    docker-compose up -d 
    # Check logs.
    docker-compose logs -f
    # Stop the Triton Server container when not in use
    # docker-compose down
    ```
- API server
    ```bash
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
- text-to-image
    - Karlo (kakaobrain)
- translator
    - ke-t5
