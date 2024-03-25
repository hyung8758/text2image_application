"""
api server

Hyungwon Yang
24.03.19
"""
import os, sys
import yaml
import logging
import uvicorn
import daemon
import daemon.pidfile
import signal
from src.handlers.ImageHandler import ImageHandler
from src.handlers.TextHandler import TextHandler
from fastapi import FastAPI, Request

cur_dir = os.getcwd()
pid_path = os.path.join(cur_dir,".pid")
pid_file_path = os.path.join(pid_path, "server.pid")
server_conf_path = os.path.join(cur_dir, "conf", "server.yaml")

app = FastAPI()

@app.post("/text2image")
async def post(request: Request):
    logging.info("start post request.")
    # Retrieve the handlers using dependency injection
    image_handler = request.app.state.image_handler
    text_handler = request.app.state.text_handler
    json_data = await request.json()
    logging.info(json_data)
    input_text = json_data['input_text']
    save_image = json_data['save_image']
    save_name = json_data['save_name']
    logging.info("INPUT TEXT: {}".format(input_text))
    if text_handler:
        input_text = text_handler.run(input_value=input_text)
        logging.info("text_handler output: {}".format(input_text))
    if image_handler:
        output_image = image_handler.run(input_value=input_text, save_image=save_image, save_name=save_name)
    logging.info("image_handler output: {}".format(output_image))
    response = {"message": "successul", "output": "will be implemented"}
    logging.info("done post request.")
    return response

def start_server() -> None:
    # config
    if not os.path.exists(pid_path): # directory NOT exists.
        os.makedirs(pid_path, mode=0o755)
    with open(server_conf_path) as f:
        server_conf = yaml.load(f, Loader=yaml.FullLoader)
    server_host = server_conf['server_host']
    server_port = server_conf['server_port']
    if os.path.exists(pid_file_path):
        logging.error("server is already running {}:{}".format(server_host, server_port))
    else:
        # task handlers.
        imageHandler = ImageHandler()
        textHandler = TextHandler()
        imageHandler.initialize(model="karlo", cuda_device=1)
        textHandler.initialize(model="ke-t5", use_cuda=False)
        # Assign the handlers to the app's state
        app.state.image_handler = imageHandler
        app.state.text_handler = textHandler
        # kwargs = dict(
        #     image_handler = imageHandler,
        #     text_handler = textHandler
        # )
        with daemon.DaemonContext(
                pidfile=daemon.pidfile.PIDLockFile(os.path.join(pid_path, "server.pid")),
                stdout=sys.stdout,
                stderr=sys.stderr,
                files_preserve=[logging.root.handlers[0].stream]
            ):
            logging.info("start server: {}:{}".format(server_host, server_port))
            uvicorn.run(app, host=server_host, port=server_port)
    
def stop_server() -> None:
    logging.info("stop server.")
    if os.path.exists(pid_file_path):
        with open(pid_file_path, "r") as pid_file:
            pid = int(pid_file.read().strip())

        # Send the SIGTERM signal to the UVicorn process
        os.kill(pid, signal.SIGTERM)
    else:
        logging.info("server is not running.")
        
def console() -> None:
    # config
    print("start console mode.")
    if not os.path.exists(pid_path): # directory NOT exists.
        os.makedirs(pid_path, mode=0o755)
    with open(server_conf_path) as f:
        server_conf = yaml.load(f, Loader=yaml.FullLoader)
    server_host = server_conf['server_host']
    server_port = server_conf['server_port']
    if os.path.exists(pid_file_path):
        print("server is already running {}:{}".format(server_host, server_port))
    else:
        # task handlers.
        imageHandler = ImageHandler()
        textHandler = TextHandler()
        imageHandler.initialize(model="karlo", cuda_device=1)
        textHandler.initialize(model="ke-t5", use_cuda=False)
        # Assign the handlers to the app's state
        app.state.image_handler = imageHandler
        app.state.text_handler = textHandler
        print("start server: {}:{}".format(server_host, server_port))
        uvicorn.run(app, host=server_host, port=server_port)