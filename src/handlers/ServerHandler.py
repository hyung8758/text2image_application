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
from fastapi import FastAPI, Request

cur_dir = os.getcwd()
pid_path = os.path.join(cur_dir,".pid")
pid_file_path = os.path.join(pid_path, "server.pid")
server_conf_path = os.path.join(cur_dir, "conf", "server.yaml")

app = FastAPI()

@app.post("/text2image")
async def post(request: Request):
    logging.info("start post request.")
    json_data = await request.json()
    logging.info(json_data)
    response = {"message": "Hello. This is text2image server"}
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