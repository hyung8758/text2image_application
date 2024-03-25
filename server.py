"""
daemon server.

Hyungwon Yang
24.03.19
"""

import os, sys
import time
from src.handlers.ServerHandler import start_server, stop_server, console
from src.utils.LogUtils import LogUtils

# main daemon server.
def main():
    if len(sys.argv) != 2:
        print("Usage: {} [start|stop|restart|console]".format(sys.argv[0]))
        sys.exit(2)
    else:
        if sys.argv[1] == 'console':
            console()
        else:
            log_path = os.path.join(os.getcwd(), "log")
            LogUtils.init_log(log_path)
            if sys.argv[1] == 'start':
                start_server()
            elif sys.argv[1] == 'stop':
                stop_server()
            elif sys.argv[1] == 'restart':
                stop_server()
                time.sleep(2)
                start_server()
            else:
                print("Unknown command")
                sys.exit(2)

if __name__ == "__main__":
    main()