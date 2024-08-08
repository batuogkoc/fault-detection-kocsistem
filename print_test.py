import time
from py_utils import *

if __name__ == "__main__":
    logger = RunningAverageLogger()
    printer = InplacePrinter(2)
    for epoch in range(10):
        logger.reset()
        printer.reset()
        print(f"EPOCH {epoch}")
        # print(end="")
        for i in range(20):
            logger.add_value(i/3)
            # print('\r\033[F', end="", flush=True)
            # print(f"Progress: {i}", end="\033[K\n\r", flush=True)
            # print(f"test {logger.get_avg()}", end="\033[K", flush=True)
            printer.print(f"Progress: {i}")
            printer.print(f"test {logger.get_avg()}")
            time.sleep(.1)
