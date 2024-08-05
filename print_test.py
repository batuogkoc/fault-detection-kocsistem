import time
from train import RunningAverageLogger

if __name__ == "__main__":
    logger = RunningAverageLogger()
    for epoch in range(10):
        logger.reset()
        print(f"EPOCH {epoch}")
        print()
        print(end="")
        for i in range(20):
            logger.add_value(i)
            print('\033[F\x1b[1K\r', end="")
            print('\x1b[1K\r', end="")
            print(f"Progress: {i}")
            print(f"test {logger.get_avg()}", end="")
            time.sleep(.1)
        print()
