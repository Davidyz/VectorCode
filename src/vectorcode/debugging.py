import logging
import os
from datetime import datetime

# import atexit

__LOG_DIR = os.path.expanduser("~/.local/share/vectorcode/logs/")

logger = logging.getLogger(name=__name__)

__tracer = None


def finish():
    if __tracer is not None:
        __tracer.stop()
        __tracer.save(
            output_file=os.path.join(
                __LOG_DIR,
                f"viztracer-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json",
            )
        )


def enable():
    global __tracer
    try:
        import coredumpy
        # import viztracer
    except ModuleNotFoundError:
        logger.warning("Failed to import modules. Please install vectorcode[debug]")
        return

    coredumpy.patch_except(directory=__LOG_DIR)
    # if __tracer is None:
    #     __tracer = viztracer.VizTracer(log_async=True)
    # __tracer.start()
    # atexit.register(finish)
