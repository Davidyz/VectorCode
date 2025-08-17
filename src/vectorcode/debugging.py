import atexit
import cProfile
import logging
import os
import pstats
from datetime import datetime

__LOG_DIR = os.path.expanduser("~/.local/share/vectorcode/logs/")

logger = logging.getLogger(name=__name__)

__profiler: cProfile.Profile | None = None


def _ensure_log_dir():
    """Ensure the log directory exists"""
    os.makedirs(__LOG_DIR, exist_ok=True)


def finish():
    """Clean up profiling and save results"""
    if __profiler is not None:
        try:
            __profiler.disable()
            stats_file = os.path.join(
                __LOG_DIR,
                f"cprofile-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.stats",
            )
            __profiler.dump_stats(stats_file)
            print(f"cProfile stats saved to: {stats_file}")

            # Print summary stats
            stats = pstats.Stats(__profiler)
            stats.sort_stats("cumulative")
            stats.print_stats(20)
        except Exception as e:
            logger.warning(f"Failed to save cProfile output: {e}")


def enable():
    """Enable cProfile-based profiling and crash debugging"""
    global __profiler

    try:
        _ensure_log_dir()

        # Initialize cProfile for comprehensive profiling
        __profiler = cProfile.Profile()
        __profiler.enable()
        atexit.register(finish)
        logger.info("cProfile profiling enabled successfully")

        try:
            import coredumpy  # noqa: F401

            logger.info("coredumpy crash debugging enabled successfully")
            coredumpy.patch_except(directory=__LOG_DIR)
        except Exception as e:
            logger.warning(
                f"Crash debugging will not be available. Failed to import coredumpy: {e}"
            )

    except Exception as e:
        logger.error(f"Failed to initialize cProfile: {e}")
        logger.warning("Profiling will not be available for this session")
