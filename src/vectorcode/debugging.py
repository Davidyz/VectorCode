import logging
import os
import cProfile
import pstats
from datetime import datetime

import atexit

__LOG_DIR = os.path.expanduser("~/.local/share/vectorcode/logs/")

logger = logging.getLogger(name=__name__)

__profiler: cProfile.Profile | None = None


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
            logger.info(f"cProfile stats saved to: {stats_file}")
            
            # Print summary stats
            stats = pstats.Stats(__profiler)
            stats.sort_stats('cumulative')
            stats.print_stats(20)
        except Exception as e:
            logger.warning(f"Failed to save cProfile output: {e}")


def enable():
    """Enable cProfile-based profiling"""
    global __profiler
    
    try:
        # Initialize cProfile for comprehensive profiling
        __profiler = cProfile.Profile()
        __profiler.enable()
        atexit.register(finish)
        logger.info("cProfile profiling enabled successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize cProfile: {e}")
        logger.warning("Profiling will not be available for this session")