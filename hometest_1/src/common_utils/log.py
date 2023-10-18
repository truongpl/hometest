##########################################
#
# [TPL] Logging, copy elsewhere I don't remember
#
###########################################

import os, sys, time
import logging
import logging.handlers

class PackagePathFilter(logging.Filter):
    def filter(self, record):
        pathname = record.pathname
        record.relativepath = None
        abs_sys_paths = map(os.path.abspath, sys.path)
        for path in sorted(abs_sys_paths, key=len, reverse=True):  # longer paths first
            if not path.endswith(os.sep):
                path += os.sep
            if pathname.startswith(path):
                record.relativepath = os.path.relpath(pathname, path)
                break
        return True

# Logging
def get_logger(service):
    logger = logging.getLogger(service)
    # default level is info
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
    # set timezone UTC for logger
    logging.Formatter.converter = time.gmtime

    # Set format
    """

    %(name)s            Name of the logger (logging channel)
    %(levelno)s         Numeric logging level for the message (DEBUG, INFO,
                        WARNING, ERROR, CRITICAL)
    %(levelname)s       Text logging level for the message ("DEBUG", "INFO",
                        "WARNING", "ERROR", "CRITICAL")
    %(pathname)s        Full pathname of the source file where the logging
                        call was issued (if available)
    %(filename)s        Filename portion of pathname
    %(module)s          Module (name portion of filename)
    %(lineno)d          Source line number where the logging call was issued
                        (if available)
    %(funcName)s        Function name
    %(created)f         Time when the LogRecord was created (time.time()
                        return value)
    %(asctime)s         Textual time when the LogRecord was created
    %(msecs)d           Millisecond portion of the creation time
    %(relativeCreated)d Time in milliseconds when the LogRecord was created,
                        relative to the time the logging module was loaded
                        (typically at application startup time)
    %(thread)d          Thread ID (if available)
    %(threadName)s      Thread name (if available)
    %(process)d         Process ID (if available)
    %(message)s         The result of record.getMessage(), computed just as
                        the record is emitted
    """
    formatter = logging.Formatter(
        fmt="[%(levelname)s][%(asctime)s%(msecs)d][%(relativepath)s:%(lineno)d][%(funcName)s] - [%(process)d][%(thread)d]: %(message)s",
        datefmt="%Y%m%d:%H%M%S.")
    formatter.converter = time.gmtime

    # Log to files rotating
    log_file_name = '/'.join([os.environ.get("LOG_PATH", "/tmp/"), service])
    file_handler = logging.handlers.TimedRotatingFileHandler(log_file_name,
                                        when='h',
                                        interval=48,
                                        backupCount=3,
                                        utc=True)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(PackagePathFilter())
    logger.addHandler(file_handler)

    # but also routing log to console
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(PackagePathFilter())
    logger.addHandler(console_handler)

    is_debug = os.environ.get("DEBUG", "")
    if is_debug:
        engine_logger = logging.getLogger('sqlalchemy.engine')
        engine_logger.addHandler(file_handler)
        engine_logger.addHandler(console_handler)
        engine_logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

    return logger

logger = get_logger(os.environ.get("SERVICE", "UNKNOWN"))

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception