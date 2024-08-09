"""Utilities for logging."""
# Standard library
import io
import logging
import logging.config
from pathlib import Path
import sys
from traceback import TracebackException
from typing import Literal, Optional

# Globals
log = logging.getLogger(__name__)

################################################################################
# Configuration

# Format options
# LOG_FMT_DEFAULT ="%(levelname)8s | %(message)s"
# LOG_FMT_DEFAULT ="%(levelname)8s | %(module)s :: %(message)s"
LOG_FMT_DEFAULT ="%(levelname)8s | %(name)s :: %(message)s"
# LOG_FMT_DEFAULT = "%(levelname)8s | %(name_last)s :: %(message)s"
# LOG_FMT_DEFAULT ="[%(asctime)s] %(levelname)8s | (%(filename)s) %(message)s"
# LOG_FMT_DEFAULT = "%(levelname)8s | (%(module)s:%(funcName)s():L%(lineno)d) %(message)s"

LOG_LEVEL_CHOICES = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")
LogLevelStr = Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"]
LogLevel = int | LogLevelStr


def configure_logging(
    file_config: Optional[Path] = None,
    dict_config: Optional[dict] = None,
    **basic_config,
) -> None:
    """Configure logging with any of the available logging config APIs."""
    # Update log files to be within output dir
    if len(basic_config) > 0:
        if 'format' not in basic_config:
            basic_config['format'] = LOG_FMT_DEFAULT
        logging.basicConfig(**basic_config)

    if file_config is not None:
        logging.config.fileConfig(file_config)

    if dict_config is not None:
        if "version" not in dict_config:
            dict_config["version"] = 1  # Only valid value according to docs
        logging.config.dictConfig(dict_config)

    n_args = sum(map(bool, (file_config, dict_config, basic_config)))
    if n_args > 1:
        log.info(
            "%d logging configurations provided. "
            "Order called: basicConfig -> fileConfig -> dictConfig",
            n_args,
        )

    redirect_exceptions_to_logger()
    # Use at your own risk. See function docstring for warnings
    # capture_python_stdout()

    # Disable NOTSET loggers from 3rd party packages
    logging.getLogger('matplotlib').setLevel('WARNING')
    logging.getLogger('asyncio').setLevel('WARNING') # for ipython


def require_root_console_handler(level: Optional[LogLevel] = None) -> None:
    """Require the root logger have a console handler when using basicConfig

    logging.basicConfig allows configuration of a console handler or file
    handler but not both without manually providing them to the 'handlers'
    argument. Call this after configuring logging with basicConfig to add a
    console handler in when the basicConfig only added a FileHandler or
    anything that is not a StreamHandler.
    """
    handlers = logging.root.handlers
    if any(type(h) is logging.StreamHandler for h in handlers):
        return
    sh = logging.StreamHandler()
    if level is not None:
        sh.setLevel(level)
    formatter = next((h.formatter for h in handlers if h.formatter), None)
    if formatter is not None:
        sh.setFormatter(formatter)
    handlers.append(sh)


def update_log_filenames(
    output_dir: Path,
    logging_cfg: dict,
) -> None:
    """Update log filenames to be relative to provided output directory"""
    if logging_cfg.get("filename"):
        logging_cfg["filename"] = str(output_dir / logging_cfg["filename"])

    if logging_cfg.get("dict_config", {}).get("handlers"):
        for hcfg in logging_cfg["dict_config"]["handlers"].values():
            if hcfg.get("filename") is None:
                # Not a FileHandler
                continue
            filename = Path(hcfg["filename"])
            # Only update filenames or paths relative to output_dir
            if len(filename.parts) > 1 and filename.parent.is_dir():
                continue
            hcfg["filename"] = str(output_dir / hcfg["filename"])


################################################################################
def level_name(level: int) -> str:
    """Convert level number into a name."""
    name = ""
    if level >= 50:
        name = "CRITICAL"
    elif level >= 40:
        name = "ERROR"
    elif level >= 30:
        name = "WARNING"
    elif level >= 20:
        name = "INFO"
    elif level >= 10:
        name = "DEBUG"
    elif level == 0:
        name = "NOTSET"
    sublevel = level % 10
    if sublevel:
        name += f"+{sublevel}"
    return name


def all_logger_names() -> list[str]:
    """Get the names of all loggers."""
    # pylint: disable=no-member
    return ["root"] + sorted(logging.root.manager.loggerDict.keys())


def logging_hierarchy_str():
    """Summarize the hierarchy of loggers."""
    ostr = ""
    for name in all_logger_names():
        if name == "root":
            name_last = name
            depth = 0
        else:
            parts = name.split(".")
            name_last = parts[-1]
            depth = len(parts)
        tabs = " " * 4 * depth
        logger = logging.getLogger(name)
        lvl_name = level_name(logger.level)
        n_handlers = len(logger.handlers)
        propagate = logger.propagate
        if (lvl_name, n_handlers, propagate) == ("NOTSET", 0, True):
            attr_str = ""
        else:
            attr_str = f" [{lvl_name}; {n_handlers} handler(s); {propagate = }]"
        prefix = "" if name == "root" else f"{tabs}- "
        ostr += f"{prefix}{name_last!r}{attr_str}\n"
    return ostr


def log_summary_str(logger):
    """Summarize logger."""
    log_lvl = logger.level
    eff_lvl = logger.getEffectiveLevel()
    min_lvl = next(
        (lvl for lvl in range(logging.CRITICAL + 1) if logger.isEnabledFor(lvl)), None
    )

    s = f"Log Summary - {logger.name}"
    s += "\n - Levels   : "
    s += f"Effective = {level_name(eff_lvl)}; "
    s += f"Logger = {level_name(log_lvl)}; "
    s += f"Enabled for >= {level_name(min_lvl) if min_lvl is not None else '?'}"
    s += f"\n - Flags    : Disabled = {logger.disabled}"
    s += f", Propagate = {logger.propagate}"
    s += f", Handlers = {logger.hasHandlers()}"
    # if logger.parent:
    #    s += f"\n - Parent : {logger.parent.name}"
    for i, hndl in enumerate(logger.handlers, 1):
        s += f"\n - Handler {i}: {hndl}"
    for i, fltr in enumerate(logger.filters, 1):
        s += f"\n - Filter {i} : {fltr}"
    return s


def log_multiline(log_call, txt):
    """Log a multiline message in multiple log messages."""
    for line in txt.splitlines():
        log_call(line)


def summarize_logging() -> str:
    """Summarize the root logger and all it's child loggers."""
    return logging_hierarchy_str() + "\n" + log_summary_str(logging.root)


class RecordAttributeAdder(logging.Filter):  # pylint: disable=too-few-public-methods
    """Pseudo-Filter that adds useful attributes to log records for
    formatting."""

    def filter(self, record: logging.LogRecord):
        """Strip off parent logger names."""
        record.name_last = record.name.rsplit(".", 1)[-1]
        return True


def redirect_exceptions_to_logger(logger: logging.Logger = logging.root):
    """Overwrite hook for processing exceptions."""

    # https://stackoverflow.com/questions/6234405/logging-uncaught-exceptions-in-python
    # https://stackoverflow.com/questions/8050775/using-pythons-logging-module-to-log-all-exceptions-and-errors
    def handle_exception(typ, val, tb):
        if issubclass(typ, KeyboardInterrupt):
            # Don't capture keyboard interrupt
            sys.__excepthook__(typ, val, tb)
            return
        nonlocal logger

        # Option 1 - trace in one log error message
        # logger.exception("Uncaught exception", exc_info=(typ, val, tb))

        # Option 2 - trace split into one log error message per newline
        logger.error("Uncaught exception")
        for lines in TracebackException(typ, val, tb).format():
            for line in lines.splitlines():
                logger.error(line)

    sys.excepthook = handle_exception


def capture_python_stdout(logger: logging.Logger = logging.root):
    """Capture all stdout/stderr and send to logger.

    NOTES/WARNINGS
    - This will capture messages from all non-child loggers, usually duplicating
      a lot of formatting (e.g. level, module, etc.)
    - This will capture output for the python debugger pdb
    - This will not capture messages sent directly to terminal stdout/stderr
      instead of via the python streams (see capture_unix_df).
    """
    name_prefix = f"{logger.name}." if logger.name != "root" else ""
    stdout_log = logging.getLogger(f"{name_prefix}stdout")
    stderr_log = logging.getLogger(f"{name_prefix}stderr")

    # Overwrite python stdout and stderr streams
    # https://stackoverflow.com/questions/19425736/how-to-redirect-stdout-and-stderr-to-logger-in-python
    sys.stdout = LoggerWriter(stdout_log.critical)
    sys.stderr = LoggerWriter(stderr_log.critical)


class LoggerWriter(io.TextIOWrapper):
    """Wrapper of stdout or stderr to redirect messages to logger."""

    def __init__(self, writer):
        """Initialize self."""
        # self.encoding = sys.stdout.encoding # Getting issues with doctest
        self._writer = writer
        self._msg = ""

    def write(self, message: str):
        """Write message to IO buffer."""
        for line in message.rstrip().splitlines():
            self._writer(line.rstrip())
        # Prevent carriage return and empty newlines
        # msg = message.lstrip("\r").lstrip("\n")

        # self._msg = self._msg + msg
        # while "\n" in self._msg:
        #    pos = self._msg.find("\n")
        #    self._writer(self._msg[:pos]+"\n")
        #    self._msg = self._msg[pos+1:]

    def flush(self):
        """Flush messages to writer."""
        # if self._msg != "":
        #     self._writer(self._msg)
        #     self._msg = ""


# import os
# import subprocess
# def capture_unix_fd():
#     # TODO: Currently doesn't work
#     # Also risk of infinite pipe loop as python stdout gets redirected back
#     # to logger that prints it to stdout
#     # https://stackoverflow.com/questions/616645/how-to-duplicate-sys-stdout-to-a-log-file
#
#     tee = subprocess.Popen(["tee", "log.txt"], stdin=subprocess.PIPE)
#     # Cause tee's stdin to get a copy of our stdin/stdout (as well as that
#     # of any child processes we spawn)
#     os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
#     os.dup2(tee.stdin.fileno(), sys.stderr.fileno())
#
#     # The flush flag is needed to guarantee these lines are written before
#     # the two spawned /bin/ls processes emit any output
#     print("\nstdout", flush=True)
#     print("stderr", file=sys.stderr, flush=True)
#
#     # These child processes' stdin/stdout are
#     os.spawnve("P_WAIT", "/bin/ls", ["/bin/ls"], {})
#     os.execve("/bin/ls", ["/bin/ls"], os.environ)
