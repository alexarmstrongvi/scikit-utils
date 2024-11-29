# Standard library
import logging
from pathlib import Path
import re
import subprocess
from typing import Optional

# Globals
log = logging.getLogger(__name__)


################################################################################
def find_working_dir(path: Path) -> Optional[Path]:
    root = Path(Path.home().root)
    while not (path / ".git").is_dir() and path != root:
        path = path.parent
    return path if path != root else None

################################################################################
def get_status(path: Path) -> str:
    return subprocess.check_output(
        args=["git", "status"],
        text=True,
        cwd=path,
    )


def get_hash(path: Path) -> str:
    return subprocess.check_output(
        args=["git", "rev-parse", "HEAD"],
        # args = ['git','log', '-n', '1', '--format', '%H'],
        text=True,
        cwd=path,
    ).strip()


def get_diff(path: Path) -> str:
    return subprocess.check_output(
        args=["git", "diff", "HEAD"],
        text=True,
        cwd=path,
    )

def summarize_version_control(path: Path) -> str:
    """Summarize the version control state of this package."""
    git_hash = get_hash(path)
    git_status = get_status(path)
    # Remove hints on how to use git from status
    lines = []
    for line in git_status.splitlines():
        line = re.sub(r'\(use "git.*\)', "", line)
        if line.strip():
            lines.append(line.replace("\t", " " * 4))
    git_status = "\n".join(lines)
    summary = f"Git Hash: {git_hash}" "\n" f"Git Status:\n\n{git_status}\n"

    ########################################
    return summary
