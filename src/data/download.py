"""Download and extract the SpamAssassin public corpus into data/raw/spamassassin/."""

import logging
import tarfile
import urllib.request
from pathlib import Path

from src.utils.config import load_config

BASE_URL = "https://spamassassin.apache.org/old/publiccorpus/"
ARCHIVES = [
    "20030228_easy_ham.tar.bz2",
    "20030228_easy_ham_2.tar.bz2",
    "20030228_hard_ham.tar.bz2",
    "20030228_spam.tar.bz2",
    "20050311_spam_2.tar.bz2",
]

logger = logging.getLogger(__name__)


def download_spamassassin(target_dir: Path) -> None:
    archive_dir = target_dir / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)

    for name in ARCHIVES:
        archive_path = archive_dir / name
        if not archive_path.exists():
            logger.info("downloading %s", name)
            urllib.request.urlretrieve(BASE_URL + name, archive_path)

        with tarfile.open(archive_path, "r:bz2") as tar:
            # Each archive holds a single top-level folder (easy_ham, spam_2, ...).
            folder = target_dir / tar.getnames()[0].split("/")[0]
            if folder.exists():
                continue
            logger.info("extracting %s", name)
            tar.extractall(target_dir, filter="data")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()
    download_spamassassin(config["paths"]["raw"] / "spamassassin")
