import hashlib
import hmac
import logging
import os
import unicodedata

from dotenv import load_dotenv


class SnapshotPseudonymizer:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def load_key_from_env(self) -> bytes:
        self.logger.info("Loading pseudonymization key from default .env file")
        load_dotenv()  # Load from default .env file
        key = os.getenv("ENV_PSEUDONYMIZATION_KEY")
        if not key:
            msg = "Pseudonymization key not found in environment variables."
            self.logger.error(msg)
            raise ValueError(msg)
        self.logger.info("Pseudonymization key loaded successfully.")
        return key.encode("utf-8")

    def normalize_email(self, email: str) -> str:
        email = email.strip().lower()
        email = unicodedata.normalize("NFC", email)
        return email  # noqa: RET504

    def pseudonymize_email(self, email: str) -> str:
        key = self.load_key_from_env()
        normalized_email = self.normalize_email(email)
        hmac_obj = hmac.new(key, normalized_email.encode("utf-8"), hashlib.sha256)
        pseudonymized_email = hmac_obj.hexdigest()
        return pseudonymized_email  # noqa: RET504
