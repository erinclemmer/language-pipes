import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from language_pipes.distributed_state_network.util.aes import (
    AES_BLOCK_SIZE,
    AES_KEY_LENGTH,
    aes_decrypt,
    aes_encrypt,
    generate_aes_key,
)


class TestAES(unittest.TestCase):
    def test_encrypt_decrypt_roundtrip(self):
        """Ciphertext should decrypt back to the original plaintext."""
        key = generate_aes_key()
        payload = b"\x05example payload for distributed state network"

        ciphertext = aes_encrypt(key, payload)
        plaintext = aes_decrypt(key, ciphertext)

        self.assertEqual(plaintext, payload)

    def test_same_plaintext_encrypts_differently(self):
        """Per-message IV should produce different ciphertext for same plaintext."""
        key = generate_aes_key()
        payload = b"\x05same payload"

        c0 = aes_encrypt(key, payload)
        c1 = aes_encrypt(key, payload)

        self.assertNotEqual(c0, c1)
        self.assertNotEqual(c0[:AES_BLOCK_SIZE], c1[:AES_BLOCK_SIZE])

    def test_reject_legacy_iv_prefixed_key_material(self):
        """Legacy [IV|KEY] material should be rejected."""
        key = generate_aes_key()
        legacy_key = os.urandom(AES_BLOCK_SIZE) + key

        with self.assertRaises(ValueError):
            aes_encrypt(legacy_key, b"payload")

        with self.assertRaises(ValueError):
            aes_decrypt(legacy_key, b"\x00" * (AES_BLOCK_SIZE * 2))

    def test_generate_key_length(self):
        """Generated key should be raw AES-128 key length only."""
        key = generate_aes_key()
        self.assertEqual(len(key), AES_KEY_LENGTH)


if __name__ == "__main__":
    unittest.main()
