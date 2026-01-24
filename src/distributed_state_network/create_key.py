import sys
from util.aes import generate_aes_key

print("Key: " + generate_aes_key().hex())