import ipaddress
from datetime import datetime, timedelta

from cryptography import x509
from cryptography.x509 import IPAddress
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption

def generate_cert(network_ip: str):
    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )

    # Generate a self-signed certificate
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "DSN"),
        x509.NameAttribute(NameOID.COMMON_NAME, "DSN"),
    ])

    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.utcnow())
        .not_valid_after(datetime.utcnow() + timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(network_ip),
                IPAddress(ipaddress.IPv4Address(network_ip))
            ]),
            critical=False
        )
        .sign(private_key, hashes.SHA256(), default_backend())
    )

    cert_bytes = certificate.public_bytes(Encoding.PEM)
    private_key_bytes = private_key.private_bytes(
        encoding=Encoding.PEM,
        format=PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=NoEncryption()
    )

    return cert_bytes, private_key_bytes