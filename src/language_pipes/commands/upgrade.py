import subprocess

def upgrade_lp():
    print("Upgrading Language Pipes packages")
    subprocess.run(["pip", "install", "language-pipes", "--upgrade"])
