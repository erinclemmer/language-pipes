import subprocess

def upgrade_lp():
    print("Upgrading Language Pipes packages")
    subprocess.run(["pip", "install", "language-pipes", "--upgrade"])
    subprocess.run(["pip", "install", "distributed-state-network", "--upgrade"])
    subprocess.run(["pip", "install", "llm-layer-collector", "--upgrade"])