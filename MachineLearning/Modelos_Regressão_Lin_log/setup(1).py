import subprocess
import sys

def run_cmd(cmd):
    print(f"\n>> Executando: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Erro ao executar: {cmd}")
        sys.exit(1)

def setup_tda_env():

    run_cmd("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    run_cmd("pip install scikit-learn")
    run_cmd("pip install pandas")
    run_cmd("pip install numpy")
    run_cmd("pip install ipykernel")

    print("\n>> Ambiente configurado com sucesso!")
if __name__ == "__main__":
    setup_tda_env()
