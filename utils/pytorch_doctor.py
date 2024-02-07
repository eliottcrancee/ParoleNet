import torch
import torchaudio
import torchvision
import psutil
from colorama import Fore, Style


def print_versions():
    print("PyTorch version:", Fore.BLUE + torch.__version__ + Style.RESET_ALL)
    print("torchaudio version:", Fore.BLUE + torchaudio.__version__ + Style.RESET_ALL)
    print("torchvision version:", Fore.BLUE + torchvision.__version__ + Style.RESET_ALL)


def print_memory_usage():
    ram_total = psutil.virtual_memory().total >> 30  # RAM totale en Go
    ram_available = psutil.virtual_memory().available >> 30  # RAM disponible en Go
    if torch.cuda.is_available():
        vram = (
            torch.cuda.get_device_properties(0).total_memory >> 30
        )  # VRAM de la première GPU en Go
        print("VRAM totale sur GPU:", Fore.GREEN + str(vram) + " Go" + Style.RESET_ALL)
    else:
        vram = "Non applicable"
    print("RAM totale:", Fore.GREEN + str(ram_total) + " Go" + Style.RESET_ALL)
    print("RAM disponible:", Fore.GREEN + str(ram_available) + " Go" + Style.RESET_ALL)


def print_device_info():
    if torch.cuda.is_available():
        print("GPU disponible")
        print(torch.cuda.get_device_name(0))
    else:
        print("Aucun GPU disponible")


if __name__ == "__main__":
    print_versions()
    print_memory_usage()
    print_device_info()
