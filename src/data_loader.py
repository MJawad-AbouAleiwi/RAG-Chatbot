# Loading the dataset
from src.config import DATA_PATH

def load_dataset(path: str = DATA_PATH) -> list[str]:
    # Read the knowledge base text file and return a list of lines.
    with open(path, "r", encoding="utf-8") as file:
        dataset = file.readlines()
    print(f"The loaded dataset contains {len(dataset)} entries.")
    return dataset