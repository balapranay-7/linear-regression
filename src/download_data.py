from pathlib import Path 
from sklearn.datasets import fetch_california_housing
fetch_california_housing(as_frame=True)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT/"data"/"raw"
def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True,exist_ok=True)
    dataset = fetch_california_housing(as_frame=True)
    output_path = RAW_DATA_DIR / "california_housing.csv"
    dataset.frame.to_csv(output_path,index=False)
    print(f"Dataset saved to:{output_path}")
    print(f"Rows:{dataset.frame.shape[0]}")
    print(f"Columns:{dataset.frame.shape[1]}")
    for column in dataset.frame.columns:
        print(f"-{column}")
if __name__ =="__main__":
    main()


