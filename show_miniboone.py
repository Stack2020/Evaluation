import numpy as np
def main() -> None:
    data = np.load("./data/miniboone.npy")
    print(f"Shape: {data.shape}")
    print(f"Dimensions: {data.ndim}")
    print(f"Size: {data.size}")
    print("First 5 rows:")
    print(data[:5])


if __name__ == "__main__":
    main()
