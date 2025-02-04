try:
    from fsspec.callbacks import _DEFAULT_CALLBACK, NoOpCallback, TqdmCallback
    print("fsspec.callbacks module imported successfully")
except ImportError as e:
    print(f"Import error: {e}")