try:
    import omegaconf
    print("Omegaconf imported successfully")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
