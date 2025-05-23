import subprocess
import os

def test_train_script():
    # Run the training script
    result = subprocess.run(['python', 'train.py'], capture_output=True, text=True)
    print("Training script output:")
    print(result.stdout)
    if result.returncode != 0:
        print("Training script failed with error:")
        print(result.stderr)
        return False

    # Check if model and pickle files are created
    files_to_check = ['model.h5', 'tokenizer.pkl', 'sentiment_encoder.pkl', 'platform_encoder.pkl']
    missing_files = [f for f in files_to_check if not os.path.exists(f)]
    if missing_files:
        print(f"Missing files after training: {missing_files}")
        return False

    print("All expected files are present after training.")
    return True

if __name__ == "__main__":
    success = test_train_script()
    if success:
        print("Train.py test passed successfully.")
    else:
        print("Train.py test failed.")
