import subprocess
import sys

def main():
    """
    Provides a simple command-line menu to run the main application
    or the test script.
    """
    # Check if the required scripts exist
    required_files = ["google_recorder.py", "testgoogle.py"]
    for filename in required_files:
        try:
            with open(filename, 'r') as f:
                pass
        except FileNotFoundError:
            print(f"Error: The required file '{filename}' was not found.")
            print("Please ensure you have created both 'google_recorder.py' and 'testgoogle.py'.")
            sys.exit(1)

    while True:
        print("\n--- Main Menu ---")
        print("1. Run the Main Recorder Application (google_recorder.py)")
        print("2. Run the Google Cloud API Test Script (testgoogle.py)")
        print("3. Exit")
        
        choice = input("Please enter your choice (1-3): ")
        
        if choice == '1':
            print("\nStarting the main recorder application...")
            try:
                # Using sys.executable to ensure the same python interpreter is used
                subprocess.run([sys.executable, "google_recorder.py"], check=True)
            except subprocess.CalledProcessError as e:
                print(f"The application exited with an error: {e}")
            except KeyboardInterrupt:
                print("\nApplication stopped by user.")
            
        elif choice == '2':
            print("\nRunning the Google Cloud API test script...")
            try:
                subprocess.run([sys.executable, "testgoogle.py"], check=True)
            except subprocess.CalledProcessError as e:
                print(f"The test script exited with an error: {e}")

        elif choice == '3':
            print("Exiting.")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()