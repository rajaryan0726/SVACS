import subprocess
import sys
import time
import os

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(base_dir, "backend")
    frontend_dir = os.path.join(base_dir, "frontend")

    print("Starting SVACS v2.0...")

    # Start Backend
    print("-> Starting FastAPI Backend on port 8000...")
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
        cwd=backend_dir
    )

    # Give backend a moment to initialize
    time.sleep(2)

    # Start Frontend
    print("-> Starting React/Vite Frontend on port 5173...")
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=frontend_dir,
        shell=True
    )

    print("\n" + "="*50)
    print("SVACS is now running!")
    print("   Frontend Dashboard: http://localhost:5173")
    print("   Backend API Docs:   http://localhost:8000/docs")
    print("="*50 + "\n")
    print("Press Ctrl+C to stop both servers.")

    try:
        # Keep the main thread alive, waiting for user interruption
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down SVACS servers...")
        backend_process.terminate()
        frontend_process.terminate()
        
        backend_process.wait()
        frontend_process.wait()
        print("Goodbye!")

if __name__ == "__main__":
    main()
