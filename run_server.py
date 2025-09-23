#!/usr/bin/env python3
"""
Startup script for the RealtimeVoiceChat server.
This script properly sets up the Python path and runs the server.
"""

if __name__ == "__main__":
    import sys
    import os

    # Add the current directory to Python path so src can be imported as a package
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    # Now import and run the server
    try:
        from src.server import app
        import uvicorn

        print("🚀 Starting RealtimeVoiceChat server...")
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

    except KeyboardInterrupt:
        print("\n🛑 Server shutdown requested by user")
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
