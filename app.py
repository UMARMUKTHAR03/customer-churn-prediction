"""
TeleChurn AI - Server Entrypoint
Imports and launches the production Flask application.
"""

import os
import sys

# Ensure current working directory is on path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 TeleChurn AI Server booting at http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
