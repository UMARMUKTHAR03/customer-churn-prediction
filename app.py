"""
ChurnIQ — Production Server Entrypoint
Author: Umar Mukthar
Tagline: Predict Churn. Understand Risk. Retain Customers.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 ChurnIQ Server booting at http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
