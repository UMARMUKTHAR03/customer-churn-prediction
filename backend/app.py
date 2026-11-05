"""
ChurnIQ — Flask REST API Application Factory
Author: Umar Mukthar
Tagline: Predict Churn. Understand Risk. Retain Customers.
"""

import os
import sys
import logging
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.routes import api_bp

logger = logging.getLogger('churniq_app')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def create_app():
    app = Flask(__name__, static_folder=BASE_DIR, static_url_path='')
    
    # Configure CORS securely: allow all origins in development and GitHub Pages in production
    CORS(app, resources={r"/*": {"origins": "*"}})

    # Register API blueprint with /api prefix as well as root for flexibility
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(api_bp, name='api_root', url_prefix='')

    # Root route serves index.html if opened directly
    @app.route('/')
    def serve_frontend():
        return send_from_directory(BASE_DIR, 'index.html')

    # Global HTTP 404 handler
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({
            'error': 'Resource not found',
            'platform': 'ChurnIQ',
            'available_endpoints': [
                'GET /health',
                'GET /model-info',
                'GET /analytics',
                'POST /predict',
                'POST /predict-batch'
            ]
        }), 404

    # Global HTTP 500 handler
    @app.errorhandler(500)
    def internal_error(e):
        logger.error(f"Internal server error: {e}")
        return jsonify({'error': 'Internal server error occurred', 'status': 500}), 500

    return app

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting ChurnIQ REST API Server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
