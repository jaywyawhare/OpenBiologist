#!/bin/bash

# OpenBiologist Startup Script
# Runs both the FastAPI backend and MCP server

set -e  # Exit on any error

echo "🧬 OpenBiologist Startup Script"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if required Python packages are installed
print_status "Checking Python dependencies..."
python3 -c "
import fastapi, uvicorn, torch, numpy, requests
print('✅ All required packages are available')
" 2>/dev/null || {
    print_error "Missing required Python packages. Please install them first:"
    echo "pip install -r requirements.txt"
    exit 1
}

# Create results directory if it doesn't exist
mkdir -p protein_folding_results

# Function to cleanup background processes
cleanup() {
    print_status "Shutting down services..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
        print_status "Backend service stopped"
    fi
    if [ ! -z "$MCP_PID" ]; then
        kill $MCP_PID 2>/dev/null || true
        print_status "MCP server stopped"
    fi
    exit 0
}

# Set up signal handlers for graceful shutdown
trap cleanup SIGINT SIGTERM

# Start the FastAPI backend
print_status "Starting FastAPI backend..."
cd "$(dirname "$0")"
python3 -c "
import sys
import os
from pathlib import Path

# Add src to Python path
current_dir = Path.cwd()
src_path = current_dir / 'src'
sys.path.insert(0, str(src_path))

from backend.main import app
import uvicorn
import threading
import time

def run_server():
    uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info')

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

# Wait a bit for server to start
time.sleep(3)
print('Backend server started on http://localhost:8000')
" &
BACKEND_PID=$!

# Wait for backend to be ready
print_status "Waiting for backend to be ready..."
for i in {1..30}; do
    if python3 -c "
import requests
try:
    response = requests.get('http://localhost:8000/health')
    if response.status_code == 200:
        print('Backend is ready!')
        exit(0)
    else:
        exit(1)
except:
    exit(1)
" 2>/dev/null; then
        print_success "Backend is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        print_error "Backend failed to start within 30 seconds"
        cleanup
        exit 1
    fi
    sleep 1
done

# Wait additional time for backend to fully initialize
print_status "Waiting for backend to fully initialize..."
sleep 5

# Start the MCP server
print_status "Starting MCP server..."
python3 app.py &
MCP_PID=$!

# Wait for MCP server to be ready
print_status "Waiting for MCP server to be ready..."
sleep 3

# Check if both services are running
if kill -0 $BACKEND_PID 2>/dev/null && kill -0 $MCP_PID 2>/dev/null; then
    print_success "All services started successfully!"
    echo ""
    echo "🌐 Services Status:"
    echo "   Backend API: http://localhost:8000"
    echo "   API Docs:    http://localhost:8000/docs"
    echo "   Health:      http://localhost:8000/health"
    echo "   MCP Server:  Running on default port"
    echo ""
    echo "Press Ctrl+C to stop all services"
    
    # Keep script running and monitor processes
    while kill -0 $BACKEND_PID 2>/dev/null && kill -0 $MCP_PID 2>/dev/null; do
        sleep 5
    done
    
    print_warning "One or more services stopped unexpectedly"
    cleanup
else
    print_error "Failed to start one or more services"
    cleanup
    exit 1
fi
