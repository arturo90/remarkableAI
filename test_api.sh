#!/bin/bash
# Test script for API endpoints

BASE_URL="http://localhost:8000"

echo "=== Testing RemarkableAI API v2.0.0 ==="
echo ""

# Test health endpoint
echo "1. Testing health endpoint..."
curl -s "$BASE_URL/health" | python -m json.tool
echo ""

# Test notes API
echo "2. Testing notes API..."
echo "   Getting all notes..."
curl -s "$BASE_URL/api/notes/" | python -m json.tool
echo ""

# Test tasks API
echo "3. Testing tasks API..."
echo "   Getting all tasks..."
curl -s "$BASE_URL/api/tasks/" | python -m json.tool
echo ""

# Test task statistics
echo "4. Testing task statistics..."
curl -s "$BASE_URL/api/tasks/statistics" | python -m json.tool
echo ""

echo "=== API Test Complete ==="

