# Basic start with integrated UI
python unified_server.py

# Start with legacy UI
python unified_server.py --ui legacy

# Start with API server enabled
python unified_server.py --api

# Start with custom port
python unified_server.py --port 8080

# Start in settings view
python unified_server.py --settings

# Full example with all options
python unified_server.py --ui integrated --api --api-port 8000 --port 8501 --debug