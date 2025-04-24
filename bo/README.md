# Agentor BackOffice

A comprehensive BackOffice application for managing the Agentor framework, providing tools for agent management, tool configuration, workflow orchestration, and LLM connection management.

## Features

- **Agent Management**: Create, configure, and monitor agents
- **Tool Management**: Register, configure, and analyze tools
- **Workflow Designer**: Create and manage agent workflows
- **LLM Connection Management**: Configure and monitor LLM providers
- **User Authentication**: Secure access with role-based permissions

## Architecture

The application is built with a modern stack:

- **Backend**: FastAPI (Python)
- **Frontend**: React with TypeScript
- **Database**: SQLite (development) / PostgreSQL (production)
- **Authentication**: JWT-based authentication

## Project Structure

```
bo/
├── backend/               # FastAPI backend
│   ├── api/               # API endpoints
│   ├── auth/              # Authentication
│   ├── db/                # Database models and utilities
│   ├── schemas/           # Pydantic schemas
│   └── services/          # Business logic
├── frontend/              # React frontend
│   ├── public/            # Static assets
│   └── src/               # Source code
│       ├── components/    # React components
│       ├── contexts/      # React contexts
│       ├── layouts/       # Page layouts
│       └── pages/         # Page components
└── docs/                  # Documentation
```

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/agentor.git
   cd agentor/bo
   ```

2. Run the development server:
   ```bash
   python run_dev.py
   ```

   This will:
   - Create a Python virtual environment if it doesn't exist
   - Install backend dependencies
   - Start the backend server on port 8000
   - Install frontend dependencies
   - Start the frontend server on port 3000
   - Open the application in your browser

### Default Login

- Username: `admin`
- Password: `Admin123`

## Development

### Backend

The backend is built with FastAPI and uses SQLAlchemy for database operations.

To run the backend separately:

```bash
cd bo/backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend

The frontend is built with React and TypeScript.

To run the frontend separately:

```bash
cd bo/frontend
npm install
npm start
```

## API Documentation

When the backend is running, you can access the API documentation at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## License

This project is licensed under the MIT License - see the LICENSE file for details.
