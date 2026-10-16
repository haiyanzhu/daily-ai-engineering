# Daily AI Engineering

A structured repository tracking hands-on projects, architectures, and daily implementations in AI Engineering.

## Topics & Roadmap

| Topic | Focus Area | Status | Link |
| :--- | :--- | :--- | :--- |
| **RAG** | Embeddings + Chroma vector search, cross-encoder reranking, graded-honesty generation, FastAPI serving | In Progress | [Explore](./RAG) |
| **Agent** | Tool calling, agent loop, termination & retry | Planned | — |
| **Eval** | Golden datasets, RAGAS, LLM-as-judge | Planned | — |
| **Deploy** | Docker, observability, permission filtering | Planned | — |
| **Optimization** | Cost/latency tuning, MCP | Planned | — |


## Prerequisites

- Python 3.11+
- Git

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/haiyanzhu/daily-ai-engineering.git
cd daily-ai-engineering
```

## Usage

Each topic module is self-contained with its own dependencies and configuration.

### Running the RAG Module

#### 1. Environment & Dependencies

```bash
# Navigate to the module directory
cd RAG

# Create and activate a virtual environment (using uv)
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
```

#### 2. Start Backend Service (FastAPI)

In your first terminal window (inside `./RAG`):

```bash
uvicorn app:app --reload
```

The API service runs by default on `http://127.0.0.1:8000`.

#### 3. Start Frontend Test Server

Open a second terminal window and serve the test client:

```bash
# Navigate to the test frontend directory
cd "RAG/fastapi test"

# Launch local HTTP server
python3 -m http.server 8080
```

#### 4. Test the Application

Open your browser and navigate to:

```text
http://localhost:8080/test.html
```

Use the web interface to submit queries to the RAG backend.


## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/) 
