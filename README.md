# LLM-Powered GraphQL Agent

A dockerized LLM-powered agent that translates natural language queries into GraphQL requests against a Jobs API.

## Features

- Translates natural language queries into GraphQL requests
- Returns human-readable responses via a simple HTTP endpoint
- Packaged as a Docker container
- Uses Azure OpenAI for language processing

## Setup & Installation

### Prerequisites

- Docker
- Python 3.9+ (for local development)
- OpenAI API credentials

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```
OPENAI_API_VERSION=2024-12-01-preview
OPENAI_API_ENDPOINT=https://aideveloper-assessment.openai.azure.com/
OPENAI_API_CREDENTIAL=your_api_credential

GRAPHQL_API_URL=https://jluatgraphqlapi-atc2asbnb5gxbehw.westeurope-01.azurewebsites.net/api/tenancy/373daf38-e716-4097-81cc-4f1f1dc99820/graphql-v2
GRAPHQL_SCHEMA_URL=https://jluatgraphqlapi-atc2asbnb5gxbehw.westeurope-01.azurewebsites.net/graphql-v2/schema.graphql
```

## Running Locally

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Run the application:
```
uvicorn main:app --reload
```

3. The API will be available at `http://localhost:8000`

## Building & Running the Docker Container

1. Build the Docker image:
```
docker build -t llm-graphql-agent .
```

2. Run the container:
```
docker run -p 8000:8000 --env-file .env llm-graphql-agent
```

3. The API will be available at `http://localhost:8000`

## API Usage

### Query Endpoint

Send natural language queries to the API:

```
POST /query
Content-Type: application/json

{
  "q": "What are the top paying jobs in London?"
}
```

Response:

```
{
  "answer": "Based on the data, the top paying jobs in London are Senior Software Engineer (£80,000-£100,000), Data Scientist (£70,000-£90,000), and DevOps Engineer (£65,000-£85,000)."
}
```

### Health Check

```
GET /health
```

Response:

```
{
  "status": "healthy"
}
```

## Example Queries

1. "What are the highest paying remote jobs?"
2. "Show me all jobs that require Python skills"
3. "List the entry level positions in New York"
4. "What jobs are available for software engineers with 5+ years of experience?"

## Logs

The application logs the GraphQL queries it generates to stdout, allowing you to see how natural language queries are translated into GraphQL.