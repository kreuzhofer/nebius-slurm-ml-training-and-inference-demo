#!/bin/bash
# =============================================================================
# query.sh — Send a SQL generation query to the vLLM server.
#
# Usage:
#   bash /mnt/data/demo/scripts/query.sh
#   bash /mnt/data/demo/scripts/query.sh "CREATE TABLE orders (id INT, total DECIMAL)" "What is the sum of all totals?"
#   bash /mnt/data/demo/scripts/query.sh "CREATE TABLE users (id INT, name TEXT)" "List all users" worker-0 8001
# =============================================================================

SCHEMA=${1:-"CREATE TABLE employees (id INT, department TEXT, salary DECIMAL)"}
QUESTION=${2:-"What is the average salary per department?"}
HOST=${3:-$(squeue --noheader -n vllm-serve -o "%N" 2>/dev/null | head -1)}
PORT=${4:-8000}
MODEL=${5:-"/mnt/data/demo/output/qwen3-8b-sql"}

if [ -z "$HOST" ]; then
    echo "Error: No vLLM server found running. Submit it first with:"
    echo "  sbatch /mnt/data/demo/scripts/serve.sbatch"
    exit 1
fi

echo "Server:   $HOST:$PORT"
echo "Schema:   $SCHEMA"
echo "Question: $QUESTION"
echo ""

RESPONSE=$(curl -s "http://${HOST}:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(cat <<EOF
{
  "model": "$MODEL",
  "messages": [
    {"role": "system", "content": "You are a SQL expert. Given a database schema and a question, write the correct SQL query. Output only the SQL query, nothing else."},
    {"role": "user", "content": "Schema:\n$SCHEMA\n\nQuestion: $QUESTION"}
  ]
}
EOF
)")

SQL=$(echo "$RESPONSE" | python -c "
import sys, json
try:
    msg = json.load(sys.stdin)['choices'][0]['message']['content']
    # Strip thinking tags if present
    if '</think>' in msg:
        msg = msg.split('</think>')[-1]
    print(msg.strip())
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    print(sys.stdin.read() if hasattr(sys.stdin, 'read') else '', file=sys.stderr)
    sys.exit(1)
" 2>&1)

echo "SQL: $SQL"
