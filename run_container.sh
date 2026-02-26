
set -e
cd "$(dirname "$0")"

IMAGE_NAME=mlflow_example
CONTAINER_NAME="${USER}_mlflow_example"
CONTAINER_PATH=/app
PROJECT_PATH="$(pwd)"

docker build -t "$IMAGE_NAME" .
echo "Starting docker container $CONTAINER_NAME"

docker run -it --rm \
  -u $(id -u):$(id -g) \
  -v "$PROJECT_PATH":"$CONTAINER_PATH" \
  -w "$CONTAINER_PATH" \
  --name "$CONTAINER_NAME" \
  "$IMAGE_NAME" \
  /bin/bash
