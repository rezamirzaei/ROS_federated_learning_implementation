#!/bin/bash
# Build and run the federated learning system
# Usage: ./scripts/run.sh [build|up|down|logs|test]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DOCKER_DIR="$PROJECT_DIR/docker"

cd "$PROJECT_DIR"

case "$1" in
    build)
        echo "Building Docker images..."
        cd "$DOCKER_DIR"
        docker-compose build
        echo "Build complete!"
        ;;

    up)
        echo "Starting federated learning system..."
        cd "$DOCKER_DIR"
        docker-compose up -d
        echo "System started! View logs with: $0 logs"
        ;;

    down)
        echo "Stopping federated learning system..."
        cd "$DOCKER_DIR"
        docker-compose down
        echo "System stopped."
        ;;

    logs)
        cd "$DOCKER_DIR"
        if [ -z "$2" ]; then
            docker-compose logs -f
        else
            docker-compose logs -f "$2"
        fi
        ;;

    test)
        echo "Running tests..."
        cd "$DOCKER_DIR"
        docker-compose run --rm aggregator bash -c "
            cd /ros2_ws &&
            python3 -m pytest tests/ -v
        "
        ;;

    shell)
        echo "Opening shell in container..."
        cd "$DOCKER_DIR"
        docker-compose run --rm aggregator bash
        ;;

    topics)
        echo "Listing ROS2 topics..."
        cd "$DOCKER_DIR"
        docker-compose exec aggregator bash -c "
            source /ros2_ws/install/setup.bash &&
            ros2 topic list
        "
        ;;

    nodes)
        echo "Listing ROS2 nodes..."
        cd "$DOCKER_DIR"
        docker-compose exec aggregator bash -c "
            source /ros2_ws/install/setup.bash &&
            ros2 node list
        "
        ;;

    *)
        echo "Usage: $0 {build|up|down|logs|test|shell|topics|nodes}"
        echo ""
        echo "Commands:"
        echo "  build   - Build Docker images"
        echo "  up      - Start all containers in background"
        echo "  down    - Stop all containers"
        echo "  logs    - View container logs (optional: specify container name)"
        echo "  test    - Run test suite"
        echo "  shell   - Open bash shell in container"
        echo "  topics  - List active ROS2 topics"
        echo "  nodes   - List active ROS2 nodes"
        exit 1
        ;;
esac
