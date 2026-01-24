#!/bin/bash
#
# ROS2 Federated Learning System - Run Script
# ============================================
# This script builds and runs the federated learning multi-robot system.
#
# Usage:
#   ./run.sh              # Build and run the system
#   ./run.sh build        # Only build Docker image
#   ./run.sh start        # Start containers (assumes image exists)
#   ./run.sh stop         # Stop all containers
#   ./run.sh logs         # View all logs
#   ./run.sh logs monitor # View specific container logs
#   ./run.sh status       # Check container status
#   ./run.sh clean        # Stop and remove containers/images
#   ./run.sh test         # Run tests inside container
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$SCRIPT_DIR/docker"
IMAGE_NAME="fl-robots:latest"

# Print colored message
print_msg() {
    echo -e "${GREEN}[FL-ROBOTS]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo ""
}

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker Desktop first."
        exit 1
    fi
}

# Build Docker image
build_image() {
    print_header "Building Docker Image"
    print_msg "Building fl-robots:latest..."

    cd "$SCRIPT_DIR"
    docker build -f docker/Dockerfile -t "$IMAGE_NAME" .

    print_msg "Build complete!"
}

# Start containers
start_containers() {
    print_header "Starting Federated Learning System"

    cd "$DOCKER_DIR"
    docker-compose up -d

    print_msg "Containers started!"
    print_msg ""
    print_msg "System Components:"
    print_msg "  • Aggregator  - FedAvg server"
    print_msg "  • Robot 1-3   - Learning agents"
    print_msg "  • Coordinator - Training orchestrator"
    print_msg "  • Monitor     - Metrics dashboard"
    print_msg ""
    print_msg "View logs with: $0 logs"
    print_msg "Stop system with: $0 stop"
}

# Stop containers
stop_containers() {
    print_header "Stopping Federated Learning System"

    cd "$DOCKER_DIR"
    docker-compose down

    print_msg "All containers stopped."
}

# View logs
view_logs() {
    cd "$DOCKER_DIR"

    if [ -z "$1" ]; then
        print_msg "Viewing all container logs (Ctrl+C to exit)..."
        docker-compose logs -f
    else
        print_msg "Viewing logs for $1 (Ctrl+C to exit)..."
        docker-compose logs -f "$1"
    fi
}

# Check status
check_status() {
    print_header "System Status"

    cd "$DOCKER_DIR"
    docker-compose ps

    echo ""
    print_msg "Active ROS2 Topics:"
    docker-compose exec -T aggregator bash -c "source /ros2_ws/install/setup.bash && ros2 topic list 2>/dev/null" || print_warn "Could not list topics"
}

# Clean up
clean_up() {
    print_header "Cleaning Up"

    print_msg "Stopping containers..."
    cd "$DOCKER_DIR"
    docker-compose down --volumes --remove-orphans 2>/dev/null || true

    print_msg "Removing Docker image..."
    docker rmi "$IMAGE_NAME" 2>/dev/null || true

    print_msg "Cleanup complete!"
}

# Run tests
run_tests() {
    print_header "Running Tests"

    cd "$DOCKER_DIR"
    docker-compose run --rm aggregator bash -c "
        cd /ros2_ws &&
        python3 -m pytest tests/ -v --tb=short
    "
}

# Show dashboard
show_dashboard() {
    print_header "Training Dashboard"

    cd "$DOCKER_DIR"
    docker-compose logs monitor 2>&1 | grep -A 20 "FEDERATED LEARNING MONITOR DASHBOARD" | tail -25
}

# Main execution
main() {
    check_docker

    case "${1:-}" in
        build)
            build_image
            ;;
        start)
            start_containers
            ;;
        stop)
            stop_containers
            ;;
        logs)
            view_logs "$2"
            ;;
        status)
            check_status
            ;;
        clean)
            clean_up
            ;;
        test)
            run_tests
            ;;
        dashboard)
            show_dashboard
            ;;
        ""|run)
            # Default: build and run
            print_header "ROS2 Federated Learning System"
            print_msg "Starting complete system..."
            echo ""

            # Check if image exists
            if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
                print_msg "Docker image not found. Building..."
                build_image
            else
                print_msg "Using existing Docker image: $IMAGE_NAME"
            fi

            start_containers

            echo ""
            print_msg "System is running!"
            print_msg ""
            print_msg "Quick Commands:"
            print_msg "  ./run.sh logs          - View all logs"
            print_msg "  ./run.sh logs monitor  - View training dashboard"
            print_msg "  ./run.sh dashboard     - Show current metrics"
            print_msg "  ./run.sh status        - Check container status"
            print_msg "  ./run.sh stop          - Stop the system"
            echo ""
            ;;
        help|--help|-h)
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  (none), run  Build and start the system"
            echo "  build        Build Docker image only"
            echo "  start        Start containers (image must exist)"
            echo "  stop         Stop all containers"
            echo "  logs [name]  View logs (optionally for specific container)"
            echo "  dashboard    Show current training metrics"
            echo "  status       Check container status"
            echo "  test         Run test suite"
            echo "  clean        Stop and remove all containers/images"
            echo "  help         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                  # Build and run everything"
            echo "  $0 logs monitor     # View monitor logs"
            echo "  $0 logs aggregator  # View aggregator logs"
            ;;
        *)
            print_error "Unknown command: $1"
            echo "Run '$0 help' for usage information."
            exit 1
            ;;
    esac
}

main "$@"
