#!/bin/bash
#
# ROS2 Federated Learning System - Run Script
# ============================================
# This script builds and runs the federated learning multi-robot system.
#
# Usage:
#   ./run.sh              # Build and run the lightweight dashboard
#   ./run.sh ros          # Build and run the full ROS stack
#   ./run.sh build lite   # Only build the lightweight dashboard image
#   ./run.sh build ros    # Only build the ROS image
#   ./run.sh start lite   # Start the lightweight dashboard
#   ./run.sh start ros    # Start the ROS stack
#   ./run.sh stop         # Stop all containers
#   ./run.sh logs         # View all logs
#   ./run.sh logs NAME    # View specific container logs
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
LITE_IMAGE_NAME="fl-robots-standalone:latest"
ROS_IMAGE_NAME="fl-robots-ros:latest"
TEST_IMAGE_NAME="fl-robots-test:latest"
COMPOSE_CMD=(docker compose)
LITE_PROFILE="lite"
ROS_PROFILE="ros"
TOOLS_PROFILE="tools"

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

resolve_mode() {
    case "${1:-lite}" in
        lite|light|dashboard)
            echo "lite"
            ;;
        ros|full)
            echo "ros"
            ;;
        *)
            print_error "Unknown mode: $1"
            echo "Use 'lite' or 'ros'."
            exit 1
            ;;
    esac
}

service_running() {
    cd "$DOCKER_DIR"
    "${COMPOSE_CMD[@]}" ps --services --status running | grep -qx "$1"
}

# Build Docker image
build_image() {
    local mode
    mode="$(resolve_mode "${1:-lite}")"
    print_header "Building Docker Image"

    cd "$DOCKER_DIR"
    if [ "$mode" = "lite" ]; then
        print_msg "Building lightweight dashboard image..."
        "${COMPOSE_CMD[@]}" --profile "$LITE_PROFILE" build dashboard
        print_msg "Build complete: $LITE_IMAGE_NAME"
    else
        print_msg "Building full ROS image..."
        "${COMPOSE_CMD[@]}" --profile "$ROS_PROFILE" build aggregator
        print_msg "Build complete: $ROS_IMAGE_NAME"
    fi
}

# Start containers
start_containers() {
    local mode
    mode="$(resolve_mode "${1:-lite}")"
    print_header "Starting Federated Learning System"

    cd "$DOCKER_DIR"
    "${COMPOSE_CMD[@]}" down >/dev/null 2>&1 || true

    if [ "$mode" = "lite" ]; then
        "${COMPOSE_CMD[@]}" --profile "$LITE_PROFILE" up --build -d dashboard
        print_msg "Lightweight dashboard started at http://localhost:5000"
        print_msg "Use '$0 ros' for the full ROS multi-container stack."
    else
        "${COMPOSE_CMD[@]}" --profile "$ROS_PROFILE" up --build -d
        print_msg "Full ROS stack started."
        print_msg "View logs with: $0 logs"
        print_msg "Stop system with: $0 stop"
    fi
}

# Stop containers
stop_containers() {
    print_header "Stopping Federated Learning System"

    cd "$DOCKER_DIR"
    "${COMPOSE_CMD[@]}" down

    print_msg "All containers stopped."
}

# View logs
view_logs() {
    cd "$DOCKER_DIR"

    if [ -z "$1" ]; then
        print_msg "Viewing all container logs (Ctrl+C to exit)..."
        "${COMPOSE_CMD[@]}" logs -f
    else
        print_msg "Viewing logs for $1 (Ctrl+C to exit)..."
        "${COMPOSE_CMD[@]}" logs -f "$1"
    fi
}

# Check status
check_status() {
    print_header "System Status"

    cd "$DOCKER_DIR"
    "${COMPOSE_CMD[@]}" ps

    if service_running dashboard; then
        echo ""
        print_msg "Dashboard URL: http://localhost:5000"
        curl -fsS http://localhost:5000/api/health || print_warn "Dashboard health endpoint is not responding yet"
    elif service_running aggregator; then
        echo ""
        print_msg "Active ROS2 Topics:"
        "${COMPOSE_CMD[@]}" exec -T aggregator bash -c "source /ros2_ws/install/setup.bash && ros2 topic list 2>/dev/null" || print_warn "Could not list topics"
    fi
}

# Clean up
clean_up() {
    print_header "Cleaning Up"

    print_msg "Stopping containers..."
    cd "$DOCKER_DIR"
    "${COMPOSE_CMD[@]}" down --volumes --remove-orphans 2>/dev/null || true

    print_msg "Removing Docker images..."
    docker rmi "$LITE_IMAGE_NAME" "$ROS_IMAGE_NAME" "$TEST_IMAGE_NAME" 2>/dev/null || true

    print_msg "Cleanup complete!"
}

# Run tests
run_tests() {
    print_header "Running Tests"

    cd "$DOCKER_DIR"
    "${COMPOSE_CMD[@]}" --profile "$TOOLS_PROFILE" run --rm test_runner
}

# Show dashboard
show_dashboard() {
    print_header "Training Dashboard"

    cd "$DOCKER_DIR"
    if service_running dashboard; then
        curl -fsS http://localhost:5000/api/status | python3 -c '
import json, sys
data = json.load(sys.stdin)
print(f"state={data[\"system\"][\"controller_state\"]}")
print(f"robots={data[\"system\"][\"robot_count\"]}")
print(f"round={data[\"system\"][\"current_round\"]}")
print(f"avg_accuracy={data[\"metrics\"][\"avg_accuracy\"]:.2f}")
print(f"avg_loss={data[\"metrics\"][\"avg_loss\"]:.3f}")
' || print_warn "Dashboard API is not responding yet"
    else
        "${COMPOSE_CMD[@]}" logs monitor 2>&1 | grep -A 20 "FEDERATED LEARNING MONITOR DASHBOARD" | tail -25
    fi
}

# Main execution
main() {
    check_docker

    case "${1:-}" in
        build)
            build_image "${2:-lite}"
            ;;
        start)
            start_containers "${2:-lite}"
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
        ros|full)
            start_containers "ros"
            ;;
        ""|run)
            print_header "FL-ROBOTS Docker"
            print_msg "Starting lightweight dashboard..."
            start_containers "lite"
            ;;
        help|--help|-h)
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  (none), run        Build and start the lightweight dashboard"
            echo "  ros               Build and start the full ROS stack"
            echo "  build [lite|ros]  Build a Docker image"
            echo "  start [lite|ros]  Start containers"
            echo "  stop         Stop all containers"
            echo "  logs [name]  View logs (optionally for specific container)"
            echo "  dashboard    Show current training metrics"
            echo "  status       Check container status"
            echo "  test         Run test suite"
            echo "  clean        Stop and remove all containers/images"
            echo "  help         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                  # Start the lightweight dashboard"
            echo "  $0 ros              # Start the full ROS stack"
            echo "  $0 logs dashboard   # View dashboard logs"
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
