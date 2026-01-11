#!/usr/bin/env python3
"""
Launcher script for AI Infrastructure Model Streamlit applications.
This script orchestrates the launching of multiple Streamlit applications
and handles ancillary tasks such as fetching token prices and cleaning up port usage.
"""

import concurrent.futures
import os
import platform
import signal
import socket
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass
from typing import List

from src.data_fetchers.TokenPriceFetcher import TokenPriceFetcher


@dataclass
class StreamlitApp:
    """Configuration for a Streamlit application."""
    name: str
    script: str
    port: int


@dataclass
class AppProcess:
    """Running process for a Streamlit application."""
    app: StreamlitApp
    process: subprocess.Popen


# Application configurations
STREAMLIT_APPS = [
    StreamlitApp("GPU Cloud Model App", "src/streamlit/gpu_cloud_model_app.py", 8501),
    StreamlitApp("Buildout Calculator App", "src/streamlit/buildout_calc_app.py", 8502),
    StreamlitApp("Inference Demand App", "src/streamlit/inference_demand_app.py", 8503),
    StreamlitApp("Datacenter Retrofit App", "src/streamlit/datacenter_retrofit_app.py", 8504),
    StreamlitApp("AI Value Chain App", "src/streamlit/value_chain_app.py", 8505),
    StreamlitApp("Labor Value Share App", "src/streamlit/labor_value_share_app.py", 8506),
    StreamlitApp("AI Lab Liability App", "src/streamlit/ai_lab_liability_app.py", 8507),
    StreamlitApp("Company Funding App", "src/streamlit/company_funding_app.py", 8508),
]


def wait_for_port(port: int, timeout: int = 30) -> bool:
    """
    Check if a port is accepting connections.
    
    Args:
        port: The port number to check
        timeout: Maximum time to wait in seconds
        
    Returns:
        True if the port is available, False otherwise
        
    Raises:
        TimeoutError: If the port isn't available within the timeout period
    """
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timed out waiting for port {port} to be ready")

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('localhost', port))
                if result == 0:  # Port is open and accepting connections
                    return True
        except socket.error:
            pass

        time.sleep(0.2)

def fetch_token_prices() -> str:
    """
    Fetch token prices after launching apps.

    Returns:
        Path to the CSV file containing the token prices
    """
    token_price_fetcher = TokenPriceFetcher()
    return token_price_fetcher.get_model_prices()

def _diagnose_port(port: int) -> None:
    """Print owning process information for a busy port (Unix-like systems)."""
    try:
        result = subprocess.run(
            f"lsof -i :{port} -P -n -sTCP:LISTEN", shell=True, capture_output=True, text=True
        )
        if result.stdout.strip():
            print(f"Port {port} is in use by:\n{result.stdout}")
    except Exception:
        pass


def _cleanup_single_port(port: int) -> None:
    """
    Clean up a single port (helper for parallel cleanup).

    Args:
        port: Port number to clean up
    """
    if platform.system() == "Windows":
        subprocess.run(f"FOR /F \"tokens=5\" %P IN ('netstat -ano ^| findstr {port}') DO taskkill /F /PID %P", shell=True)
    else:  # Unix-like systems (Linux, macOS)
        try:
            result = subprocess.run(
                f"lsof -i :{port} -t",
                shell=True,
                capture_output=True,
                text=True
            )

            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                _diagnose_port(port)
                for pid in pids:
                    if pid:
                        print(f"Killing process {pid} using port {port}")
                        try:
                            os.kill(int(pid), signal.SIGTERM)
                        except ProcessLookupError:
                            pass
        except Exception as e:
            print(f"Error cleaning up port {port}: {e}")


def cleanup_ports(ports: List[int]) -> None:
    """
    Free up the ports that the Streamlit apps use in parallel.

    Args:
        ports: List of port numbers to clean up
    """
    print("Cleaning up ports...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(_cleanup_single_port, ports)


def _is_port_in_use(port: int) -> bool:
    """
    Check if a port is currently in use.

    Args:
        port: Port number to check

    Returns:
        True if port is in use, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.1)
            return s.connect_ex(('localhost', port)) == 0
    except socket.error:
        return False


def start_app_processes(apps: List[StreamlitApp]) -> List[AppProcess]:
    """
    Start all Streamlit app processes.

    Args:
        apps: List of Streamlit app configurations

    Returns:
        List of running app processes
    """
    # Only cleanup ports that are actually in use
    busy_ports = [app.port for app in apps if _is_port_in_use(app.port)]
    if busy_ports:
        cleanup_ports(busy_ports)

    processes = []
    for app in apps:
        print(f"Starting {app.name} on port {app.port}...")

        cmd = [
            "uv", "run", "python", "-m", "streamlit", "run",
            app.script,
            "--server.port", str(app.port),
            "--server.headless", "true"
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        processes.append(AppProcess(app=app, process=process))

    return processes


def check_app_ready(app_process: AppProcess) -> StreamlitApp:
    """
    Wait for an app's port to be ready.

    Args:
        app_process: The app process to check

    Returns:
        The app configuration once ready
    """
    print(f"Waiting for {app_process.app.name} to be ready...")
    wait_for_port(app_process.app.port)
    return app_process.app


def wait_for_apps_ready(app_processes: List[AppProcess]) -> List[StreamlitApp]:
    """
    Wait for all app ports to be ready in parallel.

    Args:
        app_processes: List of running app processes

    Returns:
        List of ready apps sorted by port
    """
    print("Waiting for all apps to be ready...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        ready_apps = list(executor.map(check_app_ready, app_processes))
    return sorted(ready_apps, key=lambda app: app.port)


def open_all_browsers(apps: List[StreamlitApp]) -> None:
    """
    Open all app URLs in browser tabs.

    Args:
        apps: List of apps to open (should be sorted by port)
    """
    for app in apps:
        url = f"http://localhost:{app.port}"
        print(f"Opening {app.name} at {url}")
        webbrowser.open(url)


def monitor_processes(app_processes: List[AppProcess]) -> None:
    """
    Monitor running processes until interrupted.

    Args:
        app_processes: List of running app processes
    """
    print("\nAll apps are running. Press Ctrl+C to stop.\n")
    while True:
        time.sleep(1)
        for app_proc in app_processes:
            if app_proc.process.poll() is not None:
                print(f"{app_proc.app.name} has terminated unexpectedly with code {app_proc.process.returncode}")


def shutdown_apps(app_processes: List[AppProcess], apps: List[StreamlitApp]) -> None:
    """
    Shutdown all running app processes and clean up ports.

    Args:
        app_processes: List of running app processes
        apps: List of app configurations for port cleanup
    """
    print("\nShutting down apps...")

    # Terminate all processes at once
    for app_proc in app_processes:
        if app_proc.process.poll() is None:
            print(f"Terminating {app_proc.app.name}...")
            app_proc.process.terminate()

    # Wait for all processes in parallel
    def wait_for_process(app_proc: AppProcess) -> None:
        if app_proc.process.poll() is None:
            try:
                app_proc.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"Killing {app_proc.app.name}...")
                app_proc.process.kill()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(wait_for_process, app_processes)

    cleanup_ports([app.port for app in apps])


def launch_streamlit_apps() -> None:
    """
    Launch all Streamlit applications and open them in the browser.
    """
    app_processes = []
    try:
        # Start all app processes
        app_processes = start_app_processes(STREAMLIT_APPS)

        # Wait for all ports to be ready in parallel
        ready_apps = wait_for_apps_ready(app_processes)

        # Open all browsers in port order
        open_all_browsers(ready_apps)

        # Fetch token prices after everything is running
        print("\nFetching token prices in background...")
        price_file = fetch_token_prices()
        print(f"Token prices saved to {price_file}")

        # Monitor processes until interrupted
        monitor_processes(app_processes)

    except KeyboardInterrupt:
        pass
    finally:
        shutdown_apps(app_processes, STREAMLIT_APPS)

def main() -> int:
    """
    Main function to orchestrate the launching of the apps.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    print("=" * 80)
    print("Starting AI Infrastructure Model Streamlit Applications")
    print("=" * 80)

    try:
        launch_streamlit_apps()
        return 0
    except KeyboardInterrupt:
        print("\nShutdown requested...")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        cleanup_ports([app.port for app in STREAMLIT_APPS])
        return 1

if __name__ == "__main__":
    sys.exit(main())
