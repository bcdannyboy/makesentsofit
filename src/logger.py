"""
Logging configuration and utilities.
Provides centralized logging setup with Rich formatting.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console
from rich.traceback import install as install_rich_traceback

# Global console instance
console = Console()

# Logging format
LOG_FORMAT = "%(message)s"
LOG_DATE_FORMAT = "[%X]"

def setup_logging(
    verbose: bool = False,
    log_file: Optional[str] = None,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Configure logging with Rich formatting and optional file output.
    
    Args:
        verbose: Enable verbose (DEBUG) logging
        log_file: Optional path to log file
        log_to_console: Whether to log to console (default: True)
        
    Returns:
        Root logger instance
    """
    # Determine log level
    level = logging.DEBUG if verbose else logging.INFO
    
    # Get root logger
    root_logger = logging.getLogger()
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create handlers list
    handlers = []
    
    # Console handler with Rich
    if log_to_console:
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=verbose,
            rich_tracebacks=True,
            tracebacks_show_locals=verbose
        )
        console_handler.setLevel(level)
        handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        try:
            # Ensure log directory exists
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)  # Always log everything to file
            
            # Simple format for file logs
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            handlers.append(file_handler)
            
        except Exception as e:
            console.print(f"[red]Failed to create log file: {e}[/red]")
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=handlers
    )
    
    # Set levels for third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # Install rich traceback handler
    install_rich_traceback(show_locals=verbose)
    
    # Log initial setup
    logger = logging.getLogger(__name__)
    logger.debug(f"Logging configured: level={logging.getLevelName(level)}, "
                f"console={log_to_console}, file={log_file}")
    
    return root_logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

def log_banner(title: str, subtitle: Optional[str] = None):
    """
    Log a formatted banner message.
    
    Args:
        title: Main banner title
        subtitle: Optional subtitle
    """
    console.print("\n" + "="*50, style="blue")
    console.print(f"[bold cyan]{title}[/bold cyan]", justify="center")
    if subtitle:
        console.print(f"[dim]{subtitle}[/dim]", justify="center")
    console.print("="*50 + "\n", style="blue")

def log_success(message: str):
    """Log a success message with formatting."""
    console.print(f"[green]✓[/green] {message}")

def log_error(message: str):
    """Log an error message with formatting."""
    console.print(f"[red]✗[/red] {message}")

def log_warning(message: str):
    """Log a warning message with formatting."""
    console.print(f"[yellow]⚠[/yellow] {message}")

def log_info(message: str):
    """Log an info message with formatting."""
    console.print(f"[blue]ℹ[/blue] {message}")

def create_log_context(operation: str) -> 'LogContext':
    """
    Create a logging context for an operation.
    
    Args:
        operation: Name of the operation
        
    Returns:
        LogContext instance
    """
    return LogContext(operation)

class LogContext:
    """Context manager for operation logging."""
    
    def __init__(self, operation: str):
        """Initialize log context."""
        self.operation = operation
        self.logger = get_logger(__name__)
        self.start_time = None
    
    def __enter__(self):
        """Enter the context."""
        self.start_time = datetime.now()
        self.logger.info(f"Starting: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context."""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(f"Completed: {self.operation} ({duration:.2f}s)")
        else:
            self.logger.error(f"Failed: {self.operation} ({duration:.2f}s) - {exc_val}")
        
        return False  # Don't suppress exceptions