"""
CLI utilities and validation functions.
Handles command-line argument parsing and validation.
"""
import click
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime

# Valid platforms and formats
VALID_PLATFORMS = {'twitter', 'reddit'}
VALID_FORMATS = {'json', 'csv', 'html'}

def validate_cli_args(
    queries: List[str],
    time_window: int,
    platforms: List[str],
    formats: List[str]
) -> List[str]:
    """
    Validate CLI arguments and return list of errors.
    
    Args:
        queries: List of search queries
        time_window: Number of days to look back
        platforms: List of platforms to search
        formats: List of output formats
        
    Returns:
        List of error messages (empty if all valid)
    """
    errors = []
    
    # Validate queries
    if not queries:
        errors.append("No queries provided")
    else:
        valid_queries = [q for q in queries if q.strip()]
        if not valid_queries:
            errors.append("All queries are empty")
        elif len(valid_queries) != len(queries):
            errors.append(f"Found {len(queries) - len(valid_queries)} empty queries")
    
    # Validate time window
    if time_window < 1:
        errors.append("Time window must be at least 1 day")
    elif time_window > 365:
        errors.append("Time window cannot exceed 365 days")
    
    # Validate platforms
    if not platforms:
        errors.append("No platforms specified")
    else:
        invalid_platforms = [p for p in platforms if p not in VALID_PLATFORMS]
        if invalid_platforms:
            errors.append(f"Invalid platforms: {', '.join(invalid_platforms)}. "
                        f"Valid options: {', '.join(VALID_PLATFORMS)}")
    
    # Validate formats
    if not formats:
        errors.append("No output formats specified")
    else:
        invalid_formats = [f for f in formats if f not in VALID_FORMATS]
        if invalid_formats:
            errors.append(f"Invalid formats: {', '.join(invalid_formats)}. "
                        f"Valid options: {', '.join(VALID_FORMATS)}")
    
    return errors

def validate_queries(queries: List[str]) -> Tuple[List[str], List[str]]:
    """
    Validate and clean query list.
    
    Args:
        queries: Raw list of queries
        
    Returns:
        Tuple of (valid_queries, warnings)
    """
    valid_queries = []
    warnings = []
    
    for query in queries:
        query = query.strip()
        
        if not query:
            continue
            
        # Check query length
        if len(query) < 2:
            warnings.append(f"Query too short (skipped): '{query}'")
            continue
        
        if len(query) > 100:
            warnings.append(f"Query truncated: '{query[:50]}...'")
            query = query[:100]
        
        # Check for special characters that might cause issues
        if any(char in query for char in ['\\', '\n', '\r', '\t']):
            warnings.append(f"Special characters removed from: '{query}'")
            query = query.replace('\\', '').replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        
        valid_queries.append(query)
    
    return valid_queries, warnings

def parse_platforms(platforms_str: str) -> Tuple[List[str], List[str]]:
    """
    Parse and validate platform string.
    
    Args:
        platforms_str: Comma-separated platform string
        
    Returns:
        Tuple of (valid_platforms, invalid_platforms)
    """
    platforms = [p.strip().lower() for p in platforms_str.split(',')]
    valid = [p for p in platforms if p in VALID_PLATFORMS]
    invalid = [p for p in platforms if p not in VALID_PLATFORMS]
    
    return valid, invalid

def format_output_summary(context: Dict[str, Any]) -> str:
    """
    Format context dictionary as a readable summary.
    
    Args:
        context: Analysis context dictionary
        
    Returns:
        Formatted summary string
    """
    summary_lines = [
        "Analysis Configuration Summary",
        "=" * 30,
        f"Version: {context.get('version', 'Unknown')}",
        f"Start Time: {context.get('start_time', 'Unknown')}",
        "",
        "Queries:",
    ]
    
    for i, query in enumerate(context.get('queries', []), 1):
        summary_lines.append(f"  {i}. {query}")
    
    summary_lines.extend([
        "",
        f"Time Window: {context.get('time_window_days', 0)} days",
        f"Platforms: {', '.join(context.get('platforms', []))}",
        f"Output Formats: {', '.join(context.get('formats', []))}",
        f"Output Directory: {context.get('output_directory', 'Unknown')}",
        f"Output Prefix: {context.get('output_prefix', 'Unknown')}",
        f"Visualizations: {'Enabled' if context.get('visualize') else 'Disabled'}",
    ])
    
    return '\n'.join(summary_lines)

def create_progress_bar(current: int, total: int, width: int = 50) -> str:
    """
    Create a text-based progress bar.
    
    Args:
        current: Current progress value
        total: Total value
        width: Width of the progress bar
        
    Returns:
        Progress bar string
    """
    if total == 0:
        return "[" + "=" * width + "]"
    
    progress = int((current / total) * width)
    return "[" + "=" * progress + "-" * (width - progress) + "]"