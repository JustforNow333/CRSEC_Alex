"""
Simulation Logger
Captures all simulation output to a text file for monitoring and debugging
"""

import re
import sys
import datetime
from pathlib import Path


DEFAULT_LOG_BASENAME = "simulation_output.txt"


def log_path_for(sim_code=None):
    """Per-run log filename, derived from the target sim name.

    run_sweep.py launches concurrent subprocesses that share a working
    directory, so a single fixed filename would have every run clobbering
    the same file. Giving each run its own name keeps them separate.
    """
    if not sim_code:
        return DEFAULT_LOG_BASENAME
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", str(sim_code))
    return f"simulation_output_{safe}.txt"

class SimulationLogger:
    def __init__(self, log_file=None):
        # NOTE: deliberately does NOT touch the filesystem. A module-level
        # instance is created at import time, and truncating a log file as a
        # side effect of `import simulation_logger` destroyed the previous
        # run's output. The file is created by configure(), or lazily on the
        # first write.
        self.log_file = log_file
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.start_time = None
        self._header_written = False
        if log_file is not None:
            self.configure(log_file=log_file)

    def configure(self, log_file=None, sim_code=None):
        """Point the logger at a per-run file and write its header."""
        self.log_file = log_file or log_path_for(sim_code)
        self.start_time = datetime.datetime.now()
        self._write_header()
        return self.log_file

    def _ensure_ready(self):
        """Create the log file on first use if configure() was never called."""
        if not self._header_written:
            self.configure(log_file=self.log_file)

    def _write_header(self):
        with open(self.log_file, 'w', encoding="utf-8") as f:
            f.write(f"CRSEC Simulation Output Log\n")
            f.write(f"============================\n")
            f.write(f"Started: {self.start_time.strftime('%B %d, %Y, %H:%M:%S')}\n")
            f.write(f"Log file: {self.log_file}\n\n")
            f.write(f"This file captures:\n")
            f.write(f"- Simulation startup messages\n")
            f.write(f"- Warning messages when invalid addresses are generated\n")
            f.write(f"- Error messages and debugging information\n")
            f.write(f"- General simulation progress\n\n")
            f.write(f"==========================================\n\n")
        self._header_written = True
    
    def start_logging(self):
        """Start capturing all output to the log file"""
        sys.stdout = self
        sys.stderr = self
    
    def stop_logging(self):
        """Stop capturing output and restore original stdout/stderr"""
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
    
    def write(self, text):
        """Write to both console and log file"""
        # Write to original stdout (console)
        self.original_stdout.write(text)
        self.original_stdout.flush()
        
        # Write to log file
        self._ensure_ready()
        with open(self.log_file, 'a', encoding="utf-8") as f:
            f.write(text)
            f.flush()
    
    def flush(self):
        """Flush both console and log file"""
        self.original_stdout.flush()
        self._ensure_ready()
        with open(self.log_file, 'a', encoding="utf-8") as f:
            f.flush()
    
    def log_warning(self, message):
        """Log a warning message with timestamp"""
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        warning_msg = f"[{timestamp}] WARNING: {message}\n"
        self.write(warning_msg)
    
    def log_error(self, message):
        """Log an error message with timestamp"""
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        error_msg = f"[{timestamp}] ERROR: {message}\n"
        self.write(error_msg)
    
    def log_info(self, message):
        """Log an info message with timestamp"""
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        info_msg = f"[{timestamp}] INFO: {message}\n"
        self.write(info_msg)
    
    def log_fallback_usage(self, original_address, fallback_address):
        """Log when a fallback address is used"""
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        fallback_msg = f"[{timestamp}] FALLBACK: Original address '{original_address}' not found, using '{fallback_address}'\n"
        self.write(fallback_msg)
    
    def log_arena_validation(self, generated_arena, available_arenas, fallback_arena):
        """Log arena validation warnings"""
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        validation_msg = f"[{timestamp}] ARENA VALIDATION: Generated '{generated_arena}' not in available arenas: {available_arenas}. Using fallback: {fallback_arena}\n"
        self.write(validation_msg)
    
    def log_sector_validation(self, generated_sector, available_sectors, fallback_sector):
        """Log sector validation warnings"""
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        validation_msg = f"[{timestamp}] SECTOR VALIDATION: Generated '{generated_sector}' not in available sectors: {available_sectors}. Using fallback: {fallback_sector}\n"
        self.write(validation_msg)
    
    def log_api_error(self, error_type, error_message, retry_count=0):
        """Log API errors with retry information"""
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        api_error_msg = f"[{timestamp}] API ERROR ({error_type}): {error_message}"
        if retry_count > 0:
            api_error_msg += f" (Retry {retry_count}/3)"
        api_error_msg += "\n"
        self.write(api_error_msg)

# Global logger instance. Constructed with no path so that merely importing
# this module never creates or truncates a log file; start_simulation_logging()
# gives it a per-run name.
simulation_logger = SimulationLogger()

def start_simulation_logging(sim_code=None):
    """Start logging simulation output to a per-run file."""
    simulation_logger.configure(sim_code=sim_code)
    simulation_logger.start_logging()
    simulation_logger.log_info("Simulation logging started")

def stop_simulation_logging():
    """Stop logging simulation output"""
    simulation_logger.stop_logging()
    simulation_logger.log_info("Simulation logging stopped")
