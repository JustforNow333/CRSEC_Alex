"""
Simulation Logger
Captures all simulation output to a text file for monitoring and debugging
"""

import sys
import datetime
from pathlib import Path

class SimulationLogger:
    def __init__(self, log_file="simulation_output.txt"):
        self.log_file = log_file
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Create log file with timestamp
        self.start_time = datetime.datetime.now()
        with open(self.log_file, 'w', encoding="utf-8") as f:
            f.write(f"CRSEC Simulation Output Log\n")
            f.write(f"============================\n")
            f.write(f"Started: {self.start_time.strftime('%B %d, %Y, %H:%M:%S')}\n\n")
            f.write(f"This file captures:\n")
            f.write(f"- Simulation startup messages\n")
            f.write(f"- Warning messages when invalid addresses are generated\n")
            f.write(f"- Error messages and debugging information\n")
            f.write(f"- General simulation progress\n\n")
            f.write(f"The simulation will write to this file when it encounters:\n")
            f.write(f"- Invalid maze addresses (like 'the Ville:Giorgio Rossi's apartment:kitchen')\n")
            f.write(f"- Fallback location usage\n")
            f.write(f"- Arena validation warnings\n")
            f.write(f"- Sector validation warnings\n\n")
            f.write(f"==========================================\n\n")
    
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
        with open(self.log_file, 'a', encoding="utf-8") as f:
            f.write(text)
            f.flush()
    
    def flush(self):
        """Flush both console and log file"""
        self.original_stdout.flush()
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

# Global logger instance
simulation_logger = SimulationLogger()

def start_simulation_logging():
    """Start logging simulation output"""
    simulation_logger.start_logging()
    simulation_logger.log_info("Simulation logging started")

def stop_simulation_logging():
    """Stop logging simulation output"""
    simulation_logger.stop_logging()
    simulation_logger.log_info("Simulation logging stopped")
