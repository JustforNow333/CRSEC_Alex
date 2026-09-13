"""
Environment Manager for Reverie Backend Server

This module provides functionality to generate environment files without needing
the frontend server. It simulates the frontend's role of tracking persona positions
and creating environment JSON files that the backend server expects.

Author: Generated to make backend server standalone
"""

import json
import os
from utils import *
from global_methods import *


class EnvironmentManager:
    """
    Manages environment file generation for the backend server.
    This replaces the frontend's role of tracking persona positions.
    """
    
    def __init__(self, sim_code, maze_name="the_ville"):
        """
        Initialize the environment manager.
        
        Args:
            sim_code: The simulation code/name
            maze_name: The maze name (default: "the_ville")
        """
        self.sim_code = sim_code
        self.maze_name = maze_name
        self.sim_folder = f"{fs_storage}/{sim_code}"
        self.environment_folder = f"{self.sim_folder}/environment"
        self.movement_folder = f"{self.sim_folder}/movement"
        
        # Create folders if they don't exist
        create_folder_if_not_there(self.environment_folder)
        create_folder_if_not_there(self.movement_folder)
        
        # Track current persona positions
        self.current_positions = {}
        
    def load_initial_positions(self, step=0):
        """
        Load initial persona positions from an existing environment file.
        
        Args:
            step: The step number to load positions from (default: 0)
        """
        env_file = f"{self.environment_folder}/{step}.json"
        if check_if_file_exists(env_file):
            with open(env_file, 'r', encoding="utf-8") as f:
                data = json.load(f)
                for persona_name, info in data.items():
                    self.current_positions[persona_name] = {
                        "x": info["x"],
                        "y": info["y"],
                        "maze": info.get("maze", self.maze_name)
                    }
        else:
            print(f"Warning: Initial environment file {env_file} not found")
            
    def set_persona_position(self, persona_name, x, y, maze=None):
        """
        Set a persona's position.
        
        Args:
            persona_name: Name of the persona
            x: X coordinate
            y: Y coordinate  
            maze: Maze name (optional, uses default if not provided)
        """
        if maze is None:
            maze = self.maze_name
            
        self.current_positions[persona_name] = {
            "x": x,
            "y": y,
            "maze": maze
        }
        
    def update_positions_from_movement(self, step):
        """
        Update persona positions based on movement file from backend.
        
        Args:
            step: The step number to read movement from
        """
        movement_file = f"{self.movement_folder}/{step}.json"
        if check_if_file_exists(movement_file):
            with open(movement_file, 'r', encoding="utf-8") as f:
                movement_data = json.load(f)
                
            if "persona" in movement_data:
                for persona_name, movement_info in movement_data["persona"].items():
                    if "movement" in movement_info:
                        new_pos = movement_info["movement"]
                        if len(new_pos) >= 2:
                            self.set_persona_position(persona_name, new_pos[0], new_pos[1])
                            
    def generate_environment_file(self, step):
        """
        Generate an environment file for the given step.
        
        Args:
            step: The step number to generate environment for
            
        Returns:
            bool: True if file was created successfully
        """
        if not self.current_positions:
            print(f"Warning: No persona positions set for step {step}")
            return False
            
        # Create environment data structure
        environment_data = {}
        for persona_name, pos_info in self.current_positions.items():
            environment_data[persona_name] = {
                "maze": pos_info["maze"],
                "x": pos_info["x"],
                "y": pos_info["y"]
            }
            
        # Write environment file
        env_file = f"{self.environment_folder}/{step}.json"
        with open(env_file, 'w', encoding="utf-8") as f:
            json.dump(environment_data, f, indent=2)
            
        print(f"Generated environment file: {env_file}")
        return True
        
    def auto_generate_environment_sequence(self, start_step, end_step):
        """
        Automatically generate environment files for a sequence of steps.
        This reads movement files and generates corresponding environment files.
        
        Args:
            start_step: Starting step number
            end_step: Ending step number
        """
        for step in range(start_step, end_step + 1):
            # First, try to update positions from previous movement file
            if step > start_step:
                self.update_positions_from_movement(step - 1)
                
            # Generate environment file for current step
            if self.generate_environment_file(step):
                print(f"Step {step}: Environment file generated")
            else:
                print(f"Step {step}: Failed to generate environment file")
                
    def get_persona_position(self, persona_name):
        """
        Get current position of a persona.
        
        Args:
            persona_name: Name of the persona
            
        Returns:
            dict: Position information or None if not found
        """
        return self.current_positions.get(persona_name)
        
    def list_personas(self):
        """
        List all personas currently being tracked.
        
        Returns:
            list: List of persona names
        """
        return list(self.current_positions.keys())
        
    def reset_positions(self):
        """
        Reset all persona positions.
        """
        self.current_positions = {}
        
    def clone_environment_file(self, source_step, target_step):
        """
        Clone an environment file from one step to another.
        
        Args:
            source_step: Source step number
            target_step: Target step number
            
        Returns:
            bool: True if successful
        """
        source_file = f"{self.environment_folder}/{source_step}.json"
        target_file = f"{self.environment_folder}/{target_step}.json"
        
        if check_if_file_exists(source_file):
            with open(source_file, 'r', encoding="utf-8") as f:
                data = json.load(f)
            with open(target_file, 'w', encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"Cloned environment file from step {source_step} to {target_step}")
            return True
        else:
            print(f"Source environment file {source_file} not found")
            return False


def create_environment_manager(sim_code, maze_name="the_ville"):
    """
    Factory function to create and initialize an environment manager.
    
    Args:
        sim_code: The simulation code/name
        maze_name: The maze name (default: "the_ville")
        
    Returns:
        EnvironmentManager: Initialized environment manager
    """
    manager = EnvironmentManager(sim_code, maze_name)
    
    # Try to load initial positions from step 0
    manager.load_initial_positions(0)
    
    return manager
