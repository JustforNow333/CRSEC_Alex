"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: reverie.py
Description: This is the main program for running generative agent simulations
that defines the ReverieServer class. This class maintains and records all  
states related to the simulation. The primary mode of interaction for those  
running the simulation should be through the open_server function, which  
enables the simulator to input command-line prompts for running and saving  
the simulation, among other tasks.

Release note (June 14, 2023) -- Reverie implements the core simulation 
mechanism described in my paper entitled "Generative Agents: Interactive 
Simulacra of Human Behavior." If you are reading through these lines after 
having read the paper, you might notice that I use older terms to describe 
generative agents and their cognitive modules here. Most notably, I use the 
term "personas" to refer to generative agents, "associative memory" to refer 
to the memory stream, and "reverie" to refer to the overarching simulation 
framework.
"""
import json
import numpy
import datetime
import pickle
import time
import math
import os
import shutil
import traceback
import openai

from selenium import webdriver

from global_methods import *
from utils import *
from maze import *
from persona.persona import *

from norm.creation import *
from norm.norm_save import *
from norm.norm_evaluate import *
from environment_manager import create_environment_manager
from simulation_logger import start_simulation_logging, stop_simulation_logging, simulation_logger

##############################################################################
#                                  REVERIE                                   #
##############################################################################

def _get_last_completed_step(sim_folder):
  """Return the last step number that has a movement file, or -1 if none."""
  movement_dir = os.path.join(sim_folder, "movement")
  if not os.path.isdir(movement_dir):
    return -1
  steps = []
  for f in os.listdir(movement_dir):
    if f.endswith(".json"):
      try:
        steps.append(int(f.replace(".json", "")))
      except ValueError:
        pass
  return max(steps) if steps else -1


class ReverieServer: 
  def __init__(self, 
               fork_sim_code,
               sim_code,
               resume=False):
    # <sim_code> indicates our current simulation.
    self.sim_code = sim_code
    sim_folder = f"{fs_storage}/{self.sim_code}"

    if resume:
      # RESUME: open existing simulation without copying. Infer step from movement files.
      if not os.path.isdir(sim_folder):
        raise FileNotFoundError(f"Cannot resume: simulation folder not found: {sim_folder}")
      with open(f"{sim_folder}/reverie/meta.json", encoding="utf-8") as json_file:
        reverie_meta = json.load(json_file)
      self.fork_sim_code = reverie_meta.get("fork_sim_code", "")
      last_step = _get_last_completed_step(sim_folder)
      if last_step < 0:
        raise ValueError(f"Cannot resume: no movement files in {sim_folder}/movement/")
      # Next step to run is last_step + 1
      self.step = last_step + 1
      reverie_meta["step"] = self.step
      sec_per_step = reverie_meta['sec_per_step']
      self.start_time = datetime.datetime.strptime(
                          f"{reverie_meta['start_date']}, 00:00:00",
                          "%B %d, %Y, %H:%M:%S")
      self.curr_time = self.start_time + datetime.timedelta(seconds=(last_step + 1) * sec_per_step)
      reverie_meta["curr_time"] = self.curr_time.strftime("%B %d, %Y, %H:%M:%S")
      with open(f"{sim_folder}/reverie/meta.json", "w", encoding="utf-8") as outfile:
        outfile.write(json.dumps(reverie_meta, indent=2))
      init_env_file = f"{sim_folder}/environment/{last_step}.json"
      if not check_if_file_exists(init_env_file):
        raise FileNotFoundError(f"Cannot resume: missing {init_env_file}")
    else:
      # FORKING FROM A PRIOR SIMULATION:
      # <fork_sim_code> indicates the simulation we are forking from.
      self.fork_sim_code = fork_sim_code
      fork_folder = f"{fs_storage}/{self.fork_sim_code}"
      copyanything(fork_folder, sim_folder)

      with open(f"{sim_folder}/reverie/meta.json", encoding="utf-8") as json_file:
        reverie_meta = json.load(json_file)

      with open(f"{sim_folder}/reverie/meta.json", "w", encoding="utf-8") as outfile:
        reverie_meta["fork_sim_code"] = fork_sim_code
        outfile.write(json.dumps(reverie_meta, indent=2))

      self.step = reverie_meta['step']
      self.start_time = datetime.datetime.strptime(
                          f"{reverie_meta['start_date']}, 00:00:00",
                          "%B %d, %Y, %H:%M:%S")
      self.curr_time = datetime.datetime.strptime(reverie_meta['curr_time'],
                                                  "%B %d, %Y, %H:%M:%S")
      init_env_file = f"{sim_folder}/environment/{str(self.step)}.json"
      sec_per_step = reverie_meta['sec_per_step']

    # LOADING REVERIE'S GLOBAL VARIABLES (shared by both fork and resume)
    self.sec_per_step = reverie_meta['sec_per_step']
    self.maze = Maze(reverie_meta['maze_name'])

    # SETTING UP PERSONAS IN REVERIE
    # <personas> is a dictionary that takes the persona's full name as its 
    # keys, and the actual persona instance as its values.
    # This dictionary is meant to keep track of all personas who are part of
    # the Reverie instance. 
    # e.g., ["Isabella Rodriguez"] = Persona("Isabella Rodriguezs")
    self.personas = dict()
    # <personas_tile> is a dictionary that contains the tile location of
    # the personas (!-> NOT px tile, but the actual tile coordinate).
    # The tile take the form of a set, (row, col). 
    # e.g., ["Isabella Rodriguez"] = (58, 39)
    self.personas_tile = dict()
    
    # # <persona_convo_match> is a dictionary that describes which of the two
    # # personas are talking to each other. It takes a key of a persona's full
    # # name, and value of another persona's full name who is talking to the 
    # # original persona. 
    # # e.g., dict["Isabella Rodriguez"] = ["Maria Lopez"]
    # self.persona_convo_match = dict()
    # # <persona_convo> contains the actual content of the conversations. It
    # # takes as keys, a pair of persona names, and val of a string convo. 
    # # Note that the key pairs are *ordered alphabetically*. 
    # # e.g., dict[("Adam Abraham", "Zane Xu")] = "Adam: baba \n Zane:..."
    # self.persona_convo = dict()

    # Loading in all personas. 
    init_env_file = f"{sim_folder}/environment/{str(self.step)}.json"
    init_env = json.load(open(init_env_file, encoding="utf-8"))
    for persona_name in reverie_meta['persona_names']: 
      persona_folder = f"{sim_folder}/personas/{persona_name}"
      p_x = init_env[persona_name]["x"]
      p_y = init_env[persona_name]["y"]
      curr_persona = Persona(persona_name, persona_folder)

      self.personas[persona_name] = curr_persona
      self.personas_tile[persona_name] = (p_x, p_y)
      self.maze.tiles[p_y][p_x]["events"].add(curr_persona.scratch
                                              .get_curr_event_and_desc())

    # REVERIE SETTINGS PARAMETERS:  
    # <server_sleep> denotes the amount of time that our while loop rests each
    # cycle; this is to not kill our machine. 
    self.server_sleep = 0.1

    # SIGNALING THE FRONTEND SERVER: 
    # curr_sim_code.json contains the current simulation code, and
    # curr_step.json contains the current step of the simulation. These are 
    # used to communicate the code and step information to the frontend. 
    # Note that step file is removed as soon as the frontend opens up the 
    # simulation. 
    curr_sim_code = dict()
    curr_sim_code["sim_code"] = self.sim_code
    with open(f"{fs_temp_storage}/curr_sim_code.json", "w", encoding="utf-8") as outfile:
      outfile.write(json.dumps(curr_sim_code, indent=2))
    
    curr_step = dict()
    curr_step["step"] = self.step
    with open(f"{fs_temp_storage}/curr_step.json", "w", encoding="utf-8") as outfile:
      outfile.write(json.dumps(curr_step, indent=2))

    # ENVIRONMENT MANAGER: 
    # Initialize environment manager for standalone operation
    self.env_manager = create_environment_manager(self.sim_code, self.maze.maze_name)
    
    # Set initial positions from personas_tile
    for persona_name, (x, y) in self.personas_tile.items():
      self.env_manager.set_persona_position(persona_name, x, y)


  def save(self): 
    """
    Save all Reverie progress -- this includes Reverie's global state as well
    as all the personas.  

    INPUT
      None
    OUTPUT 
      None
      * Saves all relevant data to the designated memory directory
    """
    # <sim_folder> points to the current simulation folder.
    sim_folder = f"{fs_storage}/{self.sim_code}"

    # Save Reverie meta information.
    reverie_meta = dict() 
    reverie_meta["fork_sim_code"] = self.fork_sim_code
    reverie_meta["start_date"] = self.start_time.strftime("%B %d, %Y")
    reverie_meta["curr_time"] = self.curr_time.strftime("%B %d, %Y, %H:%M:%S")
    reverie_meta["sec_per_step"] = self.sec_per_step
    reverie_meta["maze_name"] = self.maze.maze_name
    reverie_meta["persona_names"] = list(self.personas.keys())
    reverie_meta["step"] = self.step
    reverie_meta_f = f"{sim_folder}/reverie/meta.json"
    with open(reverie_meta_f, "w", encoding="utf-8") as outfile:
      outfile.write(json.dumps(reverie_meta, indent=2))

    # Save the personas.
    for persona_name, persona in self.personas.items(): 
      save_folder = f"{sim_folder}/personas/{persona_name}/bootstrap_memory"
      persona.scratch.norm_count = persona.norm_database.norm_count
      persona.scratch.act_norm_count = persona.norm_database.act_norm_count
      persona.save(save_folder)

    for persona_name, persona in self.personas.items(): 
      save_folder = f"{sim_folder}/personas/{persona_name}/norms"
      
      print("persona.norm_database.norm_count:",persona.norm_database.norm_count)
      norm_save(persona,save_folder)


  def run_standalone(self, steps):
    """
    Run the simulation in standalone mode without requiring frontend server.
    This generates environment files automatically and runs the simulation.

    INPUT
      steps: Number of steps to run the simulation
    OUTPUT 
      None
    """
    # Start simulation logging
    start_simulation_logging()
    
    print(f"Running simulation in standalone mode for {steps} steps...")
    print(f"Simulation: {self.sim_code}")
    print(f"Current step: {self.step}")
    print(f"Current time: {self.curr_time.strftime('%B %d, %Y, %H:%M:%S')}")
    
    # Generate initial environment file if it doesn't exist
    initial_env_file = f"{fs_storage}/{self.sim_code}/environment/{self.step}.json"
    if not check_if_file_exists(initial_env_file):
      print("Generating initial environment file...")
      self.env_manager.generate_environment_file(self.step)
    
    try:
      # Run the simulation
      self.start_server(steps)
      
      print(f"Simulation completed. Final step: {self.step}")
      print(f"Final time: {self.curr_time.strftime('%B %d, %Y, %H:%M:%S')}")
    except openai.error.ServiceUnavailableError as e:
      simulation_logger.log_error(f"OpenAI API service unavailable: {str(e)}")
      simulation_logger.log_error("Simulation will pause for 30 seconds and retry...")
      print(f"OpenAI API service unavailable: {str(e)}")
      print("Simulation will pause for 30 seconds and retry...")
      time.sleep(30)  # Wait 30 seconds
      # Try to continue the simulation
      try:
        self.start_server(steps)
        print(f"Simulation completed after retry. Final step: {self.step}")
        print(f"Final time: {self.curr_time.strftime('%B %d, %Y, %H:%M:%S')}")
      except Exception as retry_e:
        simulation_logger.log_error(f"Simulation failed after retry: {str(retry_e)}")
        raise
    except Exception as e:
      simulation_logger.log_error(f"Simulation crashed: {str(e)}")
      simulation_logger.log_error(f"Traceback: {traceback.format_exc()}")
      raise
    finally:
      # Stop simulation logging
      stop_simulation_logging()


  def start_path_tester_server(self): 
    """
    Starts the path tester server. This is for generating the spatial memory
    that we need for bootstrapping a persona's state. 

    To use this, you need to open server and enter the path tester mode, and
    open the front-end side of the browser. 

    INPUT 
      None
    OUTPUT 
      None
      * Saves the spatial memory of the test agent to the path_tester_env.json
        of the temp storage. 
    """
    def print_tree(tree): 
      def _print_tree(tree, depth):
        dash = " >" * depth

        if type(tree) == type(list()): 
          if tree:
            print (dash, tree)
          return 

        for key, val in tree.items(): 
          if key: 
            print (dash, key)
          _print_tree(val, depth+1)
      
      _print_tree(tree, 0)

    # <curr_vision> is the vision radius of the test agent. Recommend 8 as 
    # our default. 
    curr_vision = 8
    # <s_mem> is our test spatial memory. 
    s_mem = dict()

    # The main while loop for the test agent. 
    while (True): 
      try: 
        curr_dict = {}
        tester_file = fs_temp_storage + "/path_tester_env.json"
        if check_if_file_exists(tester_file): 
          with open(tester_file, encoding="utf-8") as json_file:
            curr_dict = json.load(json_file)
            os.remove(tester_file)
          
          # Current camera location
          curr_sts = self.maze.sq_tile_size
          curr_camera = (int(math.ceil(curr_dict["x"]/curr_sts)), 
                         int(math.ceil(curr_dict["y"]/curr_sts))+1)
          curr_tile_det = self.maze.access_tile(curr_camera)

          # Initiating the s_mem
          world = curr_tile_det["world"]
          if curr_tile_det["world"] not in s_mem: 
            s_mem[world] = dict()

          # Iterating throughn the nearby tiles.
          nearby_tiles = self.maze.get_nearby_tiles(curr_camera, curr_vision)
          for i in nearby_tiles: 
            i_det = self.maze.access_tile(i)
            if (curr_tile_det["sector"] == i_det["sector"] 
                and curr_tile_det["arena"] == i_det["arena"]): 
              if i_det["sector"] != "": 
                if i_det["sector"] not in s_mem[world]: 
                  s_mem[world][i_det["sector"]] = dict()
              if i_det["arena"] != "": 
                if i_det["arena"] not in s_mem[world][i_det["sector"]]: 
                  s_mem[world][i_det["sector"]][i_det["arena"]] = list()
              if i_det["game_object"] != "": 
                if (i_det["game_object"] 
                    not in s_mem[world][i_det["sector"]][i_det["arena"]]):
                  s_mem[world][i_det["sector"]][i_det["arena"]] += [
                                                         i_det["game_object"]]

        # Incrementally outputting the s_mem and saving the json file. 
        print ("= " * 15)
        out_file = fs_temp_storage + "/path_tester_out.json"
        with open(out_file, "w", encoding="utf-8") as outfile:
          outfile.write(json.dumps(s_mem, indent=2))
        print_tree(s_mem)

      except:
        pass

      time.sleep(self.server_sleep * 10)


  def start_server(self, int_counter): 
    """
    The main backend server of Reverie. 
    This function retrieves the environment file from the frontend to 
    understand the state of the world, calls on each personas to make 
    decisions based on the world state, and saves their moves at certain step
    intervals. 
    INPUT
      int_counter: Integer value for the number of steps left for us to take
                   in this iteration. 
    OUTPUT 
      None
    """
    # <sim_folder> points to the current simulation folder.
    sim_folder = f"{fs_storage}/{self.sim_code}"

    # When a persona arrives at a game object, we give a unique event
    # to that object. 
    # e.g., ('double studio[...]:bed', 'is', 'unmade', 'unmade')
    # Later on, before this cycle ends, we need to return that to its 
    # initial state, like this: 
    # e.g., ('double studio[...]:bed', None, None, None)
    # So we need to keep track of which event we added. 
    # <game_obj_cleanup> is used for that. 
    game_obj_cleanup = dict()

    # The main while loop of Reverie. 
    while (True): 
      # Done with this iteration if <int_counter> reaches 0. 
      if int_counter == 0:
        try:
          self.save()
        except Exception:
          pass
        break

      # STANDALONE ENVIRONMENT GENERATION:
      # Instead of waiting for frontend to create environment files, 
      # we generate them using the environment manager based on current positions
      curr_env_file = f"{sim_folder}/environment/{self.step}.json"
      
      # Update environment manager with current positions
      for persona_name, (x, y) in self.personas_tile.items():
        self.env_manager.set_persona_position(persona_name, x, y)
      
      # Generate environment file for current step
      env_retrieved = self.env_manager.generate_environment_file(self.step)
      
      if env_retrieved:
        # Load the generated environment file
        try:
          with open(curr_env_file, encoding="utf-8") as json_file:
            new_env = json.load(json_file)
        except:
          env_retrieved = False
          pass
      
        if env_retrieved: 
          # This is where we go through <game_obj_cleanup> to clean up all 
          # object actions that were used in this cylce. 
          for key, val in game_obj_cleanup.items(): 
            # We turn all object actions to their blank form (with None). 
            self.maze.turn_event_from_tile_idle(key, val)
          # Then we initialize game_obj_cleanup for this cycle. 
          game_obj_cleanup = dict()

          # We first move our personas in the backend environment to match 
          # the frontend environment. 
          for persona_name, persona in self.personas.items(): 
            # <curr_tile> is the tile that the persona was at previously. 
            curr_tile = self.personas_tile[persona_name]
            # <new_tile> is the tile that the persona will move to right now,
            # during this cycle. 
            new_tile = (new_env[persona_name]["x"], 
                        new_env[persona_name]["y"])

            # We actually move the persona on the backend tile map here. 
            self.personas_tile[persona_name] = new_tile
            self.maze.remove_subject_events_from_tile(persona.name, curr_tile)
            self.maze.add_event_from_tile(persona.scratch
                                         .get_curr_event_and_desc(), new_tile)

            # Now, the persona will travel to get to their destination. *Once*
            # the persona gets there, we activate the object action.
            if not persona.scratch.planned_path: 
              # We add that new object action event to the backend tile map. 
              # At its creation, it is stored in the persona's backend. 
              game_obj_cleanup[persona.scratch
                               .get_curr_obj_event_and_desc()] = new_tile
              self.maze.add_event_from_tile(persona.scratch
                                     .get_curr_obj_event_and_desc(), new_tile)
              # We also need to remove the temporary blank action for the 
              # object that is currently taking the action. 
              blank = (persona.scratch.get_curr_obj_event_and_desc()[0], 
                       None, None, None)
              self.maze.remove_event_from_tile(blank, new_tile)

          # Then we need to actually have each of the personas perceive and
          # move. The movement for each of the personas comes in the form of
          # x y coordinates where the persona will move towards. e.g., (50, 34)
          # This is where the core brains of the personas are invoked. 
          movements = {"persona": dict(), 
                       "meta": dict()}
          for persona_name, persona in self.personas.items(): 
            # <next_tile> is a x,y coordinate. e.g., (58, 9)
            # <pronunciatio> is an emoji. e.g., "\ud83d\udca4"
            # <description> is a string description of the movement. e.g., 
            #   writing her next novel (editing her novel) 
            #   @ double studio:double studio:common room:sofa
            next_tile, pronunciatio, description = persona.move(
              self.maze, self.personas, self.personas_tile[persona_name], 
              self.curr_time)
            movements["persona"][persona_name] = {}
            movements["persona"][persona_name]["movement"] = next_tile
            movements["persona"][persona_name]["pronunciatio"] = pronunciatio
            movements["persona"][persona_name]["description"] = description
            movements["persona"][persona_name]["chat"] = (persona
                                                          .scratch.chat)

          # Include the meta information about the current stage in the 
          # movements dictionary. 
          movements["meta"]["curr_time"] = (self.curr_time 
                                             .strftime("%B %d, %Y, %H:%M:%S"))

          # We then write the personas' movements to a file that will be sent 
          # to the frontend server. 
          # Example json output: 
          # {"persona": {"Maria Lopez": {"movement": [58, 9]}},
          #  "persona": {"Klaus Mueller": {"movement": [38, 12]}}, 
          #  "meta": {curr_time: <datetime>}}
          curr_move_file = f"{sim_folder}/movement/{self.step}.json"
          with open(curr_move_file, "w", encoding="utf-8") as outfile:
            outfile.write(json.dumps(movements, indent=2))

          # Extract non-null chat logs to conversations_extracted.txt (once per conversation per step)
          try:
            chat_log_path = os.path.join(sim_folder, "conversations_extracted.txt")
            seen_pairs = set()
            curr_time_str = movements.get("meta", {}).get("curr_time", "")
            for pname, pdata in movements.get("persona", {}).items():
              chat = pdata.get("chat") if isinstance(pdata, dict) else None
              if not chat or len(chat) < 2:
                continue
              pair = tuple(sorted([chat[0][0], chat[1][0]]))
              if pair in seen_pairs:
                continue
              seen_pairs.add(pair)
              line_sep = "\n"
              block = (
                line_sep + "=" * 60 + line_sep
                + f"Step {self.step}  |  {curr_time_str}" + line_sep
                + f"CONVERSATION: {chat[0][0]} & {chat[1][0]}" + line_sep
                + "=" * 60 + line_sep
                + line_sep.join(f"{s}: {u}" for s, u in chat)
                + line_sep
              )
              with open(chat_log_path, "a", encoding="utf-8") as cf:
                if self.step == 0 and len(seen_pairs) == 1:
                  header = (
                    "EXTRACTED CONVERSATIONS (live run)\n"
                    + "=" * 60 + line_sep
                    + f"Simulation: {self.sim_code}" + line_sep
                    + "=" * 60 + line_sep
                  )
                  cf.write(header)
                cf.write(block)
          except Exception:
            pass  # don't fail the simulation if chat log write fails

          # Update environment manager with new positions for next iteration
          for persona_name, persona_data in movements["persona"].items():
            if "movement" in persona_data:
              next_pos = persona_data["movement"]
              if len(next_pos) >= 2:
                self.env_manager.set_persona_position(persona_name, next_pos[0], next_pos[1])

          # After this cycle, the world takes one step forward, and the 
          # current time moves by <sec_per_step> amount. 
          self.step += 1
          self.curr_time += datetime.timedelta(seconds=self.sec_per_step)

          # Persist state periodically so simulation can be resumed after interrupt
          if self.step % 25 == 0:
            try:
              self.save()
            except Exception:
              pass  # Don't crash the run if save fails

          # Log progress every 100 steps
          if self.step % 100 == 0:
            try:
              simulation_logger.log_info(f"Step {self.step}: Simulation time is {self.curr_time.strftime('%B %d, %Y, %H:%M:%S')}")
            except:
              pass  # Don't crash if logging fails

          int_counter -= 1
          
      # Sleep so we don't burn our machines. 
      time.sleep(self.server_sleep)


  def open_server(self): 
    """
    Open up an interactive terminal prompt that lets you run the simulation 
    step by step and probe agent state. 

    INPUT 
      None
    OUTPUT
      None
    """
    # Start simulation logging
    start_simulation_logging()
    
    print ("Note: The agents in this simulation package are computational")
    print ("constructs powered by generative agents architecture and LLM. We")
    print ("clarify that these agents lack human-like agency, consciousness,")
    print ("and independent decision-making.\n---")

    # <sim_folder> points to the current simulation folder.
    sim_folder = f"{fs_storage}/{self.sim_code}"

    while True: 
      sim_command = input("Enter option: ")
      sim_command = sim_command.strip()
      ret_str = ""

      try: 
        if sim_command.lower() in ["f", "fin", "finish", "save and finish"]: 
          # Finishes the simulation environment and saves the progress. 
          # Example: fin
          simulation_logger.log_info("Saving and finishing simulation")
          self.save()
          stop_simulation_logging()
          break

        elif sim_command.lower() == "start path tester mode": 
          # Starts the path tester and removes the currently forked sim files.
          # Note that once you start this mode, you need to exit out of the
          # session and restart in case you want to run something else. 
          shutil.rmtree(sim_folder) 
          self.start_path_tester_server()

        elif sim_command.lower() == "exit": 
          # Finishes the simulation environment but does not save the progress
          # and erases all saved data from current simulation. 
          # Example: exit 
          simulation_logger.log_warning("Exiting simulation without saving - all data will be lost!")
          shutil.rmtree(sim_folder) 
          stop_simulation_logging()
          break 

        elif sim_command.lower() == "save": 
          # Saves the current simulation progress. 
          # Example: save
          simulation_logger.log_info("Saving simulation progress")
          self.save()

        elif sim_command[:3].lower() == "run": 
          # Runs the number of steps specified in the prompt.
          # Example: run 1000
          int_count = int(sim_command.split()[-1])
          simulation_logger.log_info(f"Starting simulation run for {int_count} steps")
          try:
            self.start_server(int_count)
            simulation_logger.log_info(f"Completed simulation run for {int_count} steps")
          except Exception as e:
            simulation_logger.log_error(f"Simulation run failed: {str(e)}")
            simulation_logger.log_error(f"Traceback: {traceback.format_exc()}")
            raise

        elif ("print persona schedule" 
              in sim_command[:22].lower()): 
          # Print the decomposed schedule of the persona specified in the 
          # prompt.
          # Example: print persona schedule Isabella Rodriguez
          ret_str += (self.personas[" ".join(sim_command.split()[-2:])]
                      .scratch.get_str_daily_schedule_summary())

        elif ("print all persona schedule" 
              in sim_command[:26].lower()): 
          # Print the decomposed schedule of all personas in the world. 
          # Example: print all persona schedule
          for persona_name, persona in self.personas.items(): 
            ret_str += f"{persona_name}\n"
            ret_str += f"{persona.scratch.get_str_daily_schedule_summary()}\n"
            ret_str += f"---\n"

        elif ("print hourly org persona schedule" 
              in sim_command.lower()): 
          # Print the hourly schedule of the persona specified in the prompt.
          # This one shows the original, non-decomposed version of the 
          # schedule.
          # Ex: print persona schedule Isabella Rodriguez
          ret_str += (self.personas[" ".join(sim_command.split()[-2:])]
                      .scratch.get_str_daily_schedule_hourly_org_summary())

        elif ("print persona current tile" 
              in sim_command[:26].lower()): 
          # Print the x y tile coordinate of the persona specified in the 
          # prompt. 
          # Ex: print persona current tile Isabella Rodriguez
          ret_str += str(self.personas[" ".join(sim_command.split()[-2:])]
                      .scratch.curr_tile)

        elif ("print persona chatting with buffer" 
              in sim_command.lower()): 
          # Print the chatting with buffer of the persona specified in the 
          # prompt.
          # Ex: print persona chatting with buffer Isabella Rodriguez
          curr_persona = self.personas[" ".join(sim_command.split()[-2:])]
          for p_n, count in curr_persona.scratch.chatting_with_buffer.items(): 
            ret_str += f"{p_n}: {count}"

        elif ("print persona associative memory (event)" 
              in sim_command.lower()):
          # Print the associative memory (event) of the persona specified in
          # the prompt
          # Ex: print persona associative memory (event) Isabella Rodriguez
          ret_str += f'{self.personas[" ".join(sim_command.split()[-2:])]}\n'
          ret_str += (self.personas[" ".join(sim_command.split()[-2:])]
                                       .a_mem.get_str_seq_events())

        elif ("print persona associative memory (thought)" 
              in sim_command.lower()): 
          # Print the associative memory (thought) of the persona specified in
          # the prompt
          # Ex: print persona associative memory (thought) Isabella Rodriguez
          ret_str += f'{self.personas[" ".join(sim_command.split()[-2:])]}\n'
          ret_str += (self.personas[" ".join(sim_command.split()[-2:])]
                                       .a_mem.get_str_seq_thoughts())

        elif ("print persona associative memory (chat)" 
              in sim_command.lower()): 
          # Print the associative memory (chat) of the persona specified in
          # the prompt
          # Ex: print persona associative memory (chat) Isabella Rodriguez
          ret_str += f'{self.personas[" ".join(sim_command.split()[-2:])]}\n'
          ret_str += (self.personas[" ".join(sim_command.split()[-2:])]
                                       .a_mem.get_str_seq_chats())

        elif ("print persona spatial memory" 
              in sim_command.lower()): 
          # Print the spatial memory of the persona specified in the prompt
          # Ex: print persona spatial memory Isabella Rodriguez
          self.personas[" ".join(sim_command.split()[-2:])].s_mem.print_tree()

        elif ("print current time" 
              in sim_command[:18].lower()): 
          # Print the current time of the world. 
          # Ex: print current time
          ret_str += f'{self.curr_time.strftime("%B %d, %Y, %H:%M:%S")}\n'
          ret_str += f'steps: {self.step}'

        elif ("print tile event" 
              in sim_command[:16].lower()): 
          # Print the tile events in the tile specified in the prompt 
          # Ex: print tile event 50, 30
          cooordinate = [int(i.strip()) for i in sim_command[16:].split(",")]
          for i in self.maze.access_tile(cooordinate)["events"]: 
            ret_str += f"{i}\n"

        elif ("print tile details" 
              in sim_command.lower()): 
          # Print the tile details of the tile specified in the prompt 
          # Ex: print tile event 50, 30
          cooordinate = [int(i.strip()) for i in sim_command[18:].split(",")]
          for key, val in self.maze.access_tile(cooordinate).items(): 
            ret_str += f"{key}: {val}\n"

        elif ("call -- analysis" 
              in sim_command.lower()): 
          # Starts a stateless chat session with the agent. It does not save 
          # anything to the agent's memory. 
          # Ex: call -- analysis Isabella Rodriguez
          persona_name = sim_command[len("call -- analysis"):].strip() 
          self.personas[persona_name].open_convo_session("analysis")

        elif ("call -- load history" 
              in sim_command.lower()): 
          curr_file = maze_assets_loc + "/" + sim_command[len("call -- load history"):].strip() 
          # call -- load history the_ville/agent_history_init_n3.csv

          rows = read_file_to_list(curr_file, header=True, strip_trail=True)[1]
          clean_whispers = []
          for row in rows: 
            agent_name = row[0].strip() 
            whispers = row[1].split(";")
            whispers = [whisper.strip() for whisper in whispers]
            for whisper in whispers: 
              clean_whispers += [[agent_name, whisper]]

          load_history_via_whisper(self.personas, clean_whispers)

        print (ret_str)

      except:
        traceback.print_exc()
        print ("Error.")
        pass


if __name__ == '__main__':
  # rs = ReverieServer("base_the_ville_isabella_maria_klaus", 
  #                    "July1_the_ville_isabella_maria_klaus-step-3-1")
  # rs = ReverieServer("July1_the_ville_isabella_maria_klaus-step-3-20", 
  #                    "July1_the_ville_isabella_maria_klaus-step-3-21")
  # rs.open_server()

  choice = input("Resume existing simulation? (y/n): ").strip().lower()
  if choice == "y" or choice == "yes":
    target = input("Enter the simulation name to resume (e.g. test_4o_mini): ").strip()
    if not target:
      print("No name entered. Exiting.")
      exit(1)
    rs = ReverieServer(None, target, resume=True)
    print(f"Resuming from step {rs.step} (time: {rs.curr_time.strftime('%B %d, %Y, %H:%M:%S')})")
  else:
    origin = input("Enter the name of the forked simulation: ").strip()
    target = input("Enter the name of the new simulation: ").strip()
    rs = ReverieServer(origin, target)

  Create(rs)
  rs.open_server()
