"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling OpenAI APIs.
"""
import json
import random
import openai
import time 

from utils import *

openai.api_key = openai_api_key

BASE_DELAY = 1.0
MAX_DELAY = 60.0

_embedding_cache = {}  # keyed by normalized text; add a lock here when calls go concurrent

def ChatGPT_single_request(prompt):
  completion = openai.ChatCompletion.create(
    model="gpt-4o-mini", 
    messages=[{"role": "user", "content": prompt}]
  )
  return completion["choices"][0]["message"]["content"]


# ============================================================================
# #####################[SECTION 1: CHATGPT-3 STRUCTURE] ######################
# ============================================================================

def GPT4_request(prompt): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  # Add retry logic for API service issues
  max_retries = 3
  for attempt in range(max_retries):
    try:
      completion = openai.ChatCompletion.create(
      model="gpt-4o-mini",
      messages=[{"role": "user", "content": prompt}]
      )
      return completion["choices"][0]["message"]["content"]

    except (openai.error.ServiceUnavailableError, openai.error.RateLimitError) as e:
      if attempt < max_retries - 1:
        wait = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
        print(f"OpenAI API error (attempt {attempt + 1}/{max_retries}). Retrying in {wait}s...")
        # Try to log the error
        try:
          from simulation_logger import simulation_logger
          simulation_logger.log_api_error("ServiceUnavailable", str(e), attempt + 1)
        except ImportError:
          pass
        time.sleep(wait)
        continue
      else:
        print(f"OpenAI API error after {max_retries} attempts. Using fallback response.")
        # Try to log the final failure
        try:
          from simulation_logger import simulation_logger
          simulation_logger.log_api_error("ServiceUnavailable", f"Failed after {max_retries} attempts", max_retries)
        except ImportError:
          pass
        return "ChatGPT ERROR"

    except Exception as e:
      print(f"ChatGPT ERROR: {e}")
      return "ChatGPT ERROR"


def ChatGPT_request(prompt): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  # Add retry logic for API service issues
  max_retries = 3
  for attempt in range(max_retries):
    try:
      completion = openai.ChatCompletion.create(
      model="gpt-4o-mini",
      messages=[{"role": "user", "content": prompt}]
      )
      print(completion)
      return completion["choices"][0]["message"]["content"]

    except (openai.error.ServiceUnavailableError, openai.error.RateLimitError) as e:
      if attempt < max_retries - 1:
        wait = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
        print(f"OpenAI API error (attempt {attempt + 1}/{max_retries}). Retrying in {wait}s...")
        time.sleep(wait)
        continue
      else:
        print(f"OpenAI API error after {max_retries} attempts. Using fallback response.")
        return "ChatGPT ERROR"

    except Exception as e:
      print(f"ChatGPT ERROR: {e}")
      return "ChatGPT ERROR"


def GPT4_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_gpt_response = GPT4_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass

  return False


def ChatGPT_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  # prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = str(json.loads(curr_gpt_response)["output"])

      # print ("---ashdfaf")
      # print (curr_gpt_response)
      # print ("000asdfhia")
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass

  return False


def ChatGPT_safe_generate_response_OLD(prompt, 
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 
    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      if verbose: 
        print (f"---- repeat count: {i}")
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass
  print ("FAIL SAFE TRIGGERED") 
  return fail_safe_response


# ============================================================================
# ###################[SECTION 2: ORIGINAL GPT-3 STRUCTURE] ###################
# ============================================================================

def GPT_request(prompt, gpt_parameter): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  # Add retry logic for API service issues
  max_retries = 3
  for attempt in range(max_retries):
    try:
      response = openai.Completion.create(
                  model=gpt_parameter["engine"],
                  prompt=prompt,
                  temperature=gpt_parameter["temperature"],
                  max_tokens=gpt_parameter["max_tokens"],
                  top_p=gpt_parameter["top_p"],
                  frequency_penalty=gpt_parameter["frequency_penalty"],
                  presence_penalty=gpt_parameter["presence_penalty"],
                  stream=gpt_parameter["stream"],
                  stop=gpt_parameter["stop"],)
      return response.choices[0].text

    except (openai.error.ServiceUnavailableError, openai.error.RateLimitError) as e:
      if attempt < max_retries - 1:
        wait = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
        print(f"OpenAI API error (attempt {attempt + 1}/{max_retries}). Retrying in {wait}s...")
        time.sleep(wait)
        continue
      else:
        print(f"OpenAI API error after {max_retries} attempts. Using fallback response.")
        return "TOKEN LIMIT EXCEEDED"

    except Exception as e:
      print(f"TOKEN LIMIT EXCEEDED: {e}")
      return "TOKEN LIMIT EXCEEDED"


def generate_prompt(curr_input, prompt_lib_file): 
  """
  Takes in the current input (e.g. comment that you want to classifiy) and 
  the path to a prompt file. The prompt file contains the raw str prompt that
  will be used, which contains the following substr: !<INPUT>! -- this 
  function replaces this substr with the actual curr_input to produce the 
  final promopt that will be sent to the GPT3 server. 
  ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the promopt file. 
  RETURNS: 
    a str prompt that will be sent to OpenAI's GPT server.  
  """
  if type(curr_input) == type("string"): 
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  f = open(prompt_lib_file, "r")
  prompt = f.read()
  f.close()
  for count, i in enumerate(curr_input):   
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt: 
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
  return prompt.strip()


def safe_generate_response(prompt, 
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False): 
  if verbose: 
    print (prompt)

  for i in range(repeat): 
    curr_gpt_response = GPT_request(prompt, gpt_parameter)
    if func_validate(curr_gpt_response, prompt=prompt): 
      return func_clean_up(curr_gpt_response, prompt=prompt)
    if verbose: 
      print ("---- repeat count: ", i, curr_gpt_response)
      print (curr_gpt_response)
      print ("~~~~")
  return fail_safe_response


def get_embedding(text, model="text-embedding-ada-002"):
  text = text.replace("\n", " ")
  if not text:
    text = "this is blank"

  if text in _embedding_cache:
    return _embedding_cache[text]

  # Add retry logic for API service issues
  max_retries = 3
  for attempt in range(max_retries):
    try:
      embedding = openai.Embedding.create(
              input=[text], model=model)['data'][0]['embedding']
      # TODO: this write needs a lock once embedding calls run concurrently
      _embedding_cache[text] = embedding
      return embedding
    except (openai.error.ServiceUnavailableError, openai.error.RateLimitError) as e:
      if attempt < max_retries - 1:
        wait = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
        print(f"OpenAI API error (attempt {attempt + 1}/{max_retries}). Retrying in {wait}s...")
        # Try to log the error
        try:
          from simulation_logger import simulation_logger
          simulation_logger.log_api_error("EmbeddingServiceUnavailable", str(e), attempt + 1)
        except ImportError:
          pass
        time.sleep(wait)
        continue
      else:
        print(f"OpenAI API error after {max_retries} attempts. Using fallback embedding.")
        # Try to log the final failure
        try:
          from simulation_logger import simulation_logger
          simulation_logger.log_api_error("EmbeddingServiceUnavailable", f"Failed after {max_retries} attempts", max_retries)
        except ImportError:
          pass
        return [0.0] * 1536  # Standard embedding size for text-embedding-ada-002
    except Exception as e:
      print(f"Unexpected error in get_embedding: {e}")
      return [0.0] * 1536


if __name__ == '__main__':
  gpt_parameter = {"engine": "text-davinci-003", "max_tokens": 50, 
                   "temperature": 0, "top_p": 1, "stream": False,
                   "frequency_penalty": 0, "presence_penalty": 0, 
                   "stop": ['"']}
  curr_input = ["driving to a friend's house"]
  prompt_lib_file = "prompt_template/test_prompt_July5.txt"
  prompt = generate_prompt(curr_input, prompt_lib_file)

  def __func_validate(gpt_response): 
    if len(gpt_response.strip()) <= 1:
      return False
    if len(gpt_response.strip().split(" ")) > 1: 
      return False
    return True
  def __func_clean_up(gpt_response):
    cleaned_response = gpt_response.strip()
    return cleaned_response

  output = safe_generate_response(prompt, 
                                 gpt_parameter,
                                 5,
                                 "rest",
                                 __func_validate,
                                 __func_clean_up,
                                 True)

  print (output)


def GPT4_safe_generate_response_OLD(prompt,
                                    repeat=3,
                                    fail_safe_response="error",
                                    func_validate=None,
                                    func_clean_up=None,
                                    verbose=False):
    if verbose:
        print("CHAT GPT PROMPT")
        print(prompt)

    for i in range(repeat):
        try:
            curr_gpt_response = GPT4_request(prompt)  # .strip()
            if func_validate(curr_gpt_response, prompt=prompt):
                return func_clean_up(curr_gpt_response, prompt=prompt)
            if verbose:
                print(f"---- repeat count: {i}")
                print(curr_gpt_response)
                print("~~~~")

        except Exception as e:
            print("GPT4 request failed:", e)
    print("FAIL SAFE TRIGGERED")
    return fail_safe_response

def GPT4_request_t1(prompt):
    """
    Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
    server and returns the response.
    ARGS:
      prompt: a str prompt
      gpt_parameter: a python dictionary with the keys indicating the names of
                     the parameter and the values indicating the parameter
                     values.
    RETURNS:
      a str of GPT-3's response.
    """

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=1
        )
        return completion["choices"][0]["message"]["content"]

    except:
        print("ChatGPT ERROR")
        return "ChatGPT ERROR"

def GPT4_safe_generate_response_OLD_t1(prompt,
                                    repeat=3,
                                    fail_safe_response="error",
                                    func_validate=None,
                                    func_clean_up=None,
                                    verbose=False):
    if verbose:
        print("CHAT GPT PROMPT")
        print(prompt)

    for i in range(repeat):
        try:
            curr_gpt_response = GPT4_request_t1(prompt)  # .strip()
            if func_validate(curr_gpt_response, prompt=prompt):
                return func_clean_up(curr_gpt_response, prompt=prompt)
            if verbose:
                print(f"---- repeat count: {i}")
                print(curr_gpt_response)
                print("~~~~")

        except:
            pass
    print("FAIL SAFE TRIGGERED")
    return fail_safe_response

def ChatGPT_request_t0(prompt):
    """
    Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
    server and returns the response.
    ARGS:
      prompt: a str prompt
      gpt_parameter: a python dictionary with the keys indicating the names of
                     the parameter and the values indicating the parameter
                     values.
    RETURNS:
      a str of GPT-3's response.
    """
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        print(completion)
        return completion["choices"][0]["message"]["content"]

    except:
        print("ChatGPT ERROR")
        return "ChatGPT ERROR"

def ChatGPT_safe_generate_response_OLD_t0(prompt,
                                          repeat=3,
                                          fail_safe_response="error",
                                          func_validate=None,
                                          func_clean_up=None,
                                          verbose=False):
    if verbose:
        print("CHAT GPT PROMPT")
        print(prompt)

    for i in range(repeat):
        try:
            curr_gpt_response = ChatGPT_request_t0(prompt)  # .strip()
            if func_validate(curr_gpt_response, prompt=prompt):
                return func_clean_up(curr_gpt_response, prompt=prompt)
            if verbose:
                print(f"---- repeat count: {i}")
                print(curr_gpt_response)
                print("~~~~")

        except:
            pass
    print("FAIL SAFE TRIGGERED")
    return fail_safe_response


















