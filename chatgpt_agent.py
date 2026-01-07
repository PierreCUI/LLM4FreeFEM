from pathlib import Path
import time
import subprocess

from chatgpt_api import ChatGPTCodeAgent
from utils import *


def problem_classification(agent, folder):
    prompt_list = [
        ("system",
         """
            You are a highly capable AI engineering agent responsible for analysing the problem and give judgement(s) in the given structure. 
            You response in json form. You only response the required key and value. You give very clean response. You response without anything else.
        """),
        ("user",
         """
            You analyse the problem provided: {problem}. {question}
            You response:
            1. Key1: ProblemType, Value1: Choose within [simulation, shapeOptimization]
        """),
    ]

    with open(rf"{folder}/problem_description.txt", "r") as file:
        problem = file.read()
    agent.set_prompt_list(prompt_list)
    response = agent({"question": "Please work according to the demands and the descriptions."},
                     partial_dict={"problem": problem})
    response_dict = to_json_obj(response)
    return response_dict


def generate_formulas_interface(agent, problem_type, problem_name):
    if problem_type == "simulation":
        from agents.simulation.chatgpt_formulas_agent import generate_formulas
        generate_formulas(agent, problem_name)
    elif problem_type == "shapeOptimization":
        ...
    else:
        raise TypeError("Invalid Problem Type.")


def generate_mainloop_interface(agent, problem_type, problem_name):
    if problem_type == "simulation":
        from agents.simulation.chatgpt_mainloop_agent import generate_mainloop
        generate_mainloop(agent, problem_name)
    elif problem_type == "shapeOptimization":
        ...
    else:
        raise TypeError("Invalid Problem Type.")


def generate_polish_interface(agent, problem_type, problem_name, result_output):
    if problem_type == "simulation":
        from agents.simulation.chatgpt_polish_agent import generate_polish
        generate_polish(agent, problem_name, result_output)
    elif problem_type == "shapeOptimization":
        ...
    else:
        raise TypeError("Invalid Problem Type.")


agent = ChatGPTCodeAgent()
max_index = 10

problems_folder = Path(rf"./exps/simulation")  # Debug
problems_name = [f.name for f in problems_folder.iterdir() if f.is_dir()]

for problem_name in problems_name:
    print(rf"{problem_name} started.")

    folder = rf"./exps/simulation/{problem_name}"  # Debug
    os.makedirs(rf"{folder}/include", exist_ok=True)
    os.makedirs(rf"{folder}/results", exist_ok=True)
    os.makedirs(rf"{folder}/tex", exist_ok=True)
    os.makedirs(rf"{folder}/tmp", exist_ok=True)

    # problem_type = problem_classification(agent, folder)["ProblemType"]
    # time.sleep(1)

    generate_formulas_interface(agent, "simulation", problem_name)  # Debug
    time.sleep(1)

    generate_mainloop_interface(agent, "simulation", problem_name)  # Debug
    time.sleep(1)

    index = 0
    while True:
        index += 1
        result_output = subprocess.run(
            ["FreeFem++", rf"{problem_name}.edp"],
            cwd=folder,
            capture_output=True,
            text=True,
        )
        if "Normal End" not in result_output.stdout:
            if index < max_index and index % 2 == 1:
                generate_polish_interface(agent, "simulation", problem_name, result_output.stdout)  # Debug
                time.sleep(1)
            elif index < max_index and index % 2 == 0:
                generate_formulas_interface(agent, "simulation", problem_name)  # Debug
                time.sleep(1)
                generate_mainloop_interface(agent, "simulation", problem_name)  # Debug
                time.sleep(1)
            else:
                print(rf"ERROR: {problem_name} is not accomplished within the given times!")
                break
        else:
            print(f"[OK] {problem_name} is done. \n")
            break
