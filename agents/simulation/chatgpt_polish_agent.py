import glob
from utils import *


def generate_polish(agent, problem_name, result_output):
    folder = rf"./exps/simulation/{problem_name}"
    name = "".join(problem_name.split("_")[1:])

    model_name = "./bge-large-en-v1.5"
    documents_list = [
        rf"./database/simulation/equation_template.txt",
        rf"./database/simulation/equation_guide.txt",
        rf"./database/simulation/mainloop_template.txt",
        rf"./database/simulation/mesh_buildup_guide.txt",
        rf"./database/simulation/fespace_buildup_guide.txt",
    ]
    equation_paths = glob.glob(rf"{folder}/include/*.edp")
    documents_list.extend(equation_paths)
    equation_names = ["./include/" + os.path.basename(p) for p in equation_paths]
    search_type = "mmr"
    search_kwargs = {"k": 1024}
    agent.set_retriever_from_documents(model_name, documents_list, search_type, search_kwargs)

    prompt_list = [
        ("system",
         """
            You are a highly capable AI engineering agent responsible for reviewing the result of the output and correcting the mistakes if necessary. 
            You only generate very clean code for the part(s) that need(s) to be corrected.
            You generate the full version of the code that needs to be corrected, instead of a piece. Your code is directly executable.
            You work according to the retrievers provided and the output provided. You response in pure json form.
        """),
        ("user",
         """
            You analyse the problem provided: {problem}. You consider the training settings provided: {training}. {question}. 
            You use the information provided here: {context}. You review the result obtained before: {result_output}.
            You consider as well the rules used when the codes of equations are generated: {formula_rules}.
            You consider as well the rules used when the mainloop code is generated: {mainloop_rules}.
            
            You response in the following structure:
                1. key1: {name}Newton, value1: The correct code of Newton.
                2. key2: {name}Residual, value2: The correct code of Residual.
                3. key3: {name}, value3: The correct code of mainloop.
                You analyse if the result obtained is correct or not. If the result is correct, you don't generate anything (give an empty string).
                If the result is wrong, you locate the problem to the corresponding code or codes. You rewrite the code or codes to correct the mistake.
                If one code is not needed to be modified, you don't generate that key.
            
            Please take special notice to some FreeFEM grammar rules:
                1. The code in the result obtained is after "including" process. In FreeFEM mainloop code, however, you should include the equations.
                The equations are located in "./include/{name}Newton.edp" and "./include/{name}Residual.edp".
                It does not align with the standard programming rule to write all the equations directly in mainloop.
        """),
    ]

    with open(rf"{folder}/problem_description.txt", "r") as file:
        problem = file.read()
    with open(rf"{folder}/training_description.txt", "r") as file:
        training = file.read()
    with open(rf"./database/simulation/formula_rules.txt", "r") as file:
        formula_rules = file.read()
    with open(rf"./database/simulation/mainloop_rules.txt") as file:
        mainloop_rules = file.read()
    agent.set_prompt_list(prompt_list)
    response = agent.invoke({"question": "Please work according to the demand and the description."},
                            partial_dict={
                                "problem": problem,
                                "training": training,
                                "result_output": result_output,
                                "name": name,
                                "equation_names": equation_names,
                                "formula_rules": formula_rules,
                                "mainloop_rules": mainloop_rules,
                            })
    if response:
        response_json = to_json_obj(response)
        for key, value in response_json.items():
            if key == name:
                write_to_file(rf"{folder}/{problem_name}.edp", value)
            else:
                write_to_file(rf"{folder}/include/{key}.edp", value)
