import glob
from utils import *


def generate_mainloop(agent, problem_name):
    folder = rf"./exps/simulation/{problem_name}"
    name = "".join(problem_name.split("_")[1:])

    model_name = "./bge-large-en-v1.5"
    documents_list = [
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
            You are a highly capable AI engineering agent responsible for analysing the problem and generate the mainloop FreeFEM code. 
            You only generate very clean code. You don't generate any other things. Your code is directly executable.
            You work according to the templates provided, the guidance provided, and you include the code provided directly in your code.
        """),
        ("user",
         """
            You analyse the problem provided: {problem}. You consider the training settings provided: {training}.
            {question}. You use the information provided here: {context}. You make sure the code is clean enough. 
            You generate the main code to solve the problem according to the template, guidance and other information provided.
            The equations to be included in the code are with the names: {equation_names}.

            Please take special notice to some FreeFEM grammar rules: {mainloop_rules}.
        """),
    ]

    with open(rf"{folder}/problem_description.txt", "r") as file:
        problem = file.read()
    with open(rf"{folder}/training_description.txt", "r") as file:
        training = file.read()
    with open(rf"./database/simulation/mainloop_rules.txt") as file:
        mainloop_rules = file.read()
    agent.set_prompt_list(prompt_list)
    response = agent.invoke({"question": "Please work according to the demand and the description."},
                            partial_dict={
                                "problem": problem,
                                "training": training,
                                "name": name,
                                "equation_names": equation_names,
                                "mainloop_rules": mainloop_rules,
                            })

    write_to_file(rf"{folder}/{problem_name}.edp", response)

    print("[OK] Main Loop Generation is done.")
    return response


if __name__ == "__main__":
    from chatgpt_api import ChatGPTCodeAgent

    agent = ChatGPTCodeAgent()
    response = generate_mainloop(agent, "###")
    print(response)
