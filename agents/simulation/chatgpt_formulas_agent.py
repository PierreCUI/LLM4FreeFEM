from utils import *


def generate_formulas(agent, problem_name):
    folder = rf"./exps/simulation/{problem_name}"
    name = "".join(problem_name.split("_")[1:])

    model_name = "./bge-large-en-v1.5"
    documents_list = [
        rf"./database/simulation/equation_template.txt",
        rf"./database/simulation/equation_guide.txt",
    ]
    search_type = "mmr"
    search_kwargs = {"k": 1024}
    agent.set_retriever_from_documents(model_name, documents_list, search_type, search_kwargs)

    prompt_list = [
        ("system",
         """
            You are a highly capable AI engineering agent responsible for analysing the problem and write equations for FreeFEM. 
            You response in json form. You only response the required key and the corresponding code as value. 
            You give very clean response. You response without anything else. Your code is very clean so that it can be used directly later.
        """),
        ("user",
         """
            You analyse the problem provided: {problem}. {question}
            You response:
            0. Key0: reasoning, Value0: The reasoning process in detail, including all the equations and processes.
            1. Key1: {name}Newton, Value1: A code for the Jacobian equation of the problem in integration form.
            2. Key2: {name}Residual, Value2: A code for the Residual equation of the problem in integration form.
            
            Please take special notice to some FreeFEM grammar rules: {formula_rules}.
        """),
    ]

    with open(rf"{folder}/problem_description.txt", "r") as file:
        problem = file.read()
    with open(rf"./database/simulation/formula_rules.txt", "r") as file:
        formula_rules = file.read()
    agent.set_prompt_list(prompt_list)
    response = agent.invoke({"question": "Please work according to the demands and the descriptions."},
                            partial_dict={"problem": problem, "name": name, "formula_rules": formula_rules})
    response_dict = to_json_obj(response)

    write_to_file(rf"{folder}/include/{name}Newton.edp", response_dict[rf"{name}Newton"])
    write_to_file(rf"{folder}/include/{name}Residual.edp", response_dict[rf"{name}Residual"])

    print("[OK] Formulas generation process is done.")
    return response


if __name__ == "__main__":
    from chatgpt_api import ChatGPTCodeAgent

    agent = ChatGPTCodeAgent()
    response = generate_formulas(agent, "###")
    print(response)
