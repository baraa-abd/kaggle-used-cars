import google.generativeai as genai
import os
import pandas
import re   #regular expressions, to clean car name to be used as a key
import random
import time
import seaborn as sns
from statistics import median, mode, mean

#files_to_process = ['train_short']#["train", "test"]
only_alphanum = re.compile(r'[\W_]+')  #regular expression pattern that matches any non-alphanumeric characters
only_num = re.compile(r'[\D]+') #regular expression pattern that matches any non-numeric characters
#known_cars = pandas.read_csv("known_cars_handmade.csv")
#genai.configure(api_key = os.environ["GOOGLE_API_KEY"])

#possible_body_types = set(["sedan", "hatchback", "van", "crossover", "wagon", "suv", "pick-up truck", "sport"])
#batch_size = 25

#prompt_params = {"min_num":4, "max_num":8, "repeats":False, "min_len":3, "max_len":10, "balance":True, "type_size":None}
"""
min_num and max_num set bounds on number of examples per prompt
type_size: desired number of cars of each type in balanced cars DataFrame (does nothing if balance is False)
min_len, max_len: integers indicating min/max length of an example
repeats: boolean indicating whether a car can appear more than once in an example
balance: boolean indicating whether the cars DataFrame will be balanced for body types before sampling the examples
Note: even if repeats is False if balance is True and there are too few examples of some body type, we can still get repeats in examples."""



def get_car_keys_names(dfs):
    car_names = dict()
    for df in dfs:
        for i in range(len(df)):
            car_fullname = str(df["model_year"][i])+" "+df["brand"][i]+" "+df["model"][i]
            car_key = only_alphanum.sub("", car_fullname).lower()
            if car_key not in car_names: car_names[car_key] = car_fullname          #we are choosing the spelling of the first time we see a car's full name
    print("Total unique specific car models = ", len(car_names))
    return car_names

def balance_df(df, col, type_size = None):
    """
    Balances DataFrame df according to the (categorical) column col.
    type_size can determine desired number of rows with each possible value in col
    """
    sub_dfs = [df[df[col]==type] for type in df[col].unique()]
    count = int(median([df.shape[0] for df in sub_dfs])) if type_size == None else type_size
    new_df = pandas.concat([df.sample(n=count, axis = 0, replace = count>df.shape[0]) for df in sub_dfs if df.shape[0]>0])
    new_df.reset_index(inplace = True, drop = True)
    return new_df

def generate_prompt_examples(cars, n, prompt_params, body_types):
    """
    cars: DataFrame with columns full_name, body, msrp
    n: integer indicating number of examples in the prompt
    body_types: list of possible categories of body types
    returns a list of n random lists of length at most max_length consisting of a prompt and the expected answer
    """
    if n>0: assert len(cars)>0
    max_length = len(cars) if prompt_params["max_len"] == None else prompt_params["max_len"]
    if prompt_params["balance"]: cars = balance_df(cars, 'body', type_size = prompt_params["type_size"])
    examples = []
    for i in range(n):
        m = random.randint(prompt_params["min_len"],max_length) if prompt_params["repeats"] else random.randint(1,min(max_length, len(cars)))
        chosen_cars = cars.sample(n = m, axis = 0, replace = prompt_params["repeats"]).reset_index(drop = True)
        answer = "".join([f(chosen_cars.loc[i, :]) for i in chosen_cars.index for f in (lambda x:x["body"], lambda x:", "+str(x["msrp"]), lambda x:"\n")])
        example = [[chosen_cars['full_name'][i] for i in chosen_cars.index], answer[:-1]]     #remove the last \n in answer before building example
        examples.append(example)
    return examples

def get_prompt(car_list, prompt_cars, prompt_params, body_types):
    number_of_prompt_examples = random.randint(prompt_params["min_num"], prompt_params["max_num"])
    prompt_examples = generate_prompt_examples(prompt_cars, number_of_prompt_examples, prompt_params, body_types)
    prompt_examples_string = "".join([f"Example {i+1}: Given {example[0]} of length {len(example[0])}, your answer is the following between the single quotes: \n'{example[1]}' \n" for i,example in enumerate(prompt_examples)])
    prompt = (f"Consider the following set of car body types with no particular order: {body_types}.\n"
               "Given a list of car models, choose the best matching of the above types (and only of these types) for each car model,"
               " find the starting MSRP for each car model, and then give your answer for each as 'BODY_TYPE, MSRP' in a separate line with no other words."
               " Make sure you give an answer for each car.\n\n"
               f"{prompt_examples_string}"
               f" Now, give your answer for the following list, of length {len(car_list)}, as described above: {car_list}"
             )
    return prompt

def create_finetuning_examples(known_cars, possible_body_types, prompt_params, batch_size, n):
    """Generates training data for fine-tuning the model given known_cars, a DataFrame containing a car name, body type, msrp
       - possible_body_types is a list
       - prompt_params is a dictionary of parameters for creating the prompts
       - n is an integer indicating the number of examples in the training data
       Returns a list of dictionaries of the form '{"text_input":prompt, "output": answer}' """
    training_data = []
    total_known = len(known_cars)
    if prompt_params["balance"]:
        known_cars = balance_df(known_cars, 'body', type_size = prompt_params['type_size'])
        prompt_params = prompt_params.copy()
        prompt_params["balance"] = False
    for i in range(n):
        prompt_cars = known_cars.sample(n=batch_size, axis = 0, replace = batch_size>total_known).reset_index(drop = True)
        prompt_car_list = [prompt_cars['full_name'][i] for i in prompt_cars.index]
        answer = "".join([f(prompt_cars.loc[i, :]) for i in prompt_cars.index for f in (lambda x:x["body"], lambda x:", "+str(x["msrp"]), lambda x:"\n")])
        prompt = get_prompt(prompt_car_list, known_cars, prompt_params, possible_body_types)
        training_data.append({"text_input":prompt, "output": answer[:-1]})
    return training_data

def random_permutation(n, at = 0, remaining = None, s = None, s_inv = None):
    """returns two dictionaries s, s^-1 that represent a permutation s on n and its inverse"""
    if s==None: s= dict()
    if s_inv==None: s_inv = dict()
    if at == n: return [s, s_inv]
    if remaining == None: remaining = list(range(n))
    to = random.choice(remaining)
    to_ind = remaining.index(to)
    s[at] = to
    s_inv[to] = at
    return random_permutation(n, at = at+1, remaining = remaining[:to_ind]+remaining[to_ind+1:], s=s, s_inv=s_inv)

def enrich_using_model(model, car_names, prompt_cars, batch_size, possible_body_types, prompt_params,  last = None, restricted = False, rounds = 1, daily_request_limit = 1500):
    car_keys = list(car_names.keys())
    total_number = len(car_keys) if last == None else last
    print((f"Using {model}:\n"
           f"Parameters:\nBatch size = {batch_size}\nTotal number of car models = {total_number}\n"
           f"Min # of examples in each prompt = {prompt_params['min_num']}      Max # of examples in each prompt = {prompt_params['max_num']}\n"
           f"Min length of examples in each prompt = {prompt_params['min_len']}      Max length of examples in each prompt = {prompt_params['max_len']}\n"
           f"Allow repeats in each prompt example = {prompt_params['repeats']}       Number of times each batch is asked about = {rounds}\n"
        ))
    car_body_types = dict()
    car_msrps = dict()
    daily_requests_made = 0
    for i in range(total_number//batch_size+1):
        start = total_number-batch_size if i==total_number//batch_size else i*batch_size
        car_list = [car_names[car_keys[start+j]] for j in range(batch_size)]
        
        okay = False
        while not okay:
            raw_answers_total = []
            try:
                for j in range(rounds):
                    num_answers = 0
                    while num_answers != len(car_list):
                        if daily_requests_made>= daily_request_limit:
                            daily_requests_made = 0
                            time.sleep(24*60*3600)
                        s, s_inv = random_permutation(len(car_list))
                        daily_requests_made+=1
                        prompt = get_prompt([car_list[s[k]] for k in range(len(car_list))], prompt_cars, prompt_params, possible_body_types)
                        response = model.generate_content(prompt)
                        raw_answers = response.text.split("\n")
                        num_answers = len(raw_answers)
                        time.sleep(2.5)
                        print(f"num answers = {num_answers}")
                    raw_answers = [raw_answers[s_inv[k]] for k in range(len(raw_answers))]
                    raw_answers_total.append(raw_answers)
                split_answers_total = [[raw_answers_total[j][k].split(", ") for j in range(rounds)] for k in range(batch_size)]
                for j in range(batch_size):
                    body_answer = mode([only_alphanum.sub("", split_answers_total[j][k][0]).lower() for k in range(rounds)])
                    msrp_answer = round(median([int(only_num.sub("", split_answers_total[j][k][1])) for k in range(rounds)]))
                    car_key = car_keys[start+j]
                    car_body_types[car_key] = body_answer
                    car_msrps[car_key] = msrp_answer
                okay = True
            except:
                print(f"Problem in batch {i}. Trying again...")
        print(start+batch_size)
    return (car_body_types, car_msrps)

def update_car_type_msrp(dfs, car_body_types, car_msrps, write = False, file_name = None):
    if write: assert isinstance(file_name,str)
    for i, df in enumerate(dfs):
        df['full_name'] = df['model_year'].astype(str)+df['brand']+df['model']
        df['brand_model'] = df['brand']+df['model']
        df['full_name'] = df['full_name'].map(lambda x: only_alphanum.sub("", x).lower())
        df['brand_model'] = df['brand_model'].map(lambda x: only_alphanum.sub("", x).lower())
        df['body_style'] = df['full_name'].map(car_body_types)
        df['msrp'] = df['full_name'].map(car_msrps)
        if write: df.to_csv(file_name+str(i)+"_enriched.csv", index=False)

#update_car_type_msrp(files_to_process)
#print(car_body_types)
#print(car_msrps)
#model = genai.GenerativeModel("gemini-1.0-pro", generation_config = {"temperature":0.1})
#car_names = get_car_keys_names(files_to_process)
#print(augment_using_model(model, car_names, known_cars, 25, possible_body_types, prompt_params, last = 25, rounds = 5))
#print(create_finetuning_examples(known_cars, possible_body_types, prompt_params, 25, 1))
#df = pandas.DataFrame(create_finetuning_examples(known_cars, possible_body_types, prompt_params, 25, 50))
#df.to_csv("fine_tuning_test1.csv", index=False)
