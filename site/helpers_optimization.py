import pandas as pd

def calc_consumption_from_planing(plan,mat,day_consumption):
    for house in plan:
        for machine in plan[house].keys():
            for time_slot in plan[house][machine]:
                day_consumption[time_slot] += mat[machine]["power"]
    return day_consumption

def load_csv(name):
    data_input = pd.read_csv(name)
    return data_input

def make_batches(data_path,data_network_path):
    data_network = load_csv(data_network_path)
    data = load_csv(data_path)

    network_dict = make_network_dict(data_network)

    empty_plannings = generate_empty_planning_batch(data)

    batch_empty_plannings = {}

    for pl in network_dict.keys():
        houses = network_dict[pl]
        batch_empty_plannings[pl] = {}
        
        for house in houses:
            batch_empty_plannings[pl][house] = empty_plannings[house]
    return batch_empty_plannings

def make_network_dict(data_network):
    network_dict = {}

    for row in data_network.values:
        pl,house = row[0].split(";")
        if not pl in network_dict:
            network_dict[pl] = []
            
        network_dict[pl].append(house)
    
    return network_dict

def generate_empty_planning_batch(data_input):
    header = []
    for col in data_input.columns:
        header = col.split(";")

    data_output = {}

    for i in range(len(data_input)):
        row = data_input.values[i][0].split(";")
        for j in range(len(row)):
            if header[j] != "Logement":
                if row[j] == '1':
                    data_output[row[0]][header[j]] = []
            else:
                data_output[row[0]] = {}

    return data_output

