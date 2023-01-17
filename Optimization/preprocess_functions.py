import pandas as pd

def load_csv(name):
    data_input = pd.read_csv(name)
    return data_input


def generate_empty_planning(data_input):

    header = []
    for col in data_input.columns:
        header = col.split(";")
    print(header)

    data_output = []

    for i in range(len(data_input)):
        data_output.append({})
        row = data_input.values[i][0].split(";")
        for j in range(len(row)):
            if header[j] != "Logement":
                data_output[i][header[j]] = []

    return data_output
