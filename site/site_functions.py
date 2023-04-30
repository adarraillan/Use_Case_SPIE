from preprocess_functions import *
import matplotlib.pyplot as plt
import joblib
from Individual import Individual

def get_machines_of_house(house_id):
    data = load_csv("data/data.csv")
    return_dict = {}
    house = None

    header = list(list(data.columns)[0].split(";"))

    for row in data.values:
        row_list = list(row[0].split(";"))
        if row_list[0] == house_id:
            house = row_list
            break
        
    if house:
        for i in range(len(header)):
            if i != 0:
                return_dict[header[i]] = house[i]
        return return_dict
    else:
        return {"error":"Aucune maison avec cet id"}

def modif_house(house_dict):
    path_data = "data/data.csv"
    data = load_csv(path_data)

    header = list(list(data.columns)[0].split(";"))

    for i in range(len(data.values)):
        row = list(data.values[i][0].split(";"))
        if row[0] == house_dict["house_name"]:
            for j in range(1,len(header)):
                key = header[j]
                if key in house_dict:
                    row[j] = '1'
                else:
                    row[j] = '0'
            row = ';'.join(row)
            data.values[i][0] = row


    csv_data = ';'.join(header)+'\n'
    for i in range(len(data.values)):
            csv_data += str(data.values[i][0]) + '\n'

    with open(path_data,'w') as file:
        file.write(csv_data)

def plot_planning(planning,title):
    m_count = 0
    plt.figure(figsize=(13,8))
    plt.title(title)
   
    ax= plt.subplot()
    ax.xaxis.tick_top()
    ax.grid(visible=True, color='grey',
        linestyle='-.', linewidth=0.5,
        alpha=0.2)
    ax.set_facecolor('white')
    x = range(48)
    ax.set_xticks(x)
    ax.set_xticklabels(["00h", "", "1h", "", "2h", "", "3h", "", "4h", "", "5h", "", "6h", "", "7h", "",
                        "8h", "", "9h", "", "10h", "", "11h", "", "12h", "", "13h", "", "14h", "", "15h", "",
                        "16h", "", "17h", "", "18h", "", "19h", "", "20h", "", "21h", "", "22h", "", "23h", ""],
                        rotation=45)
    ax.tick_params(axis="both", direction="in", pad=15)
   
    for machine in planning.keys():
        m_count += 1
        x = []
        y = []
        for time in planning[machine]:
            x.append(time)
            y.append(machine)
       
        plt.barh(y,x[-1]-x[0]+1, height=0.4, left=x[0])

    path = "static/images/"+title+".png"
    plt.savefig(path)
    #plt.close()
    return path

def get_init_cost_random():
    data = joblib.load("data/Solution_multibatch_save.joblib")

    sum_init = 0
    sum_recuit = 0
    for sols in data:
        sum_init += max(sols[1].day_consumption)
        sum_recuit += max(sols[0].day_consumption)

    return (sum_init,sum_recuit)

