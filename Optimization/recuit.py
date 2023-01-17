# exp function
import math
# randomint function
import random

from preprocess_functions import *
from Individual import Individual
import copy
import joblib

def calc_consumption_from_planing(plan,mat,day_consumption):
    for house in plan:
        for machine in house.keys():
            for time_slot in house[machine]:
                day_consumption[time_slot] += mat[machine]["power"]
    return day_consumption


# return
def initial_solution(empty_plannings,mat):

    print("PHASE 1")
    for house in range(len(empty_plannings)):
        for machine in empty_plannings[house].keys():
            if type(mat[machine]["nb_time_slot"]) == list:
                mat[machine]["nb_time_slot"] = random.sample(mat[machine]["nb_time_slot"],1)[0]
            
            if machine in ['LL','SL','LV']:
                start_index = random.randint(0,47)
                for i in range(mat[machine]["nb_time_slot"]):
                    empty_planning[house][machine].append((start_index+i)%48)
            else:
                if mat[machine]["is_sequencable"]:
                    for i in range(mat[machine]["nb_time_slot"]):
                        index = random.randint(0,15)
                        while index in empty_planning[house][machine]:
                            index = random.randint(0,15)
                else:
                    start_index = random.randint(0,15-mat[machine]["nb_time_slot"])
                    for i in range(mat[machine]["nb_time_slot"]):
                        empty_planning[house][machine].append(start_index+i)
                

                

    print("PHASE 2")
    day_consumption = []
    for i in range(48):
        day_consumption.append(0)

    print("PHASE 3")
    day_consumption = calc_consumption_from_planing(empty_planning,mat,day_consumption)

    print("PHASE 4")
    individual = Individual({"start":0,"end":15},empty_planning,day_consumption)
    

    return individual

# param solution : list : the solution we want the neighbour of
# return neighbour_solution : list : the neighbour solution
def get_neighbour_solution(solution: Individual):
    solution_neigh = copy.deepcopy(solution)
    solution_neigh.mutate()

    return solution_neigh


# param neighbour_solution : list : hamiltonian cycle
# param current_solution : list : hamiltonian cycle
# param temperature : float : current temperature
# return double : metropolis criteria
def metropolis(neighbour_solution, current_solution,temperature):
    return math.exp( -( abs( cost_function(current_solution) - cost_function(neighbour_solution) ) / temperature ) )

# param solution : list : hamiltonian cycle
# return cost : int : the cost of solution
def cost_function(solution : Individual):
    return solution.getMax()

# param temperature : float
# return updated temperature : float
def update_temperature(temperature):
    return temperature * 0.99

# param temperature : float : initial temperature
# return best found solution : list int
def recuit_simule(temperature,current_solution):
    best_current_solution = copy.deepcopy(current_solution)
    print_count = 0
    count_iteration_without_improvment = 0

    while count_iteration_without_improvment < 1000:
        print_count += 1
        if print_count%10 == 0:
            print("\r",end="")
            print("**Recuit",print_count,count_iteration_without_improvment,end="                ")

        neighbour_solution = get_neighbour_solution(current_solution)

        if cost_function(neighbour_solution) < cost_function(current_solution) or random.random() < metropolis( neighbour_solution , current_solution , temperature):
            current_solution = neighbour_solution

        if cost_function(current_solution) < cost_function(best_current_solution):
            best_current_solution = copy.deepcopy(current_solution)
            count_iteration_without_improvment = 0
        else:
            count_iteration_without_improvment += 1

        temperature = update_temperature(temperature)
    
    return best_current_solution


matrice_type_puissance_par_demie_heure = {'LV':{"power":float(0.65),"nb_time_slot":2,"is_sequencable":False}, 
                                            'LL':{"power":float(1),"nb_time_slot":2,"is_sequencable":False}, 
                                            'SL':{"power":float(0.125),"nb_time_slot":8,"is_sequencable":False}, 
                                            'TV':{"power":float(0.05),"nb_time_slot":[1,2,3,4,5],"is_sequencable":False}, 
                                            'FG_1':{"power":float(0.1),"nb_time_slot":4,"is_sequencable":False}, 
                                            'CE_1':{"power":float(0.18),"nb_time_slot":12,"is_sequencable":False}, 
                                            'CG':{"power":float(0.1),"nb_time_slot":4,"is_sequencable":False}, 
                                            'FO':{"power":float(0.8),"nb_time_slot":[1,2],"is_sequencable":False}, 
                                            'PL':{"power":float(0.6),"nb_time_slot":[1,2],"is_sequencable":False}, 
                                            'FG_2':{"power":float(0.3),"nb_time_slot":4,"is_sequencable":False}, 
                                            'CE_2':{"power":float(0.25),"nb_time_slot":12,"is_sequencable":False}
                                            }

data = load_csv("data.csv")
empty_planning = generate_empty_planning(data)
empty_planning = empty_planning[:100]


print("Init Solution")
current_solution = initial_solution(empty_planning,matrice_type_puissance_par_demie_heure)
print("Score init",current_solution.getMax())

print("First iter")
best_found_solution = recuit_simule(1000, current_solution)

curr_best = cost_function(best_found_solution)

print("Enter While")
n=0
while  n < 3:
    if cost_function(best_found_solution) < curr_best:
        curr_best = cost_function(best_found_solution)
        print(best_found_solution.day_consumption)
        print(cost_function(best_found_solution))
    

    if n%1 == 0:
        print("iter n° :",n)
    
    n+=1
    best_found_solution = recuit_simule(1000,best_found_solution)

print(best_found_solution.day_consumption)
print(cost_function(best_found_solution))

joblib.dump(best_found_solution.plannings,"Plannings.joblib")
joblib.dump(best_found_solution.day_consumption,"Day_Consumption.joblib")
joblib.dump(best_found_solution,"Solution.joblib")