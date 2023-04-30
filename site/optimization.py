from helpers_optimization import *

def get_start_and_dir(power_indic,nb_time_slot,limit_up,limit_down):
    power_indic_sorted = power_indic.copy()
    power_indic_sorted.sort()
    direction = 1

    for curr_min in power_indic_sorted:
        index = power_indic.index(curr_min)
        if index >= limit_down and index <= limit_up:
            if index+nb_time_slot-1 <= limit_up:
                direction = 1
            else:
                direction = -1
            break

    return [index,direction]

def not_so_stupid_solution(empty_plannings,mat,fixed_machines,list_machines_HP):
    HC = {"start":0,"end":16}

    power_indic = []
    for i in range(48):
        power_indic.append(0)
    calc_consumption_from_planing(empty_plannings,mat,power_indic)

    for house in empty_plannings:
        # print("HOUSE",house)
        machine_list = []
        for machine in empty_plannings[house].keys():
            if not machine in fixed_machines:
                machine_list.append([machine,mat[machine]["nb_time_slot"]])

        machine_list.sort(key=lambda x: x[1],reverse=True)

        for machine in machine_list:
            nb_time_slot = mat[machine[0]]["nb_time_slot"]
            if machine[0] in list_machines_HP:
                [start,direction] = get_start_and_dir(power_indic,nb_time_slot,47,16)
            else:
                [start,direction] = get_start_and_dir(power_indic,nb_time_slot,15,0)

            for i in range(nb_time_slot):
                power_indic[start+(i*direction)] += mat[machine[0]]["power"]
                empty_plannings[house][machine[0]].append(start+(i*direction))
            empty_plannings[house][machine[0]].sort()
        # print("EP",empty_plannings[house])
        # print("PI",power_indic)
                
    day_consumption = []
    for i in range(48):
        day_consumption.append(int(0))

    day_consumption = calc_consumption_from_planing(empty_plannings,mat,day_consumption)
    
    return [empty_plannings,day_consumption]

def init_plannings(batch_empty_plannings,fixed_machines):
    for batch_key in batch_empty_plannings:
        for house in batch_empty_plannings[batch_key]:
            planning = batch_empty_plannings[batch_key][house]
            for machine in planning:
                if machine in fixed_machines.keys():
                    [start,end] = fixed_machines[machine]
                    for j in range(end-start):
                        batch_empty_plannings[batch_key][house][machine].append(start+j)
    
    return batch_empty_plannings

def get_mat_power(PR):
    matrice_machine = {'LV':{"power":float(0.65),"nb_time_slot":2,"is_sequencable":False}, 
                    'LL':{"power":float(1),"nb_time_slot":2,"is_sequencable":False}, 
                    'SL':{"power":float(0.125),"nb_time_slot":8,"is_sequencable":False}, 
                    'TV':{"power":float(0.05*PR),"nb_time_slot":5,"is_sequencable":False}, 
                    'FG_1':{"power":float(0.1),"nb_time_slot":4,"is_sequencable":False}, 
                    'CE_1':{"power":float(0.18),"nb_time_slot":12,"is_sequencable":False}, 
                    'CG':{"power":float(0.1),"nb_time_slot":4,"is_sequencable":False}, 
                    'FO':{"power":float(0.8*PR),"nb_time_slot":2,"is_sequencable":False}, 
                    'PL':{"power":float(0.6*PR),"nb_time_slot":2,"is_sequencable":False}, 
                    'FG_2':{"power":float(0.3),"nb_time_slot":4,"is_sequencable":False}, 
                    'CE_2':{"power":float(0.25),"nb_time_slot":12,"is_sequencable":False}
                    }
    return matrice_machine

def get_fixed_machines():
    # fixed_machines = {'TV':[36,47],'PL':[38,42],'FO':[38,42]}
    fixed_machines = {}

    return fixed_machines

def optimization_main():
    # list_machines_HP = ['LL','LV','SL','TV','FO','PL']
    list_machines_HP = ['LL','LV','SL']
    matrice_machine = get_mat_power(PR = 0.5)

    fixed_machines = get_fixed_machines()

    batch_empty_plannings = make_batches("data/data.csv","data/data_network.csv")

    batch_init_plannings = init_plannings(batch_empty_plannings,fixed_machines)

    list_max_dc = []
    all_plannings = {}
    solutions = {}

    for key in batch_empty_plannings.keys():
        [plannings,day_consumption] = not_so_stupid_solution(batch_init_plannings[key],matrice_machine,fixed_machines,list_machines_HP)
        # print(day_consumption)
        # print(max(day_consumption))
        list_max_dc.append(max(day_consumption))
        all_plannings[key] = plannings

        solutions[key] = [plannings,day_consumption]

    return solutions