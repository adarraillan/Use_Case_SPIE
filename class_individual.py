import random

matrice_type_puissance_par_demie_heure = {'LV':0.65, 'LL':1, 'SL':0.125, 'TV':0.1, 'FG_1':0.1, 'CE_1': 0.18, 'CG': 0.1, 
                                                'FO': 1.6, 'PL': 1.2, 'FG_2': 0.3, 'CE_2': 0.25}


class Individual:

    def __init__(self,HC : list[int],plannings : list[dict[str,list]], day_consumption : list[float]):
        self.HC : list[dict[str:int]]
        self.plannings : list[dict[str,list]]
        self.day_consumption : list[float]

        self.HC = HC
        self.plannings = plannings
        self.day_consumption = day_consumption

    def mutate_seq(self,limits : dict[str,int] , machine_type : str, house_index : int):
        #Get planning from house from specified machine
        machine_planning = self.plannings[house_index][machine_type]

        #Get random slot in planning
        halfhour_index_index = random.randint(0,len(machine_planning))
        #Get random direction (plus or minus)
        direction = random.sample([1,-1])

        #Retrieve slot and remove it from the planning
        halfhour_index = machine_planning[halfhour_index_index]
        machine_planning.pop(halfhour_index_index)

        #Saving for return
        halfhour_index_save = halfhour_index

        #Moving the slot
        halfhour_index += direction
        if halfhour_index > limits["end"]:
            halfhour_index = limits["start"]
        elif halfhour_index < limits["start"]:
            halfhour_index = limits["end"]
        while halfhour_index in machine_planning:
            halfhour_index += direction
            if halfhour_index > limits["end"]:
                halfhour_index = limits["start"]
            elif halfhour_index < limits["start"]:
                halfhour_index = limits["end"]

        #Putting new index in planning
        machine_planning.append(halfhour_index)
        machine_planning.sort()
        self.plannings[house_index][machine_type] = machine_planning

        # return (old_index,new_index)
        return (halfhour_index_save,halfhour_index)

    def mutate(self):
        house_index = random.randint(0, len(self.plannings))

        planning_index = self.plannings[house_index]
        machine_type = random.choice(list(planning_index.keys()))
        
        if machine_type == 'CG' or machine_type == 'FG_1' or machine_type == 'FG_2' or machine_type == 'CE_1' or machine_type == 'CE_2':
            (old_index,new_index) = self.mutate_seq(machine_type, house_index)
        else:
            (old_index,new_index) = self.mutate_no_seq(machine_type, house_index)

        self.update(old_index, new_index, machine_type)


    def mutate_no_seq(self,limits : dict[str,int], machine_type : str, house_index : int):
        #Get planning from house from specified machine
        machine_planning = self.plannings[house_index][machine_type]

        #Get random direction (plus or minus)
        direction = random.sample([1,-1])

        #Find loop start and end
        start_loop, end_loop = None,None
        for i in range(len(machine_planning)):
            if i+1 < len(machine_planning):
                if machine_planning[i+1] - machine_planning[i] != 1:
                    start_loop = machine_planning[i+1]
                    end_loop = machine_planning[i]
            else:
                start_loop = min(machine_planning)
                end_loop = max(machine_planning)

        #Calculation of old_index and start_index
        old_index,new_index = None,None
        if direction == 1:
            end_loop = (end_loop+direction)%limits["end"]
            old_index = start_loop
            new_index = end_loop
        elif direction == -1:
            start_loop = (start_loop+direction)%limits["end"]
            old_index = end_loop
            new_index = start_loop

        #Putting new index in planning
        machine_planning.remove(old_index)
        machine_planning.append(new_index)
        machine_planning.sort()
        self.plannings[house_index][machine_type] = machine_planning

        # return (old_index,new_index)
        return (old_index,new_index)
        
        
    def getMax(list):
        max = list[0]
        for i in list:
            if i > max:
                max = i
        return max

        
  
    def update(self,machine_type : str, old_index : int, new_index : int):
        #update the day_consumption
        self.day_consumption[old_index] -= matrice_type_puissance_par_demie_heure[machine_type]
        self.day_consumption[new_index] += matrice_type_puissance_par_demie_heure[machine_type]
