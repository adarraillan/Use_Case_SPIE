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

    def mutate_seq(self,machine_type : str, house_index : int):
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
        if halfhour_index > self.HC["end"]:
            halfhour_index = self.HC["start"]
        elif halfhour_index < self.HC["start"]:
            halfhour_index = self.HC["end"]
        while halfhour_index in machine_planning:
            halfhour_index += direction
            if halfhour_index > self.HC["end"]:
                halfhour_index = self.HC["start"]
            elif halfhour_index < self.HC["start"]:
                halfhour_index = self.HC["end"]

        #Putting new index in planning
        machine_planning.append(halfhour_index)
        machine_planning.sort()
        self.plannings[house_index][machine_type] = machine_planning

        # return (old_index,new_index)
        return (halfhour_index_save,halfhour_index)

    def mutate(self, matrice_type_puissance_par_demie_heure):
        house_index = random.randint(0, len(self.plannings))

        planning_index = self.plannings[house_index]
        machine_type = random.choice(list(planning_index.keys()))
        
        if machine_type == 'CG' or machine_type == 'FG_1' or machine_type == 'FG_2' or machine_type == 'CE_1' or machine_type == 'CE_2':
            (old_index,new_index) = self.mutate_seq(machine_type, house_index)
        else:
            (old_index,new_index) = self.mutate_no_seq(machine_type, house_index)


    def mutate_no_seq(self,limits : dict[str,int], machine_type : str, house_index : int):
        #Get planning from house from specified machine
        machine_planning = self.plannings[house_index][machine_type]

        #Get random direction (plus or minus)
        direction = random.sample([1,-1])

        for i in range(len(machine_planning)):
            machine_planning[i] += direction
            if machine_planning[i] > limits["end"]:
                machine_planning[i] = limits["start"]
            elif machine_planning[i] < limits["start"]:
                machine_planning[i] = limits["end"]
  
    def update(self,machine_type : str):
        pass