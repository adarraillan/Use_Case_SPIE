import random

matrice_type_puissance_par_demie_heure = {'LV':0.65, 'LL':1, 'SL':0.125, 'TV':0.1, 'FG_1':0.1, 'CE_1': 0.18, 'CG': 0.1, 
                                                'FO': 1.6, 'PL': 1.2, 'FG_2': 0.3, 'CE_2': 0.25}


class Individual:

    def __init__(self,HC : list[int],plannings : list[dict[str,list]], day_consumption : list[float]):
        self.HC : list[int]
        self.plannings : list[dict[list]]
        self.day_consumption : list[float]

        self.HC = HC
        self.plannings = plannings
        self.day_consumption = day_consumption


    def mutate(self, matrice_type_puissance_par_demie_heure):
        house_index = random.randint(0, len(self.plannings))

        planning_index = self.plannings[house_index]
        machine_type = random.choice(list(planning_index.keys()))
        
        if machine_type == 'CG' || machine_type == 'FG_1' || machine_type == 'FG_2' || machine_type == 'CE_1' || machine_type == 'CE_2':
            self.mutate_seq(machine_type, house_index)
        else:
            self.mutate_no_seq(machine_type, house_index)


    #a time unit is 30 minutes
    def mutate_seq(self, machine_type : str, house_index : int):
        planning = self.plannings

    def mutate_no_seq(self, machine_type : str, house_index : int):
        planning = self.plannings
  


   


