#dict.keys
import random


matrice_type_puissance_par_demie_heure = {'LV':0.65, 'LL':1, 'SL':0.125, 'TV':0.1, 'FG_1':0.1, 'CE_1': 0.18, 'CG': 0.1, 
                                            'FO': 1.6, 'PL': 1.2, 'FG_2': 0.3, 'CE_2': 0.25}

def mutate(matrice_type_puissance_par_demie_heure):
    #give me a random between 0 and 20739
    random_number = random.randint(0, 20739)
    
