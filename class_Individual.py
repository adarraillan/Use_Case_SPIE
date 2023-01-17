

class Individual:

    def __init__(self,HC : list[int],plannings : list[dict[str,list]], day_consumption : list[float]):
        self.HC : list[int]
        self.plannings : list[dict[list]]
        self.day_consumption : list[float]

        self.HC = HC
        self.plannings = plannings
        self.day_consumption = day_consumption

    #a time unit is 30 minutes
    def mutate_seq(self,HC : list(int),count_time_unit : int,machine_type : str, house_index : int):
        planning = self.plannings