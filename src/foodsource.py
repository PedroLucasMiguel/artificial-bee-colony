import numpy as np

class FoodSource:
    def __init__(self, solution_size:int, 
                 lower_bound:float, 
                 upper_bound:float, 
                 objective_function) -> None:
        
        self.__solution = np.random.uniform(low=lower_bound, 
                                            high=upper_bound, 
                                            size=solution_size)
        
        self.objetive_function = objective_function

        self.__objetive_function_value = self.objetive_function(self.__solution)

        self.fitness_function = lambda function_Value : \
                                    1.0/(1.0+function_Value) if function_Value >=0.0 \
                                    else 1.0 + np.abs(function_Value)
        
        self.__fitness_value = self.fitness_function(self.__objetive_function_value)

        self.trials = 0

    def get_solution_copy(self):
        return self.__solution.copy()
    
    def get_solution(self):
        return self.__solution
    
    def get_fitness_value(self) -> float:
        return self.__fitness_value

    def update_solution(self, solution, objective_function_value:float, fintness_value:float) -> None:
        self.__solution = solution
        self.__objetive_function_value = objective_function_value
        self.__fitness_value = fintness_value
        self.trials = 0

    def dump(self) -> None:
        print(f"Solution: {self.__solution}")
        print(f"Function value: {self.__objetive_function_value}")
        print(f"Fitness value: {self.__fitness_value}")
        print(f"Trials: {self.trials}")