from foodsource import *
import numpy as np

class ABC:
    def __init__(self, objective_function, 
                 solution_size:int,
                 lower_bound:float, 
                 upper_bound:float, 
                 swarm_size:int, 
                 max_it:int) -> None:
        
        # NP = Número de fontes de comida, Employeed Bees e Onlooker Bees
        self.__np = int(swarm_size/2)

        self.__solution_size = solution_size

        self.__lower_bound = lower_bound

        self.__upper_bound = upper_bound

        self.__food_sources = []

        # Criando as fontes de comida
        for _ in range(self.__np):
            self.__food_sources.append(FoodSource(solution_size, 
                                            lower_bound, 
                                            upper_bound, 
                                            objective_function))

        self.__max_it = max_it

        # Número máximo de tentativas para melhorar a aptidão de uma fonte de comida
        self.__trial_limit = self.__np * self.__solution_size
        self.best_solution = None

    # Essa função é responsável pela geração de uma nova solução para uma fonte de alimento
    # baseada nos seus visinhos
    def __generate_and_select_new_solution(self, i:int):
        # i = Fonte de alimento que tera sua solução possívelmente atualizada

        # Escolhendo parceito
        partner = np.random.randint(low=0, high=len(self.__food_sources))

        while partner == i:
            partner = np.random.randint(low=0, high=len(self.__food_sources))

        # J = index que será modificiado nessa iteração
        j = np.random.randint(low=0, high=self.__solution_size)

        # Criando nova solução
        new_solution = self.__food_sources[i].get_solution_copy()
        new_solution[j] += np.random.uniform(low=-1.0, high=1.0) * (new_solution[j] - self.__food_sources[partner].get_solution()[j])

        # Verificando se a nova solução está dentro dos limites
        if new_solution[j] > self.__upper_bound:
            new_solution[j] = self.__upper_bound
        elif new_solution[j] < self.__lower_bound:
            new_solution[j] = self.__lower_bound

        # Calculando os novos valores da função objetivo e os valores de aptidão dessa solução
        new_objective_function_value =  self.__food_sources[i].objetive_function(new_solution)
        new_fitness_value = self.__food_sources[i].fitness_function(new_objective_function_value)

        # Como o problema é de minimzação, nós usamos ">"
        if new_fitness_value > self.__food_sources[i].get_fitness_value():
            # Se a solução for melhor, atualizamos a solução
            self.__food_sources[i].update_solution(new_solution, 
                                                       new_objective_function_value, 
                                                       new_fitness_value)
        else:
            # Caso contrário, é somado 1 ao contador de tentativas
            self.__food_sources[i].trials += 1 

    # Essa função é responsável pela fase de calculo de probabilidades para a Onlooker bee phase
    def __calculate_probabilities(self):
        # Calculando probabilidades para cada solução
        probabilities = np.zeros(shape=(len(self.__food_sources)), dtype=float)
        fitness_values = []

        for food_source in self.__food_sources:
            fitness_values.append(food_source.get_fitness_value())

        # Maior valor dentro todos os fitness_values
        max_fv = np.max(fitness_values)

        for i in range(len(self.__food_sources)):
            probabilities[i] = 0.9*(self.__food_sources[i].get_fitness_value()/max_fv)+0.1

        return probabilities

    def __employed_bee_phase(self) -> None:

        for i in range(self.__np):
            self.__generate_and_select_new_solution(i)

    def __onlooker_bee_phase(self, probabilities) -> None:
        solutions = 0 
        i = 0
        # Enquanto np fontes de comida não tiverem suas soluções atualizadas
        while solutions <= self.__np:
            prob = np.random.uniform(low=0, high=1)
            
            # Fonte de comida selecionada para ter solução atualizada
            if prob < probabilities[i]:
                self.__generate_and_select_new_solution(i)
                solutions += 1

            i += 1

            if i > self.__np-1:
                i = 0
        pass

    def __scout_bee_phase(self) -> None:

        # Inicialmente memorizamos a melhor solução encontrada até agora
        best_solution = self.__food_sources[0] # Arbitrário

        for i in range(1, len(self.__food_sources)):
            if self.__food_sources[i].get_fitness_value() > best_solution.get_fitness_value():
                best_solution = self.__food_sources[i]
        
        # Se a melhor solução encontrada nessa iteração for melhor do que a melhor encontrada
        # no processo inteiro, salvamos essa solução
        if self.best_solution == None or best_solution.get_fitness_value() > self.best_solution.get_fitness_value():
            self.best_solution = best_solution

        # Verificando quais fontes de comida passaram do limite de trials
        to_be_scouted = np.zeros(shape=len(self.__food_sources))

        for i in range(len(self.__food_sources)):
            if self.__food_sources[i].trials > self.__trial_limit:
                to_be_scouted = self.__food_sources[i].trials

        i = np.argmax(to_be_scouted)        

        # Essa condição verifica três casos
        # 1° Apenas uma food source passou do seu limite de trials;
        # 2° Multiplas food sources passaram do seu limite de trial, mas uma passou mais;
        # 3° Multiplas food sources passaram do seu limite de trial, sendo esse limite o mesmo para as três,
        #    logo é necessário escolher uma aleatóriamente;
        if np.count_nonzero(to_be_scouted == i) > 1:
            possible_scouted = np.where(to_be_scouted == i)
            i = np.random.choice(possible_scouted[0])

        new_solution = np.random.uniform(low=self.__lower_bound,
                                         high=self.__upper_bound, 
                                         size=self.__solution_size)
        new_objective_function_value = self.__food_sources[i].objetive_function(new_solution)
        new_fitness_value = self.__food_sources[i].fitness_function(new_objective_function_value)
        self.__food_sources[i].update_solution(new_solution,
                                    new_objective_function_value,
                                    new_fitness_value)
    
    def run(self) -> None:

        for _ in range(self.__max_it):
            self.__employed_bee_phase()
            self.__onlooker_bee_phase(self.__calculate_probabilities())
            self.__scout_bee_phase()

        self.best_solution.dump()

# Função objetivo
def f(coordinates:list) -> float:

    if len(coordinates) != 2:
        print(coordinates)

    x = coordinates[0]
    y = coordinates[1]

    return (1-x)**2 + 100 * ((y - x**2)**2)

if __name__ == "__main__":
    a = ABC(f, 2, -5.0, 5.0, 60, 1000)
    a.run()
    pass