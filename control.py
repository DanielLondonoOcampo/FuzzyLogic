import numpy as np
import matplotlib.pyplot as plt

class FuzzySet:
    def __init__(self, universe, mf):
        self.universe = universe
        self.mf = mf

    def __and__(self, other):
        return FuzzySet(self.universe, np.minimum(self.mf, other.mf))

    def __or__(self, other):
        return FuzzySet(self.universe, np.maximum(self.mf, other.mf))

class Antecedent:
    def __init__(self, universe, label):
        self.universe = universe
        self.label = label
        self.terms = {}

    def __getitem__(self, key):
        return self.terms[key]

    def __setitem__(self, key, value):
        self.terms[key] = FuzzySet(self.universe, value)

    def view(self, sim=None):
        plt.figure(figsize=(10, 4))
        for term, fuzzy_set in self.terms.items():
            plt.plot(self.universe, fuzzy_set.mf, label=term)
        
        if sim and self.label in sim.input:
            input_value = sim.input[self.label]
            for term, fuzzy_set in self.terms.items():
                membership = fuzzy_set.mf[np.abs(self.universe - input_value).argmin()]
                plt.plot([input_value, input_value], [0, membership], 'k--')
                plt.plot(input_value, membership, 'ko')
            plt.axvline(x=input_value, color='r', linestyle='--', label='Input')

        plt.title(f'Funciones de pertenencia para {self.label.capitalize()}')
        plt.xlabel('Universo')
        plt.ylabel('Pertenencia')
        plt.legend()
        plt.grid(True)
        plt.show()

class Consequent(Antecedent):
    def view(self, sim=None):
        plt.figure(figsize=(8, 4))
        for term, fuzzy_set in self.terms.items():
            plt.plot(self.universe, fuzzy_set.mf, label=term)
        
        if sim and self.label in sim.output:
            output_value = sim.output[self.label]
            plt.axvline(x=output_value, color='r', linestyle='--', label='Output')
            plt.plot(output_value, 0, 'ro')

        plt.title(f'Funciones de pertenencia para {self.label.capitalize()}')
        plt.xlabel('Universo')
        plt.ylabel('Pertenencia')
        plt.legend()
        plt.grid(True)
        plt.show()

class Rule:
    def __init__(self, antecedent, consequent):
        self.antecedent = antecedent
        self.consequent = consequent

class ControlSystem:
    def __init__(self, rules):
        self.rules = rules

class ControlSystemSimulation:
    def __init__(self, control_system):
        self.control_system = control_system
        self.input = {}
        self.output = {}

    def compute(self):
        for rule in self.control_system.rules:
            antecedent_value = self.evaluate_antecedent(rule.antecedent)
            consequent_variable = rule.consequent[0]
            consequent_term = rule.consequent[1]
            
            if consequent_variable.label not in self.output:
                self.output[consequent_variable.label] = {}
            
            if consequent_term not in self.output[consequent_variable.label]:
                self.output[consequent_variable.label][consequent_term] = antecedent_value
            else:
                self.output[consequent_variable.label][consequent_term] = max(
                    self.output[consequent_variable.label][consequent_term],
                    antecedent_value
                )

        # Defuzzification
        for label, terms in self.output.items():
            consequent = next(rule.consequent[0] for rule in self.control_system.rules if rule.consequent[0].label == label)
            numerator = 0
            denominator = 0
            for term, strength in terms.items():
                mf = consequent[term].mf
                numerator += np.sum(mf * consequent.universe * strength)
                denominator += np.sum(mf * strength)
            
            self.output[label] = numerator / denominator if denominator != 0 else 0

    def evaluate_antecedent(self, antecedent):
        if isinstance(antecedent, FuzzySet):
            return np.max(antecedent.mf)
        elif isinstance(antecedent, FuzzyOperators):
            if antecedent.operator == 'and':
                return min(self.evaluate_antecedent(term) for term in antecedent.terms)
            elif antecedent.operator == 'or':
                return max(self.evaluate_antecedent(term) for term in antecedent.terms)
        else:
            raise ValueError("Unknown antecedent type")

class FuzzyOperators:
    def __init__(self, terms, operator):
        self.terms = terms
        self.operator = operator

    def __and__(self, other):
        return FuzzyOperators(self.terms + [other], 'and')

    def __or__(self, other):
        return FuzzyOperators(self.terms + [other], 'or')

# Monkey-patch Antecedent and FuzzySet to add fuzzy operators
Antecedent.__and__ = lambda self, other: FuzzyOperators([self, other], 'and')
Antecedent.__or__ = lambda self, other: FuzzyOperators([self, other], 'or')
FuzzySet.__and__ = lambda self, other: FuzzyOperators([self, other], 'and')
FuzzySet.__or__ = lambda self, other: FuzzyOperators([self, other], 'or')