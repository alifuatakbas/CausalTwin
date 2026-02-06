from symtable import Class

import numpy as np
import pandas as pd

class CausalNode:
    def __init__(self,name,distribution="normal",parents=None,params=None):
        self.name = name
        self.distribution = distribution
        self.parents = parents or []
        self.params = params or {}
        self.data =None

    def sample(self , n_samples):
        ## Check parents
        parent_values = [p.data for p in self.parents if p.data is not None]

        if not parent_values:
                if self.distribution == "normal":
                    self.data = np.random.normal(
                        self.params.get("mean",0),
                        self.params.get("std",1),
                        n_samples
                  )
        else:
            base = self.params.get("intercept",0)
            slope = self.params.get("slope",1)
            noise_std = self.params.get("noise",0.1)

            combined_parent_effect = np.mean(parent_values,axis=0)

            self.data = base + (slope * combined_parent_effect) + \
                                    np.random.normal(0,noise_std,n_samples)

        return self.data

class CausalGraph:
    def __init__(self):
        self.nodes = {}

    def add_node(self,node):
        self.nodes[node.name] = node

    def _get_execution_order(self):
        ordered_nodes = []
        visited = set()

        def visit(node):
            if node.name not in visited:
                # Recursively visit all parents first (Depth First Search)
                for parent in node.parents:
                    visit(parent)
                visited.add(node.name)
                ordered_nodes.append(node)

        for node in self.nodes.values():
            visit(node)

        return ordered_nodes

    def generate(self, n_samples):
        print(f"--- Starting data generation for {n_samples} samples ---")

        # Determine the correct order to avoid dependency errors
        execution_order = self._get_execution_order()
        results = {}

        for node in execution_order:
            print(f"Generating node: {node.name}...")
            results[node.name] = node.sample(n_samples)

        return pd.DataFrame(results)


if __name__ == "__main__":
    factory = CausalGraph()

    # Define nodes
    age_node = CausalNode("age", params={"mean": 40, "std": 12})

    # income depends on age
    income_node = CausalNode("income", parents=[age_node],
                             params={"intercept": 2000, "slope": 150, "noise": 300})

    # Even if we add income first, CausalGraph should handle it
    factory.add_node(income_node)
    factory.add_node(age_node)

    # Generate the data
    df = factory.generate(n_samples=1000)

    print("\nData Preview:")
    print(df.head())