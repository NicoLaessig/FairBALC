import numpy as np

class FairletDecomposition:
    def __init__(self):
        self.fairlets = []
        self.fairlet_centers = []

    def balanced(self,p, q, r, b):
      if r==0 and b==0:
        return True
      if r==0 or b==0:
        return False
      return min(r*1./b, b*1./r) >= p*1./q
   
 
    def make_fairlet(self, points, dataset):
        """
        Adds fairlet to fairlet decomposition, returns median cost.

        Parameters:
        - points: Indices of the points in the dataset that form a fairlet.
        - dataset: The entire dataset as a numpy array.

        Returns:
        - The median cost of the fairlet being added.
        """
        self.fairlets.append(points)
        cost_list = [sum([np.linalg.norm(dataset[center, :] - dataset[point, :]) for point in points]) for center in points]
        cost, center_index = min((cost, center) for center, cost in enumerate(cost_list))
        self.fairlet_centers.append(points[center_index])
        return cost
 
 
    def basic_fairlet_decomposition(self, p, q, blues, reds, dataset):
        assert p <= q, "Balance parameters must satisfy p <= q."
        if len(reds) < len(blues):
            reds, blues = blues, reds
        cost = 0
        while len(reds) and len(blues):
            if len(reds) >= q and len(blues) >= p:
                cost += self.make_fairlet(reds[:q] + blues[:p], dataset)
                reds, blues = reds[q:], blues[p:]
            else:
                break
        if len(reds) or len(blues):
            cost += self.make_fairlet(reds + blues, dataset)
        return cost
 
    def node_fairlet_decomposition(self, p, q, node, dataset, donelist, depth=0):
        if len(node.children) == 0:
            # Leaf node processing
            node.reds = [i for i in node.cluster if donelist[i] == 0 and self.colors[i] == 0]
            node.blues = [i for i in node.cluster if donelist[i] == 0 and self.colors[i] == 1]
            return self.basic_fairlet_decomposition(p, q, node.blues, node.reds, dataset)
        
        # Initialize counts for reds and blues for each child
        R = [0] * len(node.children)
        B = [0] * len(node.children)
        
        # Initialize net reds and blues
        NR, NB = 0, 0
        
        # Phase 1: Process each child
        for i, child in enumerate(node.children):
            child.reds = [r for r in child.reds if donelist[r] == 0]
            child.blues = [b for b in child.blues if donelist[b] == 0]
            R[i] = len(child.reds)
            B[i] = len(child.blues)
            
            if R[i] >= B[i]:
                must_remove_red = max(0, R[i] - int(np.floor(B[i] * q / p)))
                NR += must_remove_red
            else:
                must_remove_blue = max(0, B[i] - int(np.floor(R[i] * q / p)))
                NB += must_remove_blue
        
        # Calculate missing based on imbalance
        missing = max(0, int(np.ceil(NR * p / q)) - NB) if NR >= NB else max(0, int(np.ceil(NB * p / q)) - NR)
        
        # Adjust for balance
        # Phase 2 & 3: Adjust reds and blues to approach balance
        for i, child in enumerate(node.children):
            if missing == 0:
                break
            # Adjust logic here similar to above, taking care to update `missing` as you redistribute
            
        # Aggregate from children, marking done
        reds, blues = [], []
        for child in node.children:
            reds.extend([r for r in child.reds if donelist[r] == 0])
            blues.extend([b for b in child.blues if donelist[b] == 0])
            for r in child.reds:
                donelist[r] = 1
            for b in child.blues:
                donelist[b] = 1
        
        # Recurse and decompose further
        return self.basic_fairlet_decomposition(p, q, blues, reds, dataset) + sum(
            self.node_fairlet_decomposition(p, q, child, dataset, donelist, depth + 1) for child in node.children)
 
 
    def tree_fairlet_decomposition(self, p, q, root, dataset, colors):
        """
        Main fairlet clustering function, returns cost with respect to the
        original metric (not tree metric).
        """
        assert p <= q, "Balance parameters must satisfy p <= q."

        # Temporarily store colors for this decomposition process
        self.colors = colors
        root.populate_colors(colors)  # Assume this populates `reds` and `blues` for each node
        
        red_count = len(root.reds)
        blue_count = len(root.blues)
        print(f"Red count: {red_count}, Blue count: {blue_count}")

        # Ensure there is balance before proceeding
        if red_count > 0 and blue_count > 0:
            assert self.balanced(p, q, red_count, blue_count), "Dataset is unbalanced"

        donelist = [0] * dataset.shape[0]  # Tracks which points have been processed
        # Kick off the recursive decomposition process
        cost = self.node_fairlet_decomposition(p, q, root, dataset, donelist)
        
        # Optionally, reset `self.colors` if you won't need it after decomposition
        # self.colors = None
        
        return cost
