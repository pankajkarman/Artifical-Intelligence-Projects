import numpy as np


def print_tree(root, depth=0, indentation = '\t'):
        spaces = depth*indentation
        if isinstance(root, dict):
            print('%s[feature%d < %.3f]' % (spaces, root['feature']+1, root['value']))
            print_tree(root['left'], depth+1)
            print_tree(root['right'], depth+1)
        else:
            print('%s[%s]' % (spaces, root))
            
class DecisionTreeClassifier(object):
    def __init__(self, max_depth, min_size, criteria='information gain'):
        self.max_depth = max_depth
        self.min_size = min_size
        self.criteria = criteria
        
    def feature_based_split(self, data, col, thresh):
        left, right = [], []
        for line in data:
            if line[col] < thresh:
                left.append(line)
            else:
                right.append(line)
        return left, right
        
    def count_values(self, rows):
        count = {}
        for row in rows:
            label = row[-1]
            if label not in count:
                count[label] = 0
            count[label] += 1
        return count 
    
    def impurity(self, rows):
        count = self.count_values(rows)
        if self.criteria == 'gini':
            impurity = 1
            for label in count:
                p = count[label]/float(len(rows))
                impurity -= p**2
        if self.criteria == 'information gain':
            impurity = 0
            for label in count:
                p = count[label]/float(len(rows))
                impurity -= p*np.log2(p)
        return impurity
    
    def information_gain(self, current, left, right):
        p = float(len(left))/(len(left)+len(right))
        return self.impurity(current)-p*self.impurity(left)-(1-p)*self.impurity(right)
    
    def get_best_split(self, data):
        nrows, ncols = np.shape(data)
        labels = np.unique(data[:, -1])
        best_score = 0
        for j in range(ncols-1):
            for row in data:
                groups = self.feature_based_split(data, j, row[j])
                gain = self.information_gain(data, groups[0], groups[1])
                if gain > best_score:
                    best_feature = j
                    best_value = row[j]
                    best_score = gain
                    best_groups = groups
        best_split = {'feature':best_feature, 'value':best_value, \
                      'groups':best_groups}
        return best_split
    
    def leaf_node(self, group):
        classes, counts = np.unique(group[:,-1], return_counts=True)
        return classes[np.argmax(counts)]
    
    def get_child_node(self, node, depth):
        left, right = node['groups']
        left, right = np.array(left), np.array(right)
        del(node['groups'])
        if not list(left) or not list(right):
            sample = list(left) + list(right)
            node['left'] = node['right'] = self.leaf_node(np.array(sample))
            return
        if depth >= self.max_depth:
            node['left'], node['right'] = self.leaf_node(left), self.leaf_node(right)
            return
        if len(left) <= self.min_size:
            node['left'] = self.leaf_node(left)
        else:
            node['left'] = self.get_best_split(left)
            self.get_child_node(node['left'], depth+1)
        if len(right) <= self.min_size:
            node['right'] = self.leaf_node(right)
        else:
            node['right'] = self.get_best_split(right)
            self.get_child_node(node['right'], depth+1)
            
    def build_tree(self):
        self.root = self.get_best_split(self.train)
        self.get_child_node(self.root, 1)
        return self.root
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.train = np.column_stack((X, y))
        self.build_tree()
        return self
        
    def print_tree(self, depth = 0, indentation = '\t'):
        spaces = depth * indentation
        if isinstance(self.root, dict):
            print('%s[feature%d < %.3f]' % (spaces, self.root['feature']+1, self.root['value']))
            print_tree(self.root['left'], depth+1, indentation = indentation)
            print_tree(self.root['right'], depth+1, indentation = indentation)
        else:
            print('%s[%s]' % (spaces, self.root))
            
    def get_sample_prediction(self, node, row):
        if row[node['feature']] < node['value']:
            if isinstance(node['left'], dict):
                return self.get_sample_prediction(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.get_sample_prediction(node['right'], row)
            else:
                return node['right']
            
    def predict(self, test):
        self.pred = np.array([])
        for row in test:
            self.pred = np.append(self.pred, self.get_sample_prediction(self.root, row))
        return self.pred        
