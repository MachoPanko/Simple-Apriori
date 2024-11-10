import csv
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
from itertools import combinations
# Transform data to buckets of unique items
filename = 'POIdata_cityB.csv'
min_support = 0.1



def load_dataset(filename):
    buckets_list = []
    buckets_dict = {}
    with open(filename, 'r') as csvfile:
        datareader = csv.reader(csvfile)
        next(datareader)  
        for row in datareader:
            key = (row[0], row[1])
            if buckets_dict.get(key, 0):
                buckets_dict[key].add(row[2])
            else:
                buckets_dict[key] = {row[2]}
        
        buckets_list = [sorted(list(val)) for val in buckets_dict.values()]
     
        # print(len(buckets_dict.get(('1','35'))))
    return buckets_list

def get_initial_sets(filename):
    # Set to store unique values (sets automatically handle uniqueness)
    unique_values = set()
    #all_values = []
    # Read the CSV file
    with open(filename, 'r') as csvfile:
        
        datareader = csv.reader(csvfile)
        next(datareader) 
        # Add each value from third column to the set
        for row in datareader:
            unique_values.add(row[2])  # 2 is the index for third column (0-based indexing)
            #all_values.append(row[2])
    # Convert set to sorted list for better readability
    return [[x] for x in sorted(list(unique_values))] ##, sorted(all_values)


##counting support
def get_frequent(candidate_set , dataset , min_support):
    frequent_set = []
    denominator = len(dataset)
    for candidate in candidate_set:
        support = 0 
        candidate_set = set(candidate)
        for data in dataset:
            data_set = set(data)
            if candidate_set.issubset(data_set):
                support += 1
        if support/denominator >= min_support:
            frequent_set.append(candidate)
    return frequent_set

##utils
def get_all_combinations(char_list):
    # Get only combinations of length 2
    pairs = combinations(char_list, 2)
    # Join each pair into a string
    return [''.join(pair) for pair in pairs]


##candidate generation from frequent set
def get_candidate(frequent_set, k):
    candidate_set = []
    n = len(frequent_set)
    for i in range(n):
        for j in range(i+1,n):
            item1 = frequent_set[i]
            item2 = frequent_set[j]
            
            if k == 1:
                # For k=1, simply combine single items
                items_combined = sorted(item1 + item2)
                candidate_set.append(items_combined)
            else:
                # For k>1, check if first k-1 items are same
                if item1[:k-1] == item2[:k-1]:
                    # Combine and maintain sorted order
                    items_combined = sorted(list(set(item1 + item2)))
                    if len(items_combined) == k + 1:  # Ensure we're only getting k+1 size itemsets
                        candidate_set.append(items_combined)
    
    return candidate_set
    

def pruned_remaining_candidates(candidate_set, previous_frequent_set):
    pruned__remaining_candidates_set = []
    
    # Convert previous_frequent_set lists to tuples for set operations
    previous_frequent_tuples = set(tuple(x) for x in previous_frequent_set)
    
    for candidate in candidate_set:
        required_frequent_set = []
        # Generate all k-1 subsets of the candidate
        for i in range(len(candidate)):
            subset = candidate[:i] + candidate[i+1:]
            required_frequent_set.append(tuple(subset))
            
        # Check if all required subsets are in previous frequent set
        if set(required_frequent_set).issubset(previous_frequent_tuples):
            pruned__remaining_candidates_set.append(candidate)
            
    return pruned__remaining_candidates_set

##load data
dataset = load_dataset(filename)
# print(dataset)
##initial sets
candidate_set = get_initial_sets(filename)
k = 1
all_frequent = []
frequent_set = get_frequent(candidate_set= candidate_set , dataset= dataset, min_support= min_support)
while len(frequent_set):
    all_frequent.extend(frequent_set)
    candidate_set = get_candidate(frequent_set, k=k)
    candidate_set = pruned_remaining_candidates(candidate_set , frequent_set)
    frequent_set = get_frequent(candidate_set,dataset, min_support= min_support)
    k += 1

## at this point all_frequent represents all frequent sets

print("All Frequents: \n", all_frequent)

## Part 2 Rule Generation
min_confidence = 0.5

from itertools import combinations


def get_subsets_and_complements(itemset):
    result = []
    # Generate combinations of lengths 1 to len(itemset)-1 
    # (excluding empty set and full set)
    for i in range(1, len(itemset)):
        # Generate all combinations of length i
        for subset in combinations(itemset, i):
           
            subset = list(subset)
            # Get complement (I-A) by getting all items not in subset
            complement = [x for x in itemset if x not in subset]
            result.append((subset, complement, itemset))

    return result
print( get_subsets_and_complements(['47', '79']))
def get_support(element, dataset):
    element_set = set(element)
    denominator = len(dataset)
    support= 0
    for data in dataset:
            data_set = set(data)
            if element_set.issubset(data_set):
                support += 1
    return support/denominator

print(get_support(['47'],dataset))

def get_confidence(supplement, complement , itemset):
    confidence = get_support(itemset,dataset) / get_support(supplement,dataset)
    return confidence

def is_confident(confidence, min_confidence = min_confidence):
    return confidence >= min_confidence


all_rules = []

def get_lift( confidence , complement):
    return confidence/ get_support(complement,dataset)

for itemset in all_frequent:
    possible_rules = get_subsets_and_complements(itemset)
    for supplement, complement , itemset_og  in possible_rules:
        confidence = get_confidence(supplement, complement,itemset_og)
        lift = get_lift(confidence, complement)
        if lift >= 1.0:
            all_rules.append((supplement, complement, itemset_og, confidence , lift))
            # print(f"Rule: {supplement} -> {complement} [conf={confidence:.2f} , lift = {lift:.2f}]")


sorted_rules = sorted(all_rules, key=lambda x: x[4], reverse=True)
print("\nRules sorted by lift:")
print(f"{'Antecedent':<20} {'Consequent':<20} {'Confidence':<12} {'Lift':<10}")
print("-" * 62)
for supplement, complement, itemset_og, confidence, lift in sorted_rules:
    antecedent = str(supplement)
    consequent = str(complement)
    print(f"{antecedent:<20} {consequent:<20} {confidence:.2f}        {lift:.2f}")            
