import csv
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Transform data to buckets of unique items
filename = 'POIdata_cityB.csv'
buckets_dict = {}
buckets_list = []

with open(filename, 'r') as csvfile:
    datareader = csv.reader(csvfile)
    for row in datareader:
        key = (row[0], row[1])
        if buckets_dict.get(key, 0):
            buckets_dict[key].add(row[2])
        else:
            buckets_dict[key] = {row[2]}
    
    buckets_list = [list(val) for val in buckets_dict.values()]

# Convert transactions to DataFrame
te = TransactionEncoder()
te_ary = te.fit(buckets_list).transform(buckets_list)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Generate frequent itemsets
frequent_itemsets = fpgrowth(df, min_support=0.1, use_colnames=True)

# Calculate number of itemsets to use (using all available itemsets)
num_items = len(frequent_itemsets)

# Generate rules with num_itemsets parameter
rules = association_rules(frequent_itemsets, 
                     
                        metric="lift", 
                        min_threshold=1.0)

# Filter to show only high lift values
high_lift_rules = rules[rules['lift'] > 1]

# Sort by lift to find the strongest associations
high_lift_rules_sorted = high_lift_rules.sort_values(by='lift', ascending=False)

print("Top POI combinations by lift:")
print(high_lift_rules_sorted[['antecedents', 'consequents', 'support', 'confidence', 'lift']])