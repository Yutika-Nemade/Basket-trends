from flask import Flask, render_template, request
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def analyze():
    file = request.files['file']
    min_support = float(request.form['min_support'])
    min_confidence = float(request.form['min_confidence'])

    data = pd.read_csv(file)
    data['Description'] = data['Description'].str.strip() 
    data.dropna(axis=0, subset=['InvoiceNo'], inplace=True) 
    data['InvoiceNo'] = data['InvoiceNo'].astype('str')
    data = data[~data['InvoiceNo'].str.contains('C')] 

    mybasket = (data[data['Country'] =="India"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))
    
    def my_encode_units(x):
        if x <= 0:
            return 0
        if x >= 1:
            return 1

    my_basket_sets = mybasket.applymap(my_encode_units)

    frequent_itemsets = apriori(my_basket_sets, min_support=min_support, use_colnames=True)

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    return render_template('result.html', rules=rules.to_html())

if __name__ == '__main__':
    app.run(debug=True)
