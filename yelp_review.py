import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from  sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

import os

yelp = pd.read_csv('yelp.csv')
yelp.head()
yelp.columns

yelp.describe().columns[1]
yelp.shape
yelp['text']

yelp['comment_length'] = yelp['text'].apply(len)

for col in yelp.describe().columns:
    plt.figure(figsize = (40, 30), dpi = 100)
    sns.distplot(yelp[col], rug = True, kde = False, rug_kws = {"color": "r", "alpha":0.05, "linewidth": 4, "height":0.05})
    plt.title('Distribution of {}\n'.format(col))
    plt.show()

g = sns.FacetGrid(yelp, col = 'stars', size = 10)
g = g.map(plt.hist, 'comment_length')
plt.show()

plt.figure(figsize = (30, 20))
sns.set(font_scale = 3)
sns.boxplot(y = yelp['comment_length'], x = yelp['stars'], data = yelp)
plt.show()

corr = yelp[['cool', 'useful', 'funny', 'comment_length']].groupby(yelp['stars']).mean().corr()
plt.figure(figsize = (40, 25))
sns.heatmap(corr, cmap = 'coolwarm')
plt.title("Heatmap\n")
plt.show()

############
## Model 1
############

X = yelp[(yelp['stars'] == 5) | (yelp['stars'] == 1)]['text']
y = yelp[(yelp['stars'] == 5) | (yelp['stars'] == 1)]['stars']

X = CountVectorizer().fit(X).transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

classifier = MultinomialNB().fit(X_train, y_train)
predictions = classifier.predict(X_test)
print(classification_report(predictions, y_test))



############
## Model 2
############

pipeline = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('classifier', MultinomialNB()),            
            ])
X = yelp[(yelp['stars'] == 5) | (yelp['stars'] == 1)]['text']
y = yelp[(yelp['stars'] == 5) | (yelp['stars'] == 1)]['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y)

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
print(classification_report(predictions, y_test))