#! /usr/bin/python
# -*- coding: utf-8 -*-


import time
starttime = time.clock()

import scipy
import random
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
print "hoge"
from sklearn.cross_validation import cross_val_score

from numpy import *

import StringIO
import pydot
import logging
import numpy



#training_data = [[1,1], [2,2], [-1,-1], [-2,-2]]
training_data = []

#training_label = [1, 1, -1, -1]
training_label = []

#test_data = [[3, 3], [-3, -3]]
#test_data = [ ]
#test_key = [ ]



predict_data = []
predict_label = []

features_ = []
labels_ = []

shimdata = {}
count = 1  # shimdataカウント

time1 = time.clock()

#-----------データの下処理----------#
with open('tatsushim.1131', 'r') as fp:
    for line_ in fp:
        keyword, label = line_.rstrip().split()
        shimdata[keyword] = int(label)
        count = count + 1
        #print keyword, label, count




with open('features.tab', 'r') as fp:

    for line_ in fp:

        # shimdataをすべて取り終えたらループを抜ける
        if count == 0:
            break

        line = line_.rstrip().split()

        keyword = line[0]

        if keyword == 'keyword':
            feature_names = line[1:]
            continue

        # lambda式は「lambda 」の後に引数を指定し、「: 」の後に処理を記述
        #function を list の全ての要素に適用し、返された値からなるリストを返す。
        # map(function, list, ...)
        # - なら0、数字ならint(x)で値を返す
        node = map(lambda x: 0 if x == '-' else int(x), line[1:])

        if shimdata.has_key(keyword):

            if random.random() > 0.5:
                training_data.append(node)
                training_label.append(shimdata[keyword])
            else:
                predict_data.append(node)
                predict_label.append(shimdata[keyword])

            features_.append(node)
            labels_.append(shimdata[keyword])

            count = count - 1

        #shimdataに無いもの(ラベル付けされていないもの)はテスト用に使う
#         else:
#             test_data.append(node)
#             test_key.append(keyword)
#---------------------------------#
features = array(features_)
labels = array(labels_)

predict_data_array = array(predict_data)
predict_label_array = array(predict_label)

time2 = time.clock()


model = RandomForestClassifier(n_estimators=200, max_features=5, max_depth=12, min_samples_split=1, compute_importances=True)

# ------------------------------model.fit前のスコア--------------------------------
scores = cross_val_score(model,features,labels)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# computr the future importances
model.fit(features, labels)
importances = model.feature_importances_

#print [treeeee.feature_importances_ for treeeee in model.estimators_]

std = numpy.std([tree.feature_importances_ for tree in model.estimators_], axis=0)


indices = numpy.argsort(importances)[::-1]

# # ------------------------------model.fit後のスコア--------------------------------
# scores = model.score(predict_data_array,predict_label_array)
# #scores = cross_val_score(model,features,labels)
# print("Step2 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# #prediction = model.predict(test_data)

# #i = 0
# # for label in prediction :
# #     #print test_key[i], label
# #     i = i + 1

# Print the feature ranking
print("Feature ranking:")

for (importance, name) in sorted(zip(model.feature_importances_, feature_names), reverse=True):
    print '%s:%.5f' % (name, importance)

####for f in range(10):
#    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    ####print("%d. feature %s (%f)" % (f + 1, , importances[indices[f]]))



# Plot the feature importances of the forest
import pylab
pylab.figure()
pylab.title("Feature importances")
pylab.bar(range(10), importances[indices], color="r", yerr=std[indices], align="center",)
pylab.xticks(range(5), indices)
pylab.xlim([-1, 5])
pylab.show()


#graph visualize
dot_data = StringIO.StringIO()

tree.export_graphviz(model.estimators_[0], out_file=dot_data, feature_names = feature_names)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
try :
    graph.write_pdf("Random-Forest12maxdepth.pdf")

except :
    logging.exception('error in graphviz')

time3 = time.clock()
print time1-starttime,time2-time1,time3-time2
