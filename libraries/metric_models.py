#Based on Sally's original code, but modified to make computing the surface properties/features part
#of the model fitting and evaluation process to avoid needing to keep track of these things in the
#genetic algorithm.
#I've done my best to note where I've added things
#(I changed a bunch of variable names, but didn't note that)

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
from sklearn.base import clone

#ADDED BY JIM
from property_functions import gr_int
#Really don't want to have to do this... 
#If we already have property_function associated with a property, shouldn't need to

# for simplicity, this can only be a scalar
# need to make a separate property for each element of a vector
# in the future, maybe make a DiffusivityMetricVector that wraps this - Yeah, this makes sense to me now

#JIM switched name to Property because I use metric to describe the thing used to characterize fitness in the genetic algorithm
class rdfLinRegProperty: 
  #def __init__(self, label, property_function, *args):
  def __init__(self, label, property_function, useful=True):
    self.label = label
    self.function = property_function # do we actually need the function??? - JIM thinks yes... but need to do right
    #self.function_args = args # so that you can reuse the same function
    #self.n = n # length of property
    #self.useful = np.ones(self.n) # if you change n, need to change this as well!! somehow enforce this...
    self.useful = useful
  #def compute(self, struct, *args): # need to look up syntax for this
  #    return self.function(struct, *args)

# need initiated model
class rdfLinRegModel:
  def __init__(self, initial_properties, model, scaler, rmse_norm_threshold):
    self.properties = initial_properties
    self.model = model # check syntax
    self.scaler = scaler
    self.rmse_norm = rmse_norm_threshold # is this the best way to do it?

  #ADDED BY JIM
  def calc_properties(self, structObj):
    #I would argue that the properties that get used are part of the model
    #So don't want to save with the structure objects, want to save with the model
    #Makes less efficient, but don't worry about that
    #Will take a structure object and return a dictionary of properties
    propDict = {}

    num_gr = np.max(structObj.chainTypes) + 1
    for i in range(num_gr):
      this_gr = gr_int(structObj, i)
      for i_gr, gr_val in enumerate(this_gr):
        propDict['gr_%i_%i'%(i, i_gr)] = gr_val

    #Above is less general... what we WANT to do is something like
    #for aprop in self.properties:
    #  propDict[aprop.label] = aprop.function(structObj)
    #Won't work, though, because don't want whole rdf... each bin is a separate property

    return propDict


  def build_model(self, fname_structlib):

    #MODIFIED BY JIM (this way don't have to remember to close the file...)
    with open(fname_structlib, 'rb') as f_structlib:
      structs = pickle.load(f_structlib)

    n_structs = 0
    for struct in structs:
      if not struct.metricpredicted:
        n_structs += 1
    metrics = np.zeros(n_structs)
    
    n_features = 0
    for prop in self.properties:
      if prop.useful:
        n_features += 1
    features = np.zeros((n_structs, n_features))

    count_structs = 0
    for struct in structs:
      if not struct.metricpredicted:
        props = self.calc_properties(struct)
        count_features = 0
        for prop in self.properties: # make sure this happens in the same order each time
          if prop.useful:
            #Need to prune properties we don't need (i.e. smaller rdf, etc.)
            try:
              features[count_structs, count_features] = props[prop.label]
              count_features += 1
            except KeyError:
              #Remove this property so don't have to do this again
              prop.useful = False
        metrics[count_structs] = struct.metric
        count_structs += 1

    # cross-validation etc. etc. and change property.useful's
    # need to make sure that property.useful status is consistent with the model (has same number of features)
    # make new model to test with
    test_model = clone(self.model)
    test_scaler = clone(self.scaler)
    # split data into testing and training sets
    features_train, features_test, metrics_train, metrics_test = train_test_split(features, 
                                                                                  metrics, 
                                                                                  test_size=0.25, 
                                                                                  shuffle=True)
    # using training set, perform feature selection by selecting from fitted LASSO model
    features_train_scaled = test_scaler.fit_transform(features_train)
    features_test_scaled = test_scaler.transform(features_test)
    selector = SelectFromModel(test_model, threshold=1e-4) # HARD CODED NUMBER HERE
    selector.fit(features_train_scaled, metrics_train)
    print('number of features selected', np.sum(selector.get_support().astype(int)))
    features_train_reduced_unscaled = selector.transform(features_train)
    features_test_reduced_unscaled = selector.transform(features_test)
    
    # using training set, perform recursive feature elimination with cross-validation
#     selector = RFECV(test_model, step=1, scoring='neg_mean_squared_error')
#     features_train_new = selector.fit_transform(features_train, metrics_train)
#     print('number of features selected after cross-validation', selector.n_features_)
#     features_test_new = selector.transform(features_test)
#     features_new = selector.transform(features)
    
    # fit with reduced number of features
    features_train_reduced_scaled = test_scaler.fit_transform(features_train_reduced_unscaled)
    features_test_reduced_scaled = test_scaler.transform(features_test_reduced_unscaled)
    test_model.fit(features_train_reduced_scaled, metrics_train)
    
    # compute RMSE of test set
    # should also compute for training set??
    mse_test = mean_squared_error(metrics_test, test_model.predict(features_test_reduced_scaled))
    # Below switching to using coefficient of determination, not RMSE, but still calling it RMSE
    # This normalizes things to the variance in the data, so now want to be bigger and close to 1
    # A good cutoff is probably 0.8 or 0.9
    #rmse_norm_new = np.sqrt(mse_test)/np.mean(metrics)
    rmse_norm_new = (np.var(metrics) - mse_test) / np.var(metrics)
    print('rmse_norm_new', rmse_norm_new)
    print('self.rmse_norm', self.rmse_norm)
    #if rmse_norm_new < self.rmse_norm: # should we do something fancier than this?
      # copy model (or should we maybe refit it to all the data?? not sure if this would violate something machine learning)
    self.scaler = clone(test_scaler)
    features_train_reduced_scaled = self.scaler.fit_transform(features_train_reduced_unscaled)
    self.model = clone(test_model)
    self.model.fit(features_train_reduced_scaled, metrics_train)
    # change useful labels on properties
    count_features = 0
    selector_support = selector.get_support()
    for prop in self.properties:
      if prop.useful:
        prop.useful = selector_support[count_features]
        count_features += 1

    if rmse_norm_new > self.rmse_norm: # should we do something fancier than this?
      self.rmse_norm = rmse_norm_new
      return True
    else:
      return False

  #MODIFIED BY JIM (to compute properties dictionary and reference this instead of struct.properties)
  def model_prediction(self, struct):
    props = self.calc_properties(struct)
    features = []
    for prop in self.properties:
      if prop.useful:
        features.append(props[prop.label])
    features_scaled = self.scaler.transform(np.reshape(features, (1, len(features))))
    return np.asscalar(self.model.predict(features_scaled))


