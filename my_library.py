def test_load():
  return 'loaded'


def cond_prob(table, evidence, evidence_value, target, target_value):
  t_subset = up_table_subset(table, target, 'equals', target_value)
  e_list = up_get_column(t_subset, evidence)
  p_b_a = sum([1 if v==evidence_value else 0 for v in e_list])/len(e_list)
  return p_b_a + .01  #Laplace smoothing factor

def compute_probs(neg, pos): 
  p0 = neg/ (neg+pos)
  p1 = pos/(neg+pos)
  return [p0, p1]

def cond_probs_product(table, evidenceRow, target, targetVal):
  zipColumnRow = up_zip_lists(table.columns[:-1], evidenceRow)
  problist = []
  for i, j in zipColumnRow: 
    problist += [cond_prob(table, i, j, target, targetVal)]
  return up_product(problist)


def naive_bayes(table, evidence_row, target):
  cond_prob_N = cond_probs_product(table, evidence_row, target, 0)
  prior_prob_N = prior_prob(table, target, 0)
  
  cond_prob_Y = cond_probs_product(table, evidence_row, target, 1)
  prior_prob_Y = prior_prob(table, target, 1)
  
  prob_target_N = (cond_prob_N) * (prior_prob_N) 
  prob_target_Y = (cond_prob_Y) * (prior_prob_Y) 
  
  neg, pos = compute_probs(prob_target_N, prob_target_Y)
  return [neg, pos] 

def prior_prob(table, target, targetVal):
  columnList = up_get_column(table, target)
  pA = sum([1 if i == targetVal else 0 for i in columnList])/len(columnList)
  return pA


def metrics(zipped_list):
  assert isinstance(zipped_list, list), f'Parameter is not a list'
  assert all([isinstance(i, list) for i in zipped_list]), f'Parameter is not a list of lists'
  assert all([len(i) == 2 for i in zipped_list]), f'Parameter is not a zipped list - one or more values is not a pair of items'
  assert all([isinstance(a,(int,float)) and isinstance(b,(int,float)) for a,b in zipped_list]), f'zipped_list contains a non-int or non-float'
  assert all([float(a) in [0.0,1.0] and float(b) in [0.0,1.0] for a,b in zipped_list]), f'zipped_list contains a non-binary value'

  tn = sum([1 if pair==[0,0] else 0 for pair in zipped_list])
  tp = sum([1 if pair==[1,1] else 0 for pair in zipped_list])
  fp = sum([1 if pair==[1,0] else 0 for pair in zipped_list])
  fn = sum([1 if pair==[0,1] else 0 for pair in zipped_list])

  precision = (tp) / (tp + fp) if (tp + fp) else 0
  recall = (tp) / (tp + fn) if (tp + fn) else 0
  f1 = 2*((precision * recall) / (precision + recall)) if (precision + recall) else 0
  accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0

  results = {'Precision': precision, 'Recall': recall, 'F1': f1, 'Accuracy': accuracy}
  return results


def try_archs(full_table, target, architectures, thresholds):

  train_table, test_table = up_train_test_split(full_table, target, 0.4)

  for a in architectures:
    all_results = up_neural_net(train_table, test_table, a, target)
    all_mets = []
    for threshold in thresholds:
      all_predictions = [1 if pos>threshold else 0 for neg, pos in all_results]
      pred_act_list = up_zip_lists(all_predictions, up_get_column(test_table, target))

      mets = metrics(pred_act_list)
      mets ['Threshold'] = threshold
      all_mets += [mets]
    print(f'Architecture: {arch}')
    print(up_metrics_table(all_mets))
  return None 

from sklearn.ensemble import RandomForestClassifier  #make sure this makes it into your library

def run_random_forest(train, test, target, n):
  #target is target column name
  #n is number of trees to use
  X = up_drop_column(train, target)
  y = up_get_column(train, target)

  k_feature_table = up_drop_column(test, target)
  k_actuals = up_get_column(test, target)

  clf = RandomForestClassifier(n_estimators=11, max_depth=2, random_state=0)

  clf.fit(X, y)  #builds the trees as specified above
  probs = clf.predict_proba(k_feature_table)
  pos_probs = [p for n,p in probs]  #probs is list of [neg,pos] like we are used to seeing.

  all_mets = []
  for t in thresholds:
    all_predictions = [1 if pos>t else 0 for pos in pos_probs]
    pred_act_list = up_zip_lists(all_predictions, k_actuals)
    mets = metrics(pred_act_list)
    mets['Threshold'] = t
    all_mets = all_mets + [mets]

  metrics_table = up_metrics_table(all_mets)
  metrics_table

  return metrics_table


