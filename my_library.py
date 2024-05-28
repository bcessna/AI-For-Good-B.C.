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
  tn = sum([1 if pair==[0,0] else 0 for pair in zipped_list])
  tp = sum([1 if pair==[1,1] else 0 for pair in zipped_list])
  fp = sum([1 if pair==[1,0] else 0 for pair in zipped_list])
  fn = sum([1 if pair==[0,1] else 0 for pair in zipped_list])

  precision = tp/(tp+fp) if (tp+fp) else 0
  recall = tp/(tp+fn) if (tp+fn) else 0
  F1 = 2*((precision*recall)/(precision+recall)) if (precision+recall) else 0
  accuracy = (tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn) else 0
  predici_dict = {
      'Precision': precision,
      'Recall': recall,
      'F1': F1,
      'Accuracy': accuracy
  }
  return predici_dict
