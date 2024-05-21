def test_load():
  return 'loaded'

def cond_prob(table, A, ax, B, bx):
  subtable = up_table_subset(table, A, 'equals', ax)
  sublist = up_get_column(subtable, B)
  ColumnA = up_get_column(table, A)
  columnB = up_get_column(table, B)
  pBA = sum([1 if i == bx else 0 for i in sublist]) / len(sublist)
  pA = sum([1 if i == ax else 0 for i in columnA]) / len(columnA)
  pA = sum([1 if i == bx else 0 for i in columnB]) / len(columnB)
  return pBA * pA/pB


def compute_probs(neg, pos): 
  p0 = neg/ (neg+pos)
  p1 = pos/(neg+pos)
  return [p0, p1]

def cond_probs_product(table, evidenceRow, target, targetVal):
  zipColumnRow = up_zip_lists(table.columns[:-1], evidenceRow)
  problist = []

for i, j in zipColumnRow: 
  problist += [cond_prob(table, i, j, target, targetVal)]

def naive_bayes(table, evidence_row, target):
  cond_prob_N = cond_probs_product(table, evidence_row, target, 0)
  prior_prob_N = prior_prob(table, target, 0)

  cond_prob_Y = cond_probs_product(table, evidence_row, target, 1)
  prior_prob_Y = prior_prob(table, target, 1)

  prob_target_N = cond_prob_N * prior_prob_N 
  prob_target_Y = cond_prob_Y * prior_prob_Y 

  neg, pos = compute_probs(prob_target_N, prob_target_Y)
  return [neg, pos]
  
