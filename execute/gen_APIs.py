# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import numpy
import re
import math
import pandas as pd
import numpy as np
import datetime

APIs = {}


# With only one argument

### count
APIs['count'] = {"argument":['row'], 'output': 'num', 
                 'function': lambda t :  len(t),
                 'tostr': lambda t : "count {{ {} }}".format(t),
                 'append': True}

### unique
APIs['only'] = {"argument":['row'], 'output': 'bool',
                "function": lambda t: len(t) == 1,
                "tostr": lambda t : "only {{ {} }}".format(t),
                'append': None}


# With only two argument and the first is row
APIs['str_hop'] = {"argument":['row', 'header'], 'output': 'str', 
               'function': lambda t, col :  hop_op(t, col),
               'tostr': lambda t, col : "hop {{ {} ; {} }}".format(t, col),
               'append': True}

APIs['num_hop'] = {"argument":['row', 'header'], 'output': 'obj', 
               'function': lambda t, col :  hop_op(t, col),
               'tostr': lambda t, col : "hop {{ {} ; {} }}".format(t, col),
               'append': True}

APIs['avg'] = {"argument":['row', 'header'], 'output': 'num',
              "function": lambda t, col : agg(t, col, "mean"),
              "tostr": lambda t, col : "avg {{ {} ; {} }}".format(t, col),
              'append': True}

APIs['sum'] = {"argument":['row', 'header'], 'output': 'num',
              "function": lambda t, col : agg(t, col, "sum"),
              "tostr": lambda t, col : "sum {{ {} ; {} }}".format(t, col),
              'append': True}

APIs['max'] = {"argument":['row', 'header'], 'output': 'obj',
              "function": lambda t, col : nth_maxmin(t, col, order=1, max_or_min="max", arg=False),
              "tostr": lambda t, col : "max {{ {} ; {} }}".format(t, col),
              'append': True}

APIs['min'] = {"argument":['row', 'header'], 'output': 'obj',
                "function": lambda t, col : nth_maxmin(t, col, order=1, max_or_min="min", arg=False),
                "tostr": lambda t, col : "min {{ {} ; {} }}".format(t, col),
                'append': True}

APIs['argmax'] = {"argument":['row', 'header'], 'output': 'row',
                  'function': lambda t, col : nth_maxmin(t, col, order=1, max_or_min="max", arg=True),
                  'tostr': lambda t, col : "argmax {{ {} ; {} }}".format(t, col),
                  'append': False}

APIs['argmin'] = {"argument":['row', 'header'], 'output': 'row',
                  'function': lambda t, col :  nth_maxmin(t, col, order=1, max_or_min="min", arg=True),
                  'tostr': lambda t, col : "argmin {{ {} ; {} }}".format(t, col),
                  'append': False}


# add for ordinal
APIs['nth_argmax'] = {"argument":['row', 'header', 'num'], 'output': 'row',
                  'function': lambda t, col, ind : nth_maxmin(t, col, order=ind, max_or_min="max", arg=True),
                  'tostr': lambda t, col, ind : "nth_argmax {{ {} ; {} ; {} }}".format(t, col, ind),
                  'append': False}

APIs['nth_argmin'] = {"argument":['row', 'header', 'num'], 'output': 'row',
                  'function': lambda t, col, ind : nth_maxmin(t, col, order=ind, max_or_min="min", arg=True),
                  'tostr': lambda t, col, ind : "nth_argmin {{ {} ; {} ; {} }}".format(t, col, ind),
                  'append': False}

APIs['nth_max'] = {"argument":['row', 'header', 'num'], 'output': 'num',
              "function": lambda t, col, ind : nth_maxmin(t, col, order=ind, max_or_min="max", arg=False),
              "tostr": lambda t, col, ind : "nth_max {{ {} ; {} ; {} }}".format(t, col, ind),
              'append': True}

APIs['nth_min'] = {"argument":['row', 'header', 'num'], 'output': 'num',
                "function": lambda t, col, ind : nth_maxmin(t, col, order=ind, max_or_min="min", arg=False),
                "tostr": lambda t, col, ind : "nth_min {{ {} ; {} ; {} }}".format(t, col, ind),
                'append': True}



# With only two argument and the first is not row
APIs['diff'] = {"argument":['obj', 'obj'], 'output': 'str', 
                'function': lambda t1, t2 : obj_compare(t1, t2, type="diff"),
                'tostr': lambda t1, t2 : "diff {{ {} ; {} }}".format(t1, t2),
                'append': True}

APIs['greater'] = {"argument":['obj', 'obj'], 'output': 'bool', 
                   'function': lambda t1, t2 :  obj_compare(t1, t2, type="greater"),
                   'tostr': lambda t1, t2 : "greater {{ {} ; {} }}".format(t1, t2),
                   'append': False}

APIs['less'] = {"argument":['obj', 'obj'], 'output': 'bool', 
                'function': lambda t1, t2 :  obj_compare(t1, t2, type="less"),
                'tostr': lambda t1, t2 : "less {{ {} ; {} }}".format(t1, t2),
                'append': True}

APIs['eq'] = {"argument":['obj', 'obj'], 'output': 'bool', 
              'function': lambda t1, t2 :  obj_compare(t1, t2, type="eq"),
              'tostr': lambda t1, t2 : "eq {{ {} ; {} }}".format(t1, t2),
              'append': None}

APIs['not_eq'] = {"argument":['obj', 'obj'], 'output': 'bool', 
                 'function': lambda t1, t2 : obj_compare(t1, t2, type="not_eq"),
                 'tostr': lambda t1, t2 : "not_eq {{ {} ; {} }}".format(t1, t2),
                 "append": None}

APIs['str_eq'] = {"argument":['str', 'str'], 'output': 'bool',
                  # 'function': lambda t1, t2 :  obj_compare(t1, t2, type="eq"),
                  'function': lambda t1, t2 :  t1 in t2 or t2 in t1,
                  'tostr': lambda t1, t2 : "eq {{ {} ; {} }}".format(t1, t2),
                  "append": None}

APIs['not_str_eq'] = {"argument":['str', 'str'], 'output': 'bool',
                     # 'function': lambda t1, t2 :  obj_compare(t1, t2, type="not_eq"),
                     'function': lambda t1, t2 :  t1 not in t2 and t2 not in t1,
                     'tostr': lambda t1, t2 : "not_eq {{ {} ; {} }}".format(t1, t2),
                     "append": None}

APIs['round_eq'] = {"argument":['obj', 'obj'], 'output': 'bool', 
              'function': lambda t1, t2 :  obj_compare(t1, t2, round=True, type="eq"),
              'tostr': lambda t1, t2 : "round_eq {{ {} ; {} }}".format(t1, t2),
              'append': None}

APIs['and'] = {"argument":['bool', 'bool'], 'output': 'bool',
                'function': lambda t1, t2 :  t1 and t2,
                'tostr': lambda t1, t2 : "and {{ {} ; {} }}".format(t1, t2),
                "append": None}



# With only three argument and the first is row
# str
APIs["filter_str_eq"] = {"argument": ['row', 'header', 'str'], "output": "row", 
                        "function": lambda t, col, value: fuzzy_match_filter(t, col, value),
                        "tostr":lambda t, col, value: "filter_eq {{ {} ; {} ; {} }}".format(t, col, value),
                        'append': False}

APIs["filter_str_not_eq"] = {"argument": ['row', 'header', 'str'], "output": "row", 
                        "function": lambda t, col, value: fuzzy_match_filter(t, col, value, negate=True),
                        "tostr":lambda t, col, value: "filter_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
                        'append': False}

# obj: num or str
APIs["filter_eq"] = {"argument": ['row', 'header', 'obj'], "output": "row", 
                    "function": lambda t, col, value: fuzzy_compare_filter(t, col, value, type="eq"),
                    "tostr":lambda t, col, value: "filter_eq {{ {} ; {} ; {} }}".format(t, col, value),
                    'append': False}

APIs["filter_not_eq"] = {"argument": ['row', 'header', 'obj'], "output": "row", 
                    "function": lambda t, col, value: fuzzy_compare_filter(t, col, value, type="not_eq"),
                    "tostr":lambda t, col, value: "filter_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
                    'append': False}

APIs["filter_less"] = {"argument": ['row', 'header', 'obj'], "output": "row", 
                        "function": lambda t, col, value: fuzzy_compare_filter(t, col, value, type="less"),
                        "tostr":lambda t, col, value: "filter_less {{ {} ; {} ; {} }}".format(t, col, value),
                        "append": False}

APIs["filter_greater"] = {"argument": ['row', 'header', 'obj'], "output": "row",
                        "function": lambda t, col, value: fuzzy_compare_filter(t, col, value, type="greater"),
                        "tostr":lambda t, col, value: "filter_greater {{ {} ; {} ; {} }}".format(t, col, value),
                        "append": False}

APIs["filter_greater_eq"] = {"argument": ['row', 'header', 'obj'], "output": "row",
                             "function": lambda t, col, value: fuzzy_compare_filter(t, col, value, type="greater_eq"),
                             "tostr":lambda t, col, value: "filter_greater_eq {{ {} ; {} ; {} }}".format(t, col, value),
                             "append": False}

APIs["filter_less_eq"] = {"argument": ['row', 'header', 'obj'], "output": "row",
                          "function": lambda t, col, value: fuzzy_compare_filter(t, col, value, type="less_eq"),
                          "tostr":lambda t, col, value: "filter_less_eq {{ {} ; {} ; {} }}".format(t, col, value),
                          "append": False}

APIs["filter_all"] = {"argument": ['row', 'header'], "output": "row", 
                        "function": lambda t, col: t,
                        "tostr":lambda t, col: "filter_all {{ {} ; {} }}".format(t, col),
                        'append': False}




# all
# str
APIs["all_str_eq"] = {"argument": ['row', 'header', 'str'], "output": "bool",
                        "function": lambda t, col, value: all_str_filter(t, col, value),
                        "tostr":lambda t, col, value: "all_eq {{ {} ; {} ; {} }}".format(t, col, value),
                        "append": None}

APIs["all_str_not_eq"] = {"argument": ['row', 'header', 'str'], "output": "bool",
                  "function": lambda t, col, value: all_str_filter(t, col, value, neg=True),
                  "tostr":lambda t, col, value: "all_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
                  "append": None}

# obj: num or str
APIs["all_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                  "function": lambda t, col, value: all_filter(t, col, value, type="eq"),
                  "tostr":lambda t, col, value: "all_eq {{ {} ; {} ; {} }}".format(t, col, value),
                  "append": None}

APIs["all_not_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                  "function": lambda t, col, value: all_filter(t, col, value, type="eq",neg=True),
                  "tostr":lambda t, col, value: "all_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
                  "append": None}

APIs["all_less"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                    "function": lambda t, col, value: all_filter(t, col, value, type="less"),
                    "tostr":lambda t, col, value: "all_less {{ {} ; {} ; {} }}".format(t, col, value),
                    "append": None}

APIs["all_less_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                        "function": lambda t, col, value: all_filter(t, col, value, type="less_eq"),
                        "tostr":lambda t, col, value: "all_less_eq {{ {} ; {} ; {} }}".format(t, col, value),
                        "append": None}

APIs["all_greater"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                       "function": lambda t, col, value: all_filter(t, col, value, type="greater"),
                       "tostr":lambda t, col, value: "all_greater {{ {} ; {} ; {} }}".format(t, col, value),
                       "append": None}

APIs["all_greater_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                          "function": lambda t, col, value: all_filter(t, col, value, type="greater_eq"),
                          "tostr":lambda t, col, value: "all_greater_eq {{ {} ; {} ; {} }}".format(t, col, value),
                          "append": None}



# most
# str
APIs["most_str_eq"] = {"argument": ['row', 'header', 'str'], "output": "bool",
                        "function": lambda t, col, value: most_str_filter(t, col, value, neg=False),
                        "tostr":lambda t, col, value: "most_eq {{ {} ; {} ; {} }}".format(t, col, value),
                        "append": None}

APIs["most_str_not_eq"] = {"argument": ['row', 'header', 'str'], "output": "bool",
                  "function": lambda t, col, value: most_str_filter(t, col, value, neg=True),
                  "tostr":lambda t, col, value: "most_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
                  "append": None}

# obj: num or str
APIs["most_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                  "function": lambda t, col, value: most_filter(t, col, value, type="eq", neg=False),
                  "tostr":lambda t, col, value: "most_eq {{ {} ; {} ; {} }}".format(t, col, value),
                  "append": None}

APIs["most_not_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                  "function": lambda t, col, value: most_filter(t, col, value, type="eq", neg=True),
                  "tostr":lambda t, col, value: "most_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
                  "append": None}

APIs["most_less"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                    "function": lambda t, col, value: most_filter(t, col, value, type="less"),
                    "tostr":lambda t, col, value: "most_less {{ {} ; {} ; {} }}".format(t, col, value),
                    "append": None}

APIs["most_less_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                        "function": lambda t, col, value: most_filter(t, col, value, type="less_eq"),
                        "tostr":lambda t, col, value: "most_less_eq {{ {} ; {} ; {} }}".format(t, col, value),
                        "append": None}

APIs["most_greater"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                       "function": lambda t, col, value: most_filter(t, col, value, type="greater"),
                       "tostr":lambda t, col, value: "most_greater {{ {} ; {} ; {} }}".format(t, col, value),
                       "append": None}

APIs["most_greater_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                          "function": lambda t, col, value: most_filter(t, col, value, type="greater_eq"),
                          "tostr":lambda t, col, value: "most_greater_eq {{ {} ; {} ; {} }}".format(t, col, value),
                          "append": None}




month_map = {'january': 1, 'february':2, 'march':3, 'april':4, 'may':5, 'june':6,
                 'july':7, 'august':8, 'september':9, 'october':10, 'november':11, 'december':12, 
                 'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}

### regex list

# number format: 
'''
10
1.12
1,000,000
10:00
1st, 2nd, 3rd, 4th
'''
pat_num = r"([-+]?\s?\d*(?:\s?[:,.]\s?\d+)+\b|[-+]?\s?\d+\b|\d+\s?(?=st|nd|rd|th))"

pat_add = r"((?<==\s)\d+)"

# dates
pat_year = r"\b(\d\d\d\d)\b"
pat_day = r"\b(\d\d?)\b"
pat_month = r"\b((?:jan(?:uary)?|feb(?:ruary)?|mar(?:rch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?))\b"


class ExeError(Exception):
  def __init__(self, message="exe error"):
    self.message = message


### for filter functions. we reset index for the result

# filter_str_eq / not_eq
def fuzzy_match_filter(t, col, val, negate=False, return_index=False):

  trim_t = t[col].str.replace(" ", "")
  trim_val = val.replace(" ", "")

  if negate:
    res = t[~trim_t.str.contains(trim_val, regex=False)]
  else:
    res = t[trim_t.str.contains(trim_val, regex=False)]
  row_index = res.index
  # res = res.reset_index(drop=True)
  if return_index:
      return res, row_index, col
  else:
      return res

# filter nums ...
def fuzzy_compare_filter(t, col, val, type, return_index=False):
  '''
  fuzzy compare and filter out rows. 
  return empty pd if invalid

  type: eq, not_eq, greater, greater_eq, less, less_eq
  '''
  t = t.copy()

  t[col] = t[col].astype('str')
  # t.loc[:, col] = t.loc[:, col].astype('str')

  # dates
  if len(re.findall(pat_month, val)) > 0:
    year_list = t[col].str.extract(pat_year, expand=False)
    day_list = t[col].str.extract(pat_day, expand=False)
    month_list = t[col].str.extract(pat_month, expand=False)
    month_num_list = month_list.map(month_map)

    # pandas at most 2262
    year_list = year_list.fillna("2260").astype("int")
    day_list = day_list.fillna("1").astype("int")
    month_num_list = month_num_list.fillna("1").astype("int")

    # print (year_list)
    # print (day_list)
    # print (month_num_list)

    date_frame = pd.to_datetime(pd.DataFrame({'year': year_list,'month':month_num_list,'day':day_list}))

    # print (date_frame)

    # for val
    year_val = re.findall(pat_year, val)
    if len(year_val) == 0:
      year_val = year_list.iloc[0]
    else:
      year_val = int(year_val[0])

    day_val = re.findall(pat_day, val)
    if len(day_val) == 0:
      day_val = day_list.iloc[0]
    else:
      day_val = int(day_val[0])

    month_val = re.findall(pat_month, val)
    if len(month_val) == 0:
      month_val = month_num_list.iloc[0]
    else:
      month_val = month_map[month_val[0]]

    date_val = datetime.datetime(year_val, month_val, day_val)
    # print (date_val)

    if type == "greater":
      res = t[date_frame > date_val]
    elif type == "greater_eq":
      res = t[date_frame >= date_val]
    elif type == "less":
      res = t[date_frame < date_val]
    elif type == "less_eq":
      res = t[date_frame <= date_val]
    elif type == "eq":
      res = t[date_frame == date_val]
    elif type == "not_eq":
      res = t[~date_frame != date_val]
    row_index = res.index

    # res = res.reset_index(drop=True)
    if return_index:
        return res, row_index, col
    else:
        return res



  # numbers, or mixed numbers and strings
  val_pat = re.findall(pat_num, val)
  if len(val_pat) == 0:
    # return pd.DataFrame(columns=list(t.columns))
    # fall back to full string matching

    if type == "eq":
      res = t[t[col].str.contains(val, regex=False)]
      if return_index:
          return res, res.index, col
    elif type == "not_eq":
      res = t[~t[col].str.contains(val, regex=False)]
      if return_index:
          return res, res.index, col
    else:
      return pd.DataFrame(columns=list(t.columns))

    # return pd.DataFrame(columns=list(t.columns))

  num = val_pat[0].replace(",", "")
  num = num.replace(":", "")
  num = num.replace(" ", "")
  try:
    num = float(num)
  except:
    num = num.replace(".", "")
    num = float(num)
  # print (num)

  pats = t[col].str.extract(pat_add, expand=False)
  if pats.isnull().all():
    pats = t[col].str.extract(pat_num, expand=False)
  if pats.isnull().all():
    return pd.DataFrame(columns=list(t.columns))
  nums = pats.str.replace(",", "")
  nums = nums.str.replace(":", "")
  nums = nums.str.replace(" ", "")
  try:
    nums = nums.astype("float")
  except:
    nums = nums.str.replace(".", "")
    nums = nums.astype("float")
  # print (nums)

  if type == "greater":
    res = t[np.greater(nums, num)]
  elif type == "greater_eq":
    res = t[np.greater_equal(nums, num)]
  elif type == "less":
    res = t[np.less(nums, num)]
  elif type == "less_eq":
    res = t[np.less_equal(nums, num)]
  elif type == "eq":
    res = t[np.isclose(nums, num)]
  elif type == "not_eq":
    res = t[~np.isclose(nums, num)]

  row_index = res.index

  # res = res.reset_index(drop=True)
  if return_index:
      return res, row_index, col
  else:
      return res

  # all invalid

  return pd.DataFrame(columns=list(t.columns))


###for majority- functions, with index returned.
def most_str_filter(t, col, val, neg=False):
    res = fuzzy_match_filter(t, col, val, return_index=False)
    if neg:
        return len(t) // 3 > len(res)
    else:
        return len(t) // 3 <= len(res)

def most_filter(t, col, val, type, neg=False):
    res = fuzzy_compare_filter(t, col, val, type, return_index=False)
    if neg:
        return len(t) // 3 > len(res)
    else:
        return len(t) // 3 <= len(res)

def all_str_filter(t, col, val, neg=False):
    res = fuzzy_match_filter(t, col, val, return_index=False)
    if neg:
        return 0 == len(res)
    else:
        return len(t) == len(res)

def all_filter(t, col, val, type, neg=False):
    res = fuzzy_compare_filter(t, col, val, type, return_index=False)
    if neg:
        return 0 == len(res)
    else:
        return len(t) == len(res)

def majority_judge(subtable, table):
    sub_len = len(subtable)
    all_len = len(table.index)
    if sub_len == all_len:
        return "all"
    elif sub_len >= (all_len // 3):
        return "most"
    else:
        return None



### for comparison
def obj_compare(num1, num2, round=False, type="eq"):


  tolerance = 0.15 if round else 1e-9
  # both numeric
  try:
    num_1 = float(num1)
    num_2 = float(num2)

    # if negate:
    #   return (not math.isclose(num_1, num_2, rel_tol=tolerance))
    # return math.isclose(num_1, num_2, rel_tol=tolerance)

    if type == "eq":
      return math.isclose(num_1, num_2, rel_tol=tolerance)
    elif type == "not_eq":
      return (not math.isclose(num_1, num_2, rel_tol=tolerance))
    elif type == "greater":
      return num_1 > num_2
    elif type == "less":
      return num_1 < num_2
    elif type == "diff":
      return num_1 - num_2




  except ValueError:
    # strings
    # mixed numbers and strings
    num1 = str(num1)
    num2 = str(num2)

    # if type == "eq" and ( num1 in num2 or num2 in num1 ):
    #     return True

    # dates
    # num1
    if len(re.findall(pat_month, num1)) > 0:
      year_val1 = re.findall(pat_year, num1)
      if len(year_val1) == 0:
        year_val1 = int("2260")
      else:
        year_val1 = int(year_val1[0])

      day_val1 = re.findall(pat_day, num1)
      if len(day_val1) == 0:
        day_val1 = int("1")
      else:
        day_val1 = int(day_val1[0])

      month_val1 = re.findall(pat_month, num1)
      if len(month_val1) == 0:
        month_val1 = int("1")
      else:
        month_val1 = month_map[month_val1[0]]

      try:
        date_val1 = datetime.datetime(year_val1, month_val1, day_val1)
      except:
        return ExeError

      # num2
      year_val2 = re.findall(pat_year, num2)
      if len(year_val2) == 0:
        year_val2 = int("2260")
      else:
        year_val2 = int(year_val2[0])

      day_val2 = re.findall(pat_day, num2)
      if len(day_val2) == 0:
        day_val2 = int("1")
      else:
        day_val2 = int(day_val2[0])

      month_val2 = re.findall(pat_month, num2)
      if len(month_val2) == 0:
        month_val2 = int("1")
      else:
        month_val2 = month_map[month_val2[0]]

      try:
        date_val2 = datetime.datetime(year_val2, month_val2, day_val2)
      except:
        return ExeError

      # if negate:
      #   return date_val1 != date_val2
      # else:
      #   return date_val1 == date_val2

      if type == "eq":
        return date_val1 == date_val2
      elif type == "not_eq":
        return date_val1 != date_val2
      elif type == "greater":
        return date_val1 > date_val2
      elif type == "less":
        return date_val1 < date_val2
      # for diff return string
      elif type == "diff":
        return str((date_val1 - date_val2).days) + " days"


    # mixed string and numerical
    val_pat1 = re.findall(pat_num, num1)
    val_pat2 = re.findall(pat_num, num2)
    if len(val_pat1) == 0 or len(val_pat2) == 0:

      # fall back to full string matching
      if type == "not_eq":
        return (num1 not in num2) and (num2 not in num1)
      elif type == "eq":
        return num1 in num2 or num2 in num1
      else:
        return ExeError()

    num_1 = val_pat1[0].replace(",", "")
    num_1 = num_1.replace(":", "")
    num_1 = num_1.replace(" ", "")
    try:
      num_1 = float(num_1)
    except:
      num_1 = num_1.replace(".", "")
      num_1 = float(num_1)

    num_2 = val_pat2[0].replace(",", "")
    num_2 = num_2.replace(":", "")
    num_2 = num_2.replace(" ", "")
    try:
      num_2 = float(num_2)
    except:
      num_2 = num_2.replace(".", "")
      num_2 = float(num_2)


    # if negate:
    #   return (not math.isclose(num_1, num_2, rel_tol=tolerance))
    # return math.isclose(num_1, num_2, rel_tol=tolerance)

    if type == "eq":
      return math.isclose(num_1, num_2, rel_tol=tolerance)
    elif type == "not_eq":
      return (not math.isclose(num_1, num_2, rel_tol=tolerance))
    elif type == "greater":
      return num_1 > num_2
    elif type == "less":
      return num_1 < num_2
    elif type == "diff":
      return num_1 - num_2


### for deciding comparison type
def compare_type(num1, num2, round=False):


  tolerance = 0.15 if round else 1e-9
  # both numeric
  try:
    num_1 = float(num1)
    num_2 = float(num2)

    # if negate:
    #   return (not math.isclose(num_1, num_2, rel_tol=tolerance))
    # return math.isclose(num_1, num_2, rel_tol=tolerance)
    # Here we ignore "not_eq" and "diff", and sample for these two types in main code
    if math.isclose(num_1, num_2, rel_tol=tolerance):
        return "eq"
    elif num_1 > num_2:
        return "greater"
    else:
        return "less"




  except ValueError:
    # strings
    # mixed numbers and strings
    num1 = str(num1)
    num2 = str(num2)

    # if type == "eq" and ( num1 in num2 or num2 in num1 ):
    #     return True

    # dates
    # num1
    if len(re.findall(pat_month, num1)) > 0:
      year_val1 = re.findall(pat_year, num1)
      if len(year_val1) == 0:
        year_val1 = int("2260")
      else:
        year_val1 = int(year_val1[0])

      day_val1 = re.findall(pat_day, num1)
      if len(day_val1) == 0:
        day_val1 = int("1")
      else:
        day_val1 = int(day_val1[0])

      month_val1 = re.findall(pat_month, num1)
      if len(month_val1) == 0:
        month_val1 = int("1")
      else:
        month_val1 = month_map[month_val1[0]]

      try:
        date_val1 = datetime.datetime(year_val1, month_val1, day_val1)
      except:
        return ExeError

      # num2
      year_val2 = re.findall(pat_year, num2)
      if len(year_val2) == 0:
        year_val2 = int("2260")
      else:
        year_val2 = int(year_val2[0])

      day_val2 = re.findall(pat_day, num2)
      if len(day_val2) == 0:
        day_val2 = int("1")
      else:
        day_val2 = int(day_val2[0])

      month_val2 = re.findall(pat_month, num2)
      if len(month_val2) == 0:
        month_val2 = int("1")
      else:
        month_val2 = month_map[month_val2[0]]

      try:
        date_val2 = datetime.datetime(year_val2, month_val2, day_val2)
      except:
        return ExeError

      # if negate:
      #   return date_val1 != date_val2
      # else:
      #   return date_val1 == date_val2

      if date_val1 == date_val2:
          return "eq"
      elif date_val1 > date_val2:
          return "greater"
      else:
          return "less"
      # for diff return string
      # elif type == "diff":
      #   return str((date_val1 - date_val2).days) + " days"


    # mixed string and numerical
    val_pat1 = re.findall(pat_num, num1)
    val_pat2 = re.findall(pat_num, num2)
    if len(val_pat1) == 0 or len(val_pat2) == 0:

      # fall back to full string matching
      if  (num1 not in num2) and (num2 not in num1):
          return "not_eq"
      else:
          return "eq"

    num_1 = val_pat1[0].replace(",", "")
    num_1 = num_1.replace(":", "")
    num_1 = num_1.replace(" ", "")
    try:
      num_1 = float(num_1)
    except:
      num_1 = num_1.replace(".", "")
      num_1 = float(num_1)

    num_2 = val_pat2[0].replace(",", "")
    num_2 = num_2.replace(":", "")
    num_2 = num_2.replace(" ", "")
    try:
      num_2 = float(num_2)
    except:
      num_2 = num_2.replace(".", "")
      num_2 = float(num_2)


    # if negate:
    #   return (not math.isclose(num_1, num_2, rel_tol=tolerance))
    # return math.isclose(num_1, num_2, rel_tol=tolerance)

    if  math.isclose(num_1, num_2, rel_tol=tolerance):
        return "eq"
    elif num_1 > num_2:
        return "greater"
    else:
        return "less"

### for aggregation: sum avg

def agg(t, col, type, return_index=False):
  '''
  sum or avg for aggregation
  '''
  #do not compute on dates
  date_pats = t[col].str.extract(pat_month, expand=False)
  if not date_pats.isnull().all():
      return ExeError()
  # unused
  if t.dtypes[col] == np.int64 or t.dtypes[col] == np.float64:
    if type == "sum":
      res = t[col].sum()
    elif type == "avg":
      res = t[col].mean()

    return res, col if return_index else res

  else:

    pats = t[col].str.extract(pat_add, expand=False)
    if pats.isnull().all():
      pats = t[col].str.extract(pat_num, expand=False)
    if pats.isnull().all():
      return ExeError()
        #change return 0.0 to return ExeError
         # return 0.0, col if return_index else 0.0
    pats.fillna("0.0")
    nums = pats.str.replace(",", "")
    nums = nums.str.replace(":", "")
    nums = nums.str.replace(" ", "")
    try:
      nums = nums.astype("float")
    except:
      nums = nums.str.replace(".", "")
      nums = nums.astype("float")

    # print (nums)
    if type == "sum":
      return nums.sum()
      # return nums.sum(), col if return_index else nums.sum()
    elif type == "mean":
      return nums.mean()
    else:
      return ExeError()

def add_agg_cell(t, col):
  '''
  sum or avg for aggregation
  '''

  # unused
  if t.dtypes[col] == np.int64 or t.dtypes[col] == np.float64:
    sum_res = t[col].sum()
    avg_res = t[col].mean()

    return sum_res, avg_res

  else:

    pats = t[col].str.extract(pat_add, expand=False)
    if pats.isnull().all():
      pats = t[col].str.extract(pat_num, expand=False)
    if pats.isnull().all():
      raise ExeError
    pats.fillna("0.0")
    nums = pats.str.replace(",", "")
    nums = nums.str.replace(":", "")
    nums = nums.str.replace(" ", "")
    try:
      nums = nums.astype("float")
    except:
      nums = nums.str.replace(".", "")
      nums = nums.astype("float")

    # print (nums)

    return nums.sum(), nums.mean()

### for hop 

def hop_op(t, col, return_index=False):
  if len(t) == 0:
      return ExeError()
  if return_index:
      return t[col].values[0], col
  else:
      return t[col].values[0]



### return processed table to compute ranks of cells
def process_num_table(t, col):
    # dates
    date_pats = t[col].str.extract(pat_month, expand=False)
    if not date_pats.isnull().all():
        year_list = t[col].str.extract(pat_year, expand=False)
        day_list = t[col].str.extract(pat_day, expand=False)
        month_list = t[col].str.extract(pat_month, expand=False)
        month_num_list = month_list.map(month_map)

        # pandas at most 2262
        year_list = year_list.fillna("2260").astype("int")
        day_list = day_list.fillna("1").astype("int")
        month_num_list = month_num_list.fillna("1").astype("int")


        try:
            date_series = pd.to_datetime(pd.DataFrame({'year': year_list, 'month': month_num_list, 'day': day_list}))
            # print (date_series)



            return date_series

        except:
            pass

    # mixed string and numerical
    pats = t[col].str.extract(pat_add, expand=False)
    if pats.isnull().all():
        pats = t[col].str.extract(pat_num, expand=False)
    if pats.isnull().values.any():
        raise ExeError()
    nums = pats.str.replace(",", "")
    nums = nums.str.replace(":", "")
    nums = nums.str.replace(" ", "")
    try:
        nums = nums.astype("float")
    except:
        nums = nums.str.replace(".", "")
        nums = nums.astype("float")

    return nums


def judge_num_table(t, col):
    # dates
    date_pats = t[col].str.extract(pat_month, expand=False)
    if not date_pats.isnull().all():
        return True

    # mixed string and numerical
    pats = t[col].str.extract(pat_add, expand=False)
    if pats.isnull().all():
        pats = t[col].str.extract(pat_num, expand=False)
    if pats.isnull().values.any():
        return False
    else:
        return True


def preprocess_num_val(val):
    '''

    :param val: a value selected as target value for filters
    :return: process the value into standard numerical format
    '''


    # dates
    if len(re.findall(pat_month, val)) > 0:
        #if val is a date, do not change it.
        return val


    # numbers, or mixed numbers and strings
    val_pat = re.findall(pat_num, val)
    if len(val_pat) == 0:
        #if val does not contain numbers, keep its original form.
        return val

    #If val is mixed numbers and strings, extract its first numerical term.
    num = val_pat[0].replace(",", "")
    num = num.replace(":", "")
    num = num.replace(" ", "")
    try:
        num = float(num)
    except:
        num = num.replace(".", "")
        num = float(num)
    return str(num)




def filter_check(val):

  val_pat = re.findall(pat_num, val)


  num = val_pat[0].replace(",", "")
  num = num.replace(":", "")
  num = num.replace(" ", "")
  try:
    num = float(num)
  except:
    num = num.replace(".", "")
    num = float(num)
  print (num)






def nth_maxmin(t, col, order=1, max_or_min="max", arg=False, return_index=False):
    '''
      for max, min, argmax, argmin,
      nth_max, nth_min, nth_argmax, nth_argmin

      return string or rows
      '''

    order = int(order)
    ### return the original content for max,min
    # dates
    date_pats = t[col].str.extract(pat_month, expand=False)
    if not date_pats.isnull().all():
        year_list = t[col].str.extract(pat_year, expand=False)
        day_list = t[col].str.extract(pat_day, expand=False)
        month_list = t[col].str.extract(pat_month, expand=False)
        month_num_list = month_list.map(month_map)

        # pandas at most 2262
        year_list = year_list.fillna("2260").astype("int")
        day_list = day_list.fillna("1").astype("int")
        month_num_list = month_num_list.fillna("1").astype("int")

        # print (year_list)
        # print (day_list)
        # print (month_num_list)

        try:
            date_series = pd.to_datetime(pd.DataFrame({'year': year_list, 'month': month_num_list, 'day': day_list}))
            # print (date_series)

            if max_or_min == "max":
                tar_row = date_series.nlargest(order).iloc[[-1]]
            elif max_or_min == "min":
                tar_row = date_series.nsmallest(order).iloc[[-1]]
                # order *= -1

            ind = list(tar_row.index.values)
            if arg:
                res = t.iloc[ind]
            else:
                res = t.iloc[ind][col].values[0]

            if return_index:
                return res, ind, col, t.index
            else:
                return res

        except:
            pass

    # mixed string and numerical
    pats = t[col].str.extract(pat_add, expand=False)
    if pats.isnull().all():
        pats = t[col].str.extract(pat_num, expand=False)
    if pats.isnull().all():
        return ExeError()
    nums = pats.str.replace(",", "")
    nums = nums.str.replace(":", "")
    nums = nums.str.replace(" ", "")
    try:
        nums = nums.astype("float")
    except:
        nums = nums.str.replace(".", "")
        nums = nums.astype("float")

    try:
        if max_or_min == "max":
            tar_row = nums.nlargest(order).iloc[[-1]]
        elif max_or_min == "min":
            tar_row = nums.nsmallest(order).iloc[[-1]]
        ind = list(tar_row.index.values)
        # print (ind)
        # print (t.iloc[ind][col].values)
        if arg:
            res = t.iloc[ind]
        else:
            res = t.iloc[ind][col].values[0]

    except:
        return ExeError()

    # print (res)
    if return_index:
        return res, ind, col, (t.index, order)
    else:
        return res





def is_ascii(s):
    return all(ord(c) < 128 for c in s)



if __name__ == '__main__':
    val = '0.65 - 1.4 v'
    filter_check(val)

    # t = pd.DataFrame([('bird', 389.0),
    #                    ('bird', 24.0),
    #                    ('anc', 80.5),
    #                    ('mammal', np.nan)],
    #                   columns=('class', 'max_speed'))
    # col = 'class'
    # val = 'bird'
    # trim_t = t[col].str.replace(" ", "")
    # trim_val = val.replace(" ", "")
    #
    #
    #
    # res = t[trim_t.str.contains(trim_val, regex=False)]
    #
    # row = res.nlargest(2, columns="max_speed").iloc[[-1]].index
    # print(row)
    # # print(res.index[0], res.)


