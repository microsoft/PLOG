import json
import random
from tqdm import tqdm
from collections import defaultdict
import itertools as it

from .APIs import *

func_map = {
  "hop" : "num_hop"
}

func_map_str_replace = {
        "num_hop": "str_hop",
        "eq": "str_eq",
        "filter_eq": "filter_str_eq",
        "not_eq": "not_str_eq",
        "filter_not_eq": "filter_str_not_eq",
        "all_eq": "all_str_eq",
        "all_not_eq": "all_str_not_eq",
        "most_eq": "most_str_eq",
        'most_not_eq': 'most_str_not_eq'
    }

class Node(object):
    def __init__(self, full_table, dict_in):
        '''
		construct tree
		'''
        self.swap_dict = defaultdict(list)
        for op, attr in APIs.items():
            self.swap_dict[' '.join(attr['argument'])].append(op)

        self.full_table = full_table
        self.func = dict_in["func"]
        self.dict_in = dict_in

        # row, num, str, obj, header, bool
        self.arg_type_list = APIs[self.func]["argument"]
        self.arg_list = []

        # [("text_node", a), ("func_node", b)]
        self.child_list = []
        child_list = dict_in["args"]

        assert len(self.arg_type_list) == len(child_list)

        # bool, num, str, row
        self.out_type = APIs[self.func]["output"]

        for each_child in child_list:
            if isinstance(each_child, str):
                self.child_list.append(("text_node", each_child))
            elif isinstance(each_child, dict):
                sub_func_node = Node(self.full_table, each_child)
                self.child_list.append(("func_node", sub_func_node))
            else:
                raise ValueError("child type error")

        self.result = None

    def eval(self):

        for each_child, each_type in zip(self.child_list, self.arg_type_list):
            if each_child[0] == "text_node":
                if each_child[1] == "all_rows":
                    self.arg_list.append(self.full_table)
                else:
                    self.arg_list.append(each_child[1])
            else:
                sub_result = each_child[1].eval()
                # print ("exit func: ", each_child[1].func)

                # invalid
                if isinstance(sub_result, ExeError):
                    # print ("sublevel error")
                    return ExeError()
                elif each_type == "row":
                    if not isinstance(sub_result, pd.DataFrame):
                        # print ("error function return type")
                        return ExeError()
                elif each_type == "bool":
                    if not isinstance(sub_result, bool):
                        # print ("error function return type")
                        return ExeError()
                elif each_type == "str":
                    if not isinstance(sub_result, str):
                        # print ("error function return type")
                        return ExeError()

                self.arg_list.append(sub_result)

        result = APIs[self.func]["function"](*self.arg_list)
        return result

    def eval_index(self):
        row, col, row_scope = [], [], []

        for each_child, each_type in zip(self.child_list, self.arg_type_list):
            if each_child[0] == "text_node":
                if each_child[1] == "all_rows":
                    self.arg_list.append(self.full_table)
                else:
                    self.arg_list.append(each_child[1])
            else:
                sub_result = each_child[1].eval_index()
                if isinstance(sub_result, tuple) and len(sub_result) == 4:
                    sub_result, new_row, new_col, row_scope = sub_result
                    if not isinstance(new_row, list):
                        new_row = new_row.to_list()
                    # if not isinstance(row_scope, list):
                    #     row_scope = row_scope.to_list()
                    # if row_scope:
                    #     scope, order = row_scope
                    #     if not isinstance(scope, list):
                    #         scope = scope.to_list()
                    #     row_scope = (scope, order)


                    col.extend(new_col)
                    row.extend(new_row)


                elif isinstance(sub_result, tuple) and len(sub_result) == 3:
                    sub_result, new_row, new_col = sub_result
                    if not isinstance(new_row, list):
                        new_row = new_row.to_list()

                    col.extend(new_col)
                    row.extend(new_row)
                    # if self.func == 'count' or self.func == 'only':
                    #     return row, col
                elif isinstance(sub_result, tuple) and len(sub_result) == 2:
                    sub_result,  new_col = sub_result
                    col.extend(new_col)
                    # return sub_result



                # print ("exit func: ", each_child[1].func)

                # invalid
                if isinstance(sub_result, ExeError):
                    # print ("sublevel error")
                    return ExeError(), row, col, row_scope
                elif each_type == "row":
                    if not isinstance(sub_result, pd.DataFrame):
                        # print ("error function return type")
                        return ExeError(), row, col, row_scope
                elif each_type == "bool":
                    if not isinstance(sub_result, bool):
                        # print ("error function return type")
                        return ExeError(), row, col, row_scope
                elif each_type == "str":
                    if not isinstance(sub_result, str):
                        # print ("error function return type")
                        return ExeError(), row, col, row_scope

                self.arg_list.append(sub_result)

        result = APIs[self.func]["function"](*self.arg_list)
        if isinstance(result, tuple) and len(result) == 4:
            #row_scope is the subtable rows for functions like max/min, we store it to compute ranks
            new_result, new_row, new_col, row_scope = result
            col.append(new_col)
            return new_result, new_row, col, row_scope

        elif isinstance(result, tuple) and len(result) == 3:
            new_result, new_row, new_col = result
            col.append(new_col)
            return new_result, new_row, col, row_scope
        elif isinstance(result, tuple) and len(result) == 2:
            new_result, new_col = result
            col.append(new_col)
            return new_result, row, col, row_scope
        else:
            return result, row, col, row_scope

    def to_str(self):
        arg_list = []
        for each_child, each_type in zip(self.child_list, self.arg_type_list):
            if each_child[0] == "text_node":
                if each_child[1] == "all_rows":
                    arg_list.append('all_rows')
                else:
                    arg_list.append(each_child[1])
            else:
                sub_result = each_child[1].to_str()
                # print ("exit func: ", each_child[1].func)

                arg_list.append(sub_result)

        result = APIs[self.func]["tostr"](*arg_list)
        return result

    def _mutate_dict(self, dict_in, alpha=0.5, beta=0.5, gamma=0.6, theta=0.15, omega=0.2):
        new_dict = {}
        # mutate function
        new_func = dict_in['func']
        if random.random() > alpha:
            for arg, ops in self.swap_dict.items():
                if dict_in['func'] in ops:
                    swap_func = random.choice(ops)  # have chance not changing
                    new_func = swap_func
                    break
        new_dict['func'] = new_func

        # deal with args
        new_dict['args'] = []
        for each_child in dict_in["args"]:
            if isinstance(each_child, str):
                new_child = each_child
                # mutate int
                if each_child.isnumeric() and random.random() < theta:
                    new_child = max(int(each_child) + random.randint(-10, 10), 0)
                    new_child = str(new_child)  # TODO: float numbers

                # mutate columns
                cols = self.full_table.columns
                if each_child in cols:
                    if random.random() > beta:
                        new_child = random.choice(cols)  # have chance not changing
                        # TODO: content mutation
                new_dict['args'].append(new_child)

            elif isinstance(each_child, dict):
                new_child = self._mutate_dict(each_child)
                new_dict['args'].append(new_child)
            else:
                raise ValueError("child type error")

        return new_dict

    def mutate(self, mutate_num_max=500, alpha=0.5, beta=0.5, gamma=0.6, theta=0.15, omega=0.2):
        mutations = []
        visited_node = set()
        for i in range(mutate_num_max):
            new_dict = self._mutate_dict(self.dict_in, alpha=alpha, beta=beta, gamma=gamma, theta=theta, omega=omega)
            if str(new_dict) not in visited_node:
                visited_node.add(str(new_dict))
                new_node = Node(self.full_table, new_dict)
                # test node
                try:
                    new_result = str(new_node.eval())
                except:
                    continue
                # print(new_result)
                if 'ExeError():' not in new_result:
                # if new_result == 'True':
                    mutations.append(new_node)
                    break
        return mutations

    def _str_replace(self, dict_in, perm):
        new_dict = {}
        # mutate function
        new_func = dict_in['func']
        if dict_in['func'] in func_map_str_replace:
            if perm[0] == 1:
                swap_func =  func_map_str_replace[new_func]
                new_func = swap_func
                perm = perm[1:]
            else:
                perm = perm[1:]
        new_dict['func'] = new_func

        # deal with args
        new_dict['args'] = []
        for each_child in dict_in["args"]:
            if isinstance(each_child, str):
                new_child = each_child
                new_dict['args'].append(new_child)
            elif isinstance(each_child, dict):
                new_child = self._str_replace(each_child, perm)
                new_dict['args'].append(new_child)
            else:
                raise ValueError("child type error")

        return new_dict

    def _num_to_replace(self, dict_in):
        '''

        :param dict_in:
        :return: count of alternative function names that can be replaced
        '''
        count = 0
        func = dict_in['func']
        if func in func_map_str_replace:
            count += 1
        for each_child in dict_in["args"]:
            if isinstance(each_child, dict):
                count += self._num_to_replace(each_child)
        return count

    def str_rep(self):
        '''
        Try to replace functions like "eq" into "str_eq"
        Returns:

        '''
        mutations = set()
        visited_node = set()
        num_to_replace = self._num_to_replace(self.dict_in)
        #Generate full permutations of n-digit binary numbers
        #Each elem in a perm indicates whether (1) or not (0) to replace an alternative func
        s = list(it.product(range(2), repeat=num_to_replace))
        for perm in s:
            new_dict = self._str_replace(self.dict_in, perm)
            if str(new_dict) not in visited_node:
                visited_node.add(str(new_dict))
                new_node = Node(self.full_table, new_dict)
                # test node
                try:
                    new_result = str(new_node.eval_index()[0])
                except:
                    continue
                # print(new_result)
                # if 'ExeError():' not in new_result:
                mutations.add(new_node)
                if new_result == 'True':
                    # print("one correct")
                    return True

        return False

    def str_rep_form(self):
        mutations = set()
        visited_node = set()
        num_to_replace = self._num_to_replace(self.dict_in)
        #Generate full permutations of n-digit binary numbers
        #Each elem in a perm indicates whether (1) or not (0) to replace an alternative func
        s = list(it.product(range(2), repeat=num_to_replace))
        for perm in s:
            new_dict = self._str_replace(self.dict_in, perm)
            if str(new_dict) not in visited_node:
                visited_node.add(str(new_dict))
                new_node = Node(self.full_table, new_dict)
                # test node
                try:
                    new_result = str(new_node.eval_index()[0])
                except:
                    continue
                # print(new_result)
                # if 'ExeError():' not in new_result:
                mutations.add(new_node)
                if new_result == 'True':
                    # print("one correct")
                    return new_dict
        return False

def to_str_all(json_in):
    '''
	transform all logic forms into strings
	'''

    with open(json_in) as f:
        data_in = json.load(f)

    num_all = 0
    num_correct = 0

    for data in tqdm(data_in):

        num_all += 1
        logic = data["logic"]
        logic_str = data['logic_str']

        table_header = data["table_header"]
        table_cont = data["table_cont"]

        try:
            pd_in = defaultdict(list)
            for ind, header in enumerate(table_header):
                for inr, row in enumerate(table_cont):

                    # remove last summarization row
                    if inr == len(table_cont) - 1 \
                            and ("all" in row[0] or "total" in row[0] or "sum" in row[0] or \
                                 "a l l" in row[0] or "t o t a l" in row[0] or "s u m" in row[0]):
                        continue
                    pd_in[header].append(row[ind])

            pd_table = pd.DataFrame(pd_in)
        except Exception:
            continue

        root = Node(pd_table, logic)
        res = root.to_str()

        if res == logic_str[:-7]:
            num_correct += 1
        else:
            print(res)
            print(logic_str)

    print("All: ", num_all)
    print("Correct: ", num_correct)

    print("Correctness Rate: ", float(num_correct) / num_all)

    return num_all, num_correct



def execute(data):
    '''
    Execute a logical form on a table
    Args:
        data:

    Returns:

    '''
    logic = data["logic"]
    table_header = data["table_header"]
    table_cont = data["table_cont"]

    try:
        pd_in = defaultdict(list)
        for ind, header in enumerate(table_header):
            for inr, row in enumerate(table_cont):
                # remove last summarization row
                if inr == len(table_cont) - 1 \
                        and ("all" in row[0] or "total" in row[0] or "sum" in row[0] or
                             "a l l" in row[0] or "t o t a l" in row[0] or "s u m" in row[0]):
                    continue
                pd_in[header].append(row[ind])

        pd_table = pd.DataFrame(pd_in)
    except Exception:
        return False


    root = Node(pd_table, logic)
    res = root.eval_index()
    res = res[0]

    if 'ExeError' in str(res) or not res:
        # The initial trial of execution is based on raw function names like "eq", "filter_eq"
        # However, sometimes "str_eq" or "filter_str_eq" should be the real function name
        # str_eq() enables enumerating all possible replacements such as "eq" to "str_eq"
        res = root.str_rep()
    return res


def execute_logicnlg(pd_table, logic):
    root = Node(pd_table, logic)
    res = root.eval_index()
    res = res[0]
    if 'ExeError' in str(res) or not res:
        # print("res incorrect, try mutations")
        res = root.str_rep()
        # print(res)
    return res


def tostr(data):
    logic = data["logic"]

    table_header = data["table_header"]
    table_cont = data["table_cont"]

    try:
        pd_in = defaultdict(list)
        for ind, header in enumerate(table_header):
            for inr, row in enumerate(table_cont):

                # remove last summarization row
                if inr == len(table_cont) - 1 \
                        and ("all" in row[0] or "total" in row[0] or "sum" in row[0] or
                             "a l l" in row[0] or "t o t a l" in row[0] or "s u m" in row[0]):
                    continue
                pd_in[header].append(row[ind])

        pd_table = pd.DataFrame(pd_in)
    except Exception:
        return False

    root = Node(pd_table, logic)
    res = root.to_str()
    return res