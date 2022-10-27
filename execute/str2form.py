#coding=utf-8

class ArrayStack():
    """LIFO Stack implementation using a Python list as underlying storage"""

    def __init__(self, n):
        """Create an empty stack."""
        self.data = []
        self.maxLen = n  # n : an integer that represent the max elements capacity of the stack

    def __len__(self):
        """Return the number of elements in the stack"""
        return len(self.data)

    def is_empty(self):
        """Return True if the stack is empty"""
        return len(self.data) == 0

    def is_full(self):
        """Return True if the stack is full"""
        return len(self.data) == self.maxLen

    def push(self, e):
        """Add element e to the top of the stack

         Raise Empty exception if the stack is full"""
        if self.is_full():
            raise AssertionError('Stack is full')
        return self.data.append(e)

    def top(self):
        """Return the element at the top of the stack, but not move it.

        Raise Empty exception if the stack is empty"""
        if self.is_empty():
            raise AssertionError('Stack is empty')
        return self.data[-1]

    def pop(self):
        """Return the element at the top of the stack, meanwhile move it.

        Raise Empty exception if the stack is empty"""
        if self.is_empty():
            raise AssertionError('Stack is empty')
        return self.data.pop()




def str2list(logic_str : str) -> list:
    '''

    :param logic_str: original logic_str
    :return: splitted token list
    '''

    # Prune affix "= true" if it exists
    if logic_str.endswith('true'):
        logic_str = logic_str[:-7]
    # detect empty column headers
    while logic_str.find("; }") > 0:
        idx = logic_str.find("; }")
        logic_str = logic_str[:idx+2] + "[None] " + logic_str[idx+2:]
    while logic_str.find("; ;") > 0:
        idx = logic_str.find("; ;")
        logic_str = logic_str[:idx+2] + "[None] " + logic_str[idx+2:]
    unreplaced_logic = logic_str[:].split(" ")
    logic = []
    for tok in unreplaced_logic:
        if tok == "[None]":
            tok = ""
        logic.append(tok)
    token_list = []
    i = 0
    while i < len(logic):
        cur_token = logic[i]
        if cur_token in ["{", "}", ";"]:
            token_list.append(cur_token)
            i = i + 1
            continue
        i = i + 1
        while i < len(logic) and not logic[i] in ["{", "}", ";"]:
            cur_token = " ".join([cur_token, logic[i]])
            i = i + 1
        token_list.append(cur_token)
    return token_list

def parse_str(logic_str : str, func_map):
    '''
    Parsing a logical form from a logic str
    Args:
        logic_str: a logic str
        func_map: a function-to-function map

    Returns:
        final_form: a structured logical form, dict
    '''
    token_list = str2list(logic_str)
    logic_stack = ArrayStack(len(token_list))
    func_stack = []
    i = 0
    func_idx = 0
    while i < len(token_list):
        cur_dict = {}
        cur_args = []
        while token_list[i] != "}":
            logic_stack.push(token_list[i])
            i = i + 1
        while logic_stack.top() != "{":
            if logic_stack.top() != ";" and isinstance(logic_stack.top(), str):
                cur_args.append(logic_stack.pop())
            elif logic_stack.top() == ";":
                logic_stack.pop()
            elif isinstance(logic_stack.top(), int):
                cur_args.append(func_stack[logic_stack.pop()])
        # pop "{"
        logic_stack.pop()
        # pop and store the function
        func = logic_stack.pop()
        if func in func_map.keys():
            func = func_map[func]
        cur_dict["func"] = func
        cur_dict["args"] = cur_args[::-1]
        func_stack.append(cur_dict)
        # push the index into logic_stack
        logic_stack.push(func_idx)
        func_idx += 1
        i = i + 1
    final_form = func_stack[-1]
    return final_form


