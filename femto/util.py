import femto as ss


def get_functions(as_string=False):
    "Returns a list of functions, optionally as string function names"
    funcs = []
    funcs_in_dict = func_dict()
    for key in funcs_in_dict:
        for func in funcs_in_dict[key]:
            funcs.append(func)
    if as_string:
        funcs = [f.__name__ for f in funcs]
    return funcs


def func_dict():
    d = {}
    d['sums'] = [
                 ss.sum00,
                 ss.sum01,
                 ss.sum02,
                 ss.sum03,
                 ss.sum04,
                 ss.sum10,
                 ss.sum11,
                 ss.sum12,
                 ss.p_sum01,
                 ss.p_sum02,
                 ss.p_sum03,
                 ss.p_sum04,
                 ]
    return d
