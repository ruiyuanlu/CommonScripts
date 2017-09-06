# @author Alu

from sys import argv

def make_choice(question_str, options=["Yy", "Nn"], default='', end='\n'):
    """
    return valid or not, and the input
    (True, input): input valid
    (False, input): input invalid
    """
    option_str = "|".join(options)
    if isinstance(default, str) and len(default) > 0:
        default_choice = '(default: %s)' % default
    else :
        default_choice = ''
    choice = input('%s [%s] %s:' % (question_str, option_str, default_choice))

    if len(default) > 0 and choice == '':
        return True, default
    # if is valid choice
    valid = False
    for op in options:
        if choice in op:
            valid = True
    return valid , choice