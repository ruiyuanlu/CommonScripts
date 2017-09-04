from sys import argv

def make_choice(question_str, *options, default='', end='\n'):
    """
    return valid or not, and the input
    (True, input): input valid
    (False, input): input invalid
    """
    iterable = False
    if isinstance(options, (tuple, list)):
        option_str = "/".join(options)
        iterable = True
    else :
        option_str = str(options)
    if isinstance(default, str) and len(default) > 0:
        default_str = '(default: %s)' % default
    else :
        default_str = ''
    input_str = input('%s [%s] %s:' % (question_str, option_str, default_str))

    if len(default) > 0 and input_str == '':
        return True, default

    return ((iterable and options.count(input_str) > 0) 
            or (not iterable and option_str == input_str)) , input_str

# test
# print(make_choice("have you?", 'Y','y','N','n', default='Y'))