# imported files: execute_cmd.py & shellChoose.py
# @author Alu

from execute_cmd import CmdExecuter as executer
import shellChoose as shc
import os

user = "luruiyuan"
email = "625079914@qq.com"

git_cmd_template = "git config --{level} --{op_type} {operation}"
LOCAL, GLB, SYS = "local", "global", "system" # 3 levels of git
_add, _rm, _get, _list = "add", "rm", "get", "l" # common commands

def get_git_query(level, op_type, op_item, op_value=''):
    """
    if op_type is sub or get, then op_value should be None
    """
    op_type = op_type.lower()
    operation = " ".join([op_item, op_value]) if op_type in ["add", "get"] else op_item
    return git_cmd_template.format(level=level, op_type=op_type,\
                     operation=operation)

def check_info(item, item_value, cmds, level=GLB):
    """
    if do not have item, then add it use the item_value
    """
    if executer.execute(get_git_query(level, _get, item))[1] == '':
        cmds.append(get_git_query(level, _add, item, item_value))


def path_split(raw_path):
    """
    preprocess the path in __file__, and return path as the result

    Return return (type, dir, base) as the result
    1. Relative: ("relative", "", "../test", "test.py")
    2. Absolute: ("absolute", "d:", "d:\\test", "test.py")
    """
    dir, base = os.path.split(raw_path)
    type = "absolute" if os.path.isabs(raw_path) else "relative"
    return type, dir, base

def set_work_dir(input_path):
    """
     same file, exit
     otherwise, set the wording folder to the input_path
    """
    type, dir, base = path_split(input_path)
    # if different, then set the working path    
    if not os.path.samefile(os.getcwd(), dir):
        os.chdir(dir)
    print("current work path: %s" % os.getcwd())
    
def git_first_push():

    txt = """
    before you use this scripts, please make sure you have created
    a repository on gitHub.

    !!! Note: This operation will initialize your git repository in current folder,
        Do you make sure this is a new repository and you wan to push to git?

    """
    status, in_str = shc.make_choice(txt, default="y")
    if status is True and in_str in 'Yy':
        set_work_dir(__file__)
        cmds = []
        cmds.append("git init")
        # check info validation and generate commands
        check_info("user.name", user, cmds)
        check_info("user.email", email, cmds)
        # input repository names
        local_name = repo_name = commit = ''
        while local_name == '' or repo_name == '' or commit == '':
            local_name = input("please input local repository name:")
            repo_name = input("please input original repository name:")
            commit = input("please input your first commit description:")
        # generate commands
        cmd = 'git remote add {local_name} git@github.com:{account}/{repo_name}.git'\
                .format(local_name=local_name, account=user, repo_name=repo_name)
        cmds.append(cmd)
        # print git status for check
        executer.execute("git status")
        # if validate, then execute other commands, else interrupt
        status, in_str = shc.make_choice("If is correct, then press y/Y to continue:", default='n')
        if status is True and in_str in 'Nn':
            return
        # add, commit, and push
        cmds.append("git add .")
        cmds.append("git commit -m \"%s\"" % (commit))
        cmds.append('git push -u {local_name} master'.format(local_name=local_name))
        # execute all cmds
        executer.execute(cmds)

if __name__ == '__main__':
    git_first_push()