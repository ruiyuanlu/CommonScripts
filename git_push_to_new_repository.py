from execute_cmd import execute_cmd as execute
import shellChoose as shc

user = "luruiyuan"
email = "625079914@qq.com"

git_cmd_template = "git config --{level} --{op_type} {operation}"
LOCAL, GLB, SYS = "local", "global", "system" # 3 levels of git
_add, _rm, _get, _list = "add", "rm", "get", "l" # common commands


def get_git_query(level, op_type, op_item, op_value=None):
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
    if execute(get_git_query(level, _get, item))[1] == '':
        cmds.append(get_git_query(level, _add, item, item_value))

def git_first_push():

    txt = """
    before you use this scripts, please make sure you have created
    a repository on gitHub.
    """

    print(txt)
    status, input_str = shc.make_choice("have you?", "YyNn", default="y")

    if status and input_str in 'Yy':
        cmds = []
        # check info validation and generate commands
        check_info("user.name", user, cmds)
        check_info("user.email", email, cmds)

        # execute first push
        local_name = input("please input local repository name:")
        repo_name = input("please input original repository name:")

        cmd = 'git remote add {local_name} git@github.com: {accou \
                nt_name}/{repo_name}.git'.format(local_name=local_name,\
                account=user, repo_name=repo_name)
        cmds.append(cmd)
        cmds.append('git push -u {local} master'.format(local_name))
        
        # execute all cmds
        for c in cmds:
            execute(c, timeout=240)

if __name__ == '__main__':
    git_first_push()