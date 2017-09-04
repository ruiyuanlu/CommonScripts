import subprocess as sub
import logging as logg

def execute_cmd(cmd, print_statusoutput=True, stdin=None, input=None, stdout=None, stderr=None, shell=False, cwd=None, timeout=None, check=True, encoding=None, errors=None):
    if not isinstance(cmd, str):
        raise ArgumentException("传入的命令不是字符串, 类型非法")
    if cmd is None or len(cmd) == 0:
        raise ArgumentException("传入的命令为空")

    print("executing: ", cmd)
    
    status, output = sub.getstatusoutput(cmd)
    if print_statusoutput is True:    
        print("[status: %d] [output: %s]" % (status, output))

    print("executing:", cmd, "finished!")
    
# test
execute_cmd("dir", shell=True, print_statusoutput=False)