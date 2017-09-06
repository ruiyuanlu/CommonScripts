# @author Alu

import subprocess as sub

class CmdExecuter:

    @classmethod
    def execute(cls, cmd, print_statusoutput=True, stdin=None, input=None, stdout=None, stderr=None, shell=False, cwd=None, timeout=None, check=True, encoding=None, errors=None):
        
        def __single_cmd__(cmd, print_statusoutput=True, stdin=None, input=None, stdout=None, stderr=None, shell=False, cwd=None, timeout=None, check=True, encoding=None, errors=None):
            print("exec: ", cmd)
            if cmd == "cmd":
                print("cmd:", cmd)
                return "执行cmd命令就会在windows上卡住, 强制退出了"
            status, output = sub.getstatusoutput(cmd)
            print("[status: %d] [output: %s]" % (status, output))
            print("exec done!")
            return status, output

        def __multiple_cmds__(cmds, print_statusoutput=True, stdin=None, input=None, stdout=None, stderr=None, shell=False, cwd=None, timeout=None, check=True, encoding=None, errors=None):
            results = []
            for c in cmds:
                results.append(__single_cmd__(c))
            return results
        
        # 配置对命令执行函数的分派行为
        dispatch_table = {
            str: __single_cmd__,
            list: __multiple_cmds__,
            tuple: __multiple_cmds__,
        }

        if cmd is None or len(cmd) == 0:
            raise ValueError("传入的命令为空")
        for t, f in dispatch_table.items():
            if isinstance(cmd, t):
                return f (cmd, print_statusoutput=True, stdin=stdin, input=input,
                    stdout=stdout, stderr=stderr, shell=shell, cwd=cwd,\
                    timeout=timeout, check=check, encoding=encoding, errors=errors)
        raise TypeError("传入的 cmd 变量类型非法")

if __name__ == '__main__':
    print("multiple cmmands")
    CmdExecuter.execute(["dir", "git status"])
    print("single command")
    CmdExecuter.execute("git commit")

    # 尝试过的一些 windows 命令会导致阻塞, 如
    # cmd 命令, winver 命令
