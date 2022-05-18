##
# 用于在 python 以及 ipynb 环境中调用 Linux Terminal，方法是
# 用 c 调用 popen 函数来将用户端在 python 上输入的命令通过管道来开启一个 shell 的子进程
# 从而执行用户命令并返回结果
# 
# @author zzn
##

from ctypes import CDLL, c_char_p
# 载入动态链接库
lib = CDLL('../lib/Bash.so.0')


def bash(cmd: str) -> str:
    ''' 在python环境中提供terminal接口，接收用户给定的系统命令并返回输出结果
    
    params: cmd: 进行系统调用的指令
    
    return: 系统返回的结果
    '''
    # 设置句柄
    bash_in_c = lib.inputForShellMode
    bash_in_c.argtypes = [c_char_p]
    bash_in_c.restype = c_char_p
    # 调用以及格式转换
    cmd_char = bytes(cmd, encoding='utf-8')
    result =  bash_in_c(cmd_char)
    end_index = result.rfind(b'\n')
    try:
        result = result[: end_index].decode()
    except Exception:
        print('请输入正确的指令！如有问题可以使用 "对应指令 --help" 的方法查看系统帮助')
    print(result)

    
def bashToFile(cmd: str) -> None:
    ''' 在python环境中提供terminal接口，接收用户给定的系统命令并将返回的输出结果存放在调用程序统一目录下的 "BashOutput.txt" 中
        该函数能详细地给出调用错误
    
    params: cmd: 进行系统调用的指令
    '''
    # 设置句柄
    bash_in_c = lib.inputForFileMode
    bash_in_c.argtypes = [c_char_p]
    # 调用以及格式转换
    cmd_char = bytes(cmd, encoding='utf-8')
    bash_in_c(cmd_char)
