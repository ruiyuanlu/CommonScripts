Get more detailed info at 

my blog:  <url>http://www.cnblogs.com/luruiyuan/p/7600931.html<url>

# haven't finished:

logger: httpHandler, socketHandler, Windows support still has bugs.

# updated:

| files     | updated items                            | comments                                 |
| --------- | ---------------------------------------- | ---------------------------------------- |
| logger.py | a log related code                       | Add a colorful logger                    |
| logger.py | forbidden color settings if in Win platform |                                          |
| logger.py | add relative path support                | Add relative/absolute path as prefix of filename, and the logger will save log files into that directory automatically (Create new dir and files if neccessary) **Note: the relative path calculation is based on the locatoin of logger.py!!!** |
|logger.py|Posix support bug fixed|

## Parameters:

| param                         | purpose                                  |
| :---------------------------- | ---------------------------------------- |
| loggername='',                | the name of logger                       |
| cmdlog=True,                  | whether write to cmd                     |
| filelog=True,                 | whether write to file                    |
| filename='myApp.log',         | the name of log file, default is 'myApp.log', |
| filemode='a',                 | write mode                               |
| colorful=True,                | whether the shell output should be set to colorful |
| cmd_color_dict=None,          | if none, then it will be set to  {'debug': 'green', 'warning':'yellow', 'error':'red', 'critical':'purple'} |
| cmdlevel='DEBUG',             | default shell log level                  |
| cmdfmt=DEFAULT_FMT,           | '\[%(levelname)s\]\[%(asctime)s\] %(filename)s [line:%(lineno)d] %(message)s' |
| cmddatefmt=DEFAULT_DATE_FMT,  | '%Y-%m-%d %a, %p %H:%M:%S'               |
| filelevel='INFO',             | default file log level                   |
| filefmt=DEFAULT_FMT,          | '\[%(levelname)s\]\[%(asctime)s\] %(filename)s [line:%(lineno)d] %(message)s' |
| filedatefmt=DEFAULT_DATE_FMT, | '%Y-%m-%d %a, %p %H:%M:%S'               |
| backup_count=0,               | the number of backup log files           |
| limit=10240,                  | 10240=10*1024k=10M                       |
| when=None                     | when is used to set for Time based log   |



# Finished

| files         | functional description                   | comments                                 |
| ------------- | ---------------------------------------- | ---------------------------------------- |
| fileHandler   | can be used to write message to the file | filelog: True \| False, whether write to the file |
|               |                                          | filelevel: debug, info, warning, error, critical |
| streamHandler | can be used to write message to the shell | cmdlog: True \| False, whether write to the shell |
|               |                                          | cmdlevel: debug, info, warning, error, critical |
|               |                                          | colorful: if write colorful messages to shell |

