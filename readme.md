# haven't finished:

empty

# updated:

| files                         | updated items                        | comments                                 |
| ----------------------------- | ------------------------------------ | ---------------------------------------- |
| git_push_to_new_repository.py | add cmd support to transfer path arg | git_push_to_new_repository.py now can support command line arguments as --path |
| git_push_to_new_repository.py | bug fix                              |                                          |
| logger.py                     | new colorful logger                  |                                          |

## file(s):

- git_push_to_new_repository.py: add cmd support to transfer path arg

  - eg: copy 3 files: **<u>execute_cmd.py+shellChoose.py+git_push_to_new_repository.py</u>**

     to the new repository directory, like "./test/new_repo",  then open the "**git_push_to_new_repository.py**" and modify <u>*user, email*</u> use you own user name and email address for gitHub. 

  - Save the file you edit, and use command to run the scripts from any where.

    for example:

  ```bash
  python ./test/new_repo/git_push_to_new_repository.py
  ```

  You will have to fill in 3 blanks: <u>*local_name, repo_name, commit*</u> 

  - local_name  : used to commit when you are working at local device
  - repo_name : used to set remote repository on the github.com
  - commit : used to explain what you've done in order to help others understand your work

- logger.py: a colorful logger file

  ```python
  ''' 
  simple examples for logger.py
  '''
  import logger
  log1 = logger.getLogger(loggername='test')
  log2 = logger.getLogger(loggername='test', filename='myApp.log', colorful=True)
  log2.info('Info test')
  log2.debug('debug message')
  log2.error('Error occured!')
  log2.exception('This is an exception')
  log.debug(log1 is log2)

  # set attibutes
  log.set_logger(backup_count=10)
  log.set_logger(loggername='new_test_logger')
  ```

  ​

# Finished

| files                         | functional description                   | comments                                 |
| ----------------------------- | ---------------------------------------- | ---------------------------------------- |
| git_push_to_new_repository.py | can be used to push github only for the first time |                                          |
| execute_cmd.py                | can execute multiple commands or singal commands for the same interface | But when tested on Windows 10, the process will be stucked by some commands like "cmd" or "winver". It seems like a bug for the "subprocess" module. |
|                               |                                          |                                          |

