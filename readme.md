# haven't finished:

empty

# updated:

| files                         | updated items                        | comments                                 |
| ----------------------------- | ------------------------------------ | ---------------------------------------- |
| git_push_to_new_repository.py | add cmd support to transfer path arg | git_push_to_new_repository.py now can support command line arguments as --path |
| git_push_to_new_repository.py | bug fix                              |                                          |

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

# Finished

| files                         | functional description                   | comments                                 |
| ----------------------------- | ---------------------------------------- | ---------------------------------------- |
| git_push_to_new_repository.py | can be used to push github only for the first time |                                          |
| execute_cmd.py                | can execute multiple commands or singal commands for the same interface | But when tested on Windows 10, the process will be stucked by some commands like "cmd" or "winver". It seems like a bug for the "subprocess" module. |
|                               |                                          |                                          |

