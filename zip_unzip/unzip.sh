#!/bin/bash
# unzip tar.gz files in the dir recursively.

# unzip file(as $1) into some dir(as $2)
# if $2 is empty, then it will use current dir
# else if $2 is not found, it will create a new dir
unzip()
{
	if [ $# -eq 1 ] # $# 表示函数接收到的变量个数
	then
		tar -xvf $1
	else
		if [ ! -x $2 ] # 判断当前目录是否存在并且具有可执行权限
		then
			mkdir $2
		fi
		tar -xvf $1 -C $2
	fi
}

# traverse all files in particular dir, and decide whether to unzip them
# $1 is used as the particular dir
# if $1 is not passed, then current dir is used
traverse_files()
{
	# local 用于声明局部变量，否则会默认当做全局变量，递归时会导致值被覆盖
	local root="./" # current root path
	local file_type="tar.gz" # tar.gz files only
	if [ $# -eq 1 ]
	then
		root=$1 # specific path
	fi

	# 如果有文件夹, 递归遍历
	for cur_dir in `ls $root`
	do
		cur_path=$root"./"$cur_dir
		# 递归地遍历子目录
		if [ -d $cur_path ]
		then
			traverse_files $cur_path
		fi
	done

	# 解压当前文件夹下的所有 file_type 类型的文件
	for file in `ls $root|grep $file_type`;do
	{		
		filename=$(basename $file)
		extention=${filename#*.} # 取两个点为后缀，得到 tar.gz, 如果改为 ##*. 则 extention 变为 gz
		name=${filename%%.*} # name 为去掉了 tar.gz 后缀的文件名，如果改为 %.* 则会保留 .tar
		unzip $file $root"./"$name
	}& # 并发执行解压缩
	done
}

traverse_files
wait # 等待所有进程结束
echo "congradulations! unzip finished!"
