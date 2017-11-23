# coding=utf-8
# author: luruiyaun
# file: md_center.py
import re

def _center_title(line):
    """
    cneterize titiles in markdown.
    line: each titile in markwon file.
    """
    if line.startswith('#'):
        match_obj = re.match('#+ ', line) #  (?...) 非捕获分组，即空格不出现再结果中, match 只从字符串中匹配第一组结果
        if match_obj is not None:
            sep = len(match_obj.group())
            front, behind = line[:sep], line[sep:]
            if behind.find('<center>') == -1:
                return "%s <center>%s</center>" % (front, behind)
    return line

def _center_img(line):
    """
    centerize all <img> labels in the markdown file.
    """
    img_labels = set(re.findall('<img.+?>', line))
    for img in img_labels:
        line = line.replace(img, "<center>%s</center>" % img)
    return line

def centerize(file, encoding='utf-8', center_func=_center_title):
    """
    cneterize sth in markdown file.
    call different center_func to centerize different part
    """
    with open(file, 'r', encoding=encoding) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        lines[i] = center_func(line)
    with open(file, 'w', encoding=encoding) as f:
        f.write("".join(lines))

centerize('test.md',center_func=_center_img) # centerize img labels
centerize('就医情况说明.md') # centerize titles