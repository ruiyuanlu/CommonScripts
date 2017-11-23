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
        front, behind = re.split('(?: )+', line.strip()) # (?: )+ 用空格分隔, (?...) 非捕获分组，即空格不出现再结果中
        if behind.find('<center>') == -1:
            return "%s <center>%s</center>\n" % (front, behind)
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
centerize('test.md') # centerize titles