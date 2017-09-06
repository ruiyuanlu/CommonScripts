# encoding=utf-8
# @author ruiyuanlu
# @time 05:15:59 PM, 星期日 03, 九月 2017
# @function get gpa for each term, and draw a chart
# In the input file # is used to mark a start of new semester
# and the format of 'credit, grade' each line to record credit and grade.
# The library used to draw charts is pyecharts.

# input file path
file_path = r"C:\Users\luruiyuan\Desktop\score.txt"

class Term:
    
    # grade_section 左闭右开, 如, 71 分gpa为2.0, 72分则为2.3
    # 西交 4.3 GPA 算法
    grade_section = [0, 60, 64, 68, 72, 75, 78, 81, 85, 90, 95, 101]
    gpa_section =   [0, 1.3, 1.7, 2, 2.3, 2.7, 3, 3.3, 3.7, 4, 4.3]

    # # 人大 4.0 GPA 算法
    # grade_section = [0, 60, 63, 66, 70, 73, 76, 80, 83, 86, 90, 101]
    # gpa_section =   [0, 1, 1.3, 1.7, 2, 2.3, 2.7, 3, 3.3, 3.7, 4]

    # # 网上国外基于 4.0 GPA 的算法
    # grade_section = [0, 65, 67, 70, 73, 77, 80, 83, 87, 90, 93, 101]
    # gpa_section =   [0, 1, 1.3, 1.7, 2, 2.3, 2.7, 3, 3.3, 3.7, 4]

    # # 北大 4.0 GPA 的算法
    # grade_section = [0, 60, 64, 68, 72, 75, 79, 83, 86, 90, 101]
    # gpa_section =   [0, 1, 1.5, 2, 2.3, 2.7, 3, 3.3, 3.7, 4]

    # # 传说中的 WES 算法, 不知道对不对
    # grade_section = [0, 60, 70, 80, 90, 101]
    # gpa_section =   [0, 1, 2, 3, 4]

    def __init__(self, term):
        self.term = term
        self.gpa = .0
        self.classes = {}
        self.clas_num = 0
        self.total_credit = 0
        self.major_gpa = .0 # 主要科目gpa
    
    def _get_single_gpa(self, grade):
        for i in range(len(Term.grade_section)):
            if grade < Term.grade_section[i]:
                return Term.gpa_section[i-1]

    def add_class(self, name, grade, credit):
        self.clas_num += 1
        gpa = self._get_single_gpa(grade)
        self.classes['{0}_{1}_{2}'.format(self.term, self.clas_num, name)] = (grade, credit, gpa)
    
    def get_gpa(self):
        gpa, major_gpa, total_credit, major_credit = .0, .0, .0, .0
        for name, (g, c, s_gpa) in self.classes.items():
            gpa += s_gpa*abs(c)
            total_credit += abs(c)
            if c > 0:
                major_credit += c
                major_gpa += s_gpa*c
        gpa /= total_credit
        major_gpa /= major_credit
        self.gpa = gpa
        self.major_gpa = major_gpa
        self.total_credit = total_credit
        return self.gpa, self.total_credit

    def __str__(self):
        gpa, credit = self.get_gpa()
        grades = [('[{0} 学分:{1} GPA:{2}]'.format(name, c, s_gpa)) for name, (g, c, s_gpa) in self.classes.items()]
        return "\n".join(grades) + "\n第{term}学期 总学分{credit} GPA:{gpa}\n\n".format(term=self.term, gpa=self.gpa, credit=credit)

    @classmethod
    def draw_gpa_bar(self, terms):
        from pyecharts import Line
        from os import path

        avg_gpa = sum([ t.gpa*t.total_credit for t in terms]) / sum([t.total_credit for t in terms])
        line = Line("gpa折线图-平均{avg: .3f}".format(avg=avg_gpa))
        x = ["第{0}学期".format(t.term) for t in terms]
        y = [t.gpa for t in terms]
        y1 = [t.major_gpa for t in terms]
        line.add("全科GPA", x, y, is_stack=False, symbol='roundRect')
        line.add("主科GPA", x, y1, is_stack=False, symbol='triangle')
        save_path = path.join(__file__, "../GPA Line Figure.html")
        line.render(save_path) # save it to file
        return save_path

def main():
    terms = []
    path = file_path
    def read_file(path):
        cnt = 0
        with open(path) as f:
            lines = f.readlines()
            for l in lines:
                if l[0] == '#':
                    cnt += 1
                    terms.append(Term(cnt))
                else:
                    credit, grade = l.split(',')
                    grade = float(grade)
                    credit = float(credit)
                    terms[cnt-1].add_class("", grade, credit)
    
    def open_url(url):
        import webbrowser as web
        web.open(url)

    read_file(path)
    for t in terms:
        t.get_gpa()

    path = Term.draw_gpa_bar(terms)
    open_url(path)

if __name__ == '__main__':
    main()
