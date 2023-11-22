seq = ['1', '2']
print(type(seq[0]))
exit()

from os import name
from string import Template

templ = Template("hello $name, i am your $relation")
string = templ.substitute(name = "TuRan", relation = "dady")
print(string)