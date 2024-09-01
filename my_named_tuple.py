from argparse import Namespace
mydict = {"a":10,"b":"hello"}
ns = Namespace(**mydict)
print(ns.a)
