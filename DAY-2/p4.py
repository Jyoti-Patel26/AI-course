
"""
file:
project:
author:
version:
release date:
"""

#sequential datatypes

"""
1.list -[]
        - mutable
        -any data type supported

2.tuple-()

        -immutable

3. set{}

        -store only unique value

4.dictionary-{key:value}
        - mutable
        -{key:value}
"""

num=[1,2,3]

cart=["phone","book",10,False,None,{1:"a"},num]

print(cart)

cart[0]="pc"

print(cart)
print(type(cart))

marks=(1,55,44,77,44,1)
print(marks)

#marks[0]=60
#print(marks)

print(type(marks))
