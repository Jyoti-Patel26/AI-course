#1
def isEven(n):
    return n % 2 == 0

#2
def login(user, password):
    return True if user == "jyoti" and password == "ja220" else False

#3
def greet(name, hour):
    if hour<12 :
        return "Good Morning"
#4
def isAdult(age):
    return "Adult" if age>=18 else "Minor"

#5
def reverseStr(s):
    return s[::-1]

#6
def count(s):
    return len(s.split())

#7
def factorial(n):
    f = 1
    for i in range(1,n+1): f*=i
    return f

# 8
def repeatWord(s, times):
    return s*times
