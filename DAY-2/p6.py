
person={
    "firstname":"abs",
    "middlenae":"ass",
    "lastname":"xxy",
    "parent":{
        "father":{
            "name":"aaa"
        },
        "mother":{}
    },
    "qualification":["SSC","HSC","BCA"],
    "hobbies":{},
    "isAlive":True
}
print(person)
print(person["isAlive"])

person["isAlive"]=False
print(person["isAlive"])

print(type(person))
